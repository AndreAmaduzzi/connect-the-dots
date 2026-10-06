"""Ask a VLM what the finished picture shows.

The model sees a clean line drawing: the strokes printed in the puzzle plus the
solved path, black on white, without the page texture, digits or title. On the
raw photo the printed numbers dominate and the model guesses worse (e.g.
"a dinosaur" for a farmer with a pitchfork).

The guess is cached in a JSON file next to the video, so re-rendering is
reproducible and needs no GPU. When the AI is wrong, the correct answer is
stored next to it ("answer") and the reel shows both: the AI's guess is never
overwritten.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np

from .detect import THIN_STROKE, puzzle_ink

# Keep the prompt free of example answers or suggestive words: the model copies
# them into its guesses ("a giraffe eating leaves", "holding ..."). Asking for
# the whole scene avoids naming one object ("a trident") or part ("a beak").
PROMPT = (
    "This is a finished connect-the-dots picture. Describe the whole scene it shows: "
    "the main subject and what is happening. "
    "Reply with only a short phrase of at most 6 words, starting with an article."
)


def printed_strokes(rect_img: np.ndarray, r: float) -> np.ndarray:
    """Mask of the strokes printed in the puzzle, without digits or dots.

    Same idea as dot detection: an opening at ~0.7r removes thin digit strokes
    and keeps the thicker printed lines; small blobs (dots) are then dropped.
    """
    mask = puzzle_ink(rect_img)
    k = 2 * int(round(THIN_STROKE * r)) + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    strokes = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(strokes)
    h, w = mask.shape
    m = int(0.03 * max(h, w))
    x0, y0 = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    x1, y1 = x0 + stats[:, cv2.CC_STAT_WIDTH], y0 + stats[:, cv2.CC_STAT_HEIGHT]
    # Leftovers of the frame line (curled pages) hug the image border.
    at_border = (x0 < m) | (y0 < m) | (x1 > w - m) | (y1 > h - m)
    keep = (stats[:, cv2.CC_STAT_AREA] > 4 * np.pi * r * r) & ~at_border
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def render_drawing(rect_img: np.ndarray, path, r: float, erase_boxes=()) -> np.ndarray:
    """The finished picture as a clean black-on-white line drawing.
    `erase_boxes` (x0, y0, x1, y1): areas to blank in the printed strokes, e.g.
    the label boxes, to drop digits glued to printed lines."""
    out = np.full((*rect_img.shape[:2], 3), 255, np.uint8)
    strokes = printed_strokes(rect_img, r)
    for x0, y0, x1, y1 in erase_boxes:
        strokes[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 0
    out[strokes > 0] = 0
    t = max(3, int(round(1.2 * r)))
    pts = [tuple(np.round(p).astype(int)) for _, p in path]
    for a, b in zip(pts, pts[1:]):
        cv2.line(out, a, b, (0, 0, 0), t, cv2.LINE_AA)
    return out


FILL_MIN = 0.10          # enclosed area / drawing box needed to show a silhouette
FILL_BGR = (90, 160, 240)   # warm fill colour
FILL_NAME = "orange"        # ...which the model then mentions ("an orange bird"): dropped
LINE_BGR = (30, 30, 30)


def inside_mask(drawing: np.ndarray) -> np.ndarray:
    """Pixels enclosed by the drawn lines: everything the white background,
    flooded from the image border, cannot reach."""
    ink = (cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))   # close hairline gaps
    free = cv2.copyMakeBorder(1 - ink, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    h, w = free.shape
    cv2.floodFill(free, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    return (free[1:-1, 1:-1] == 1)


def guess_view(drawing: np.ndarray) -> np.ndarray:
    """What the VLM is shown: the drawing as a colour-filled silhouette.

    On thin line drawings Qwen3.5-9B often misses the subject (the 56-dot deer
    was "a dog attacking a tower" in 9/9 tries); filled, it reads shapes far
    better (12 pictures with known answers, 2 seeds: 10/24 -> 17/24 correct
    subjects; with the rule below, 38/48 over 4 seeds). A silhouette needs a
    real enclosed shape: when the outline encloses less than FILL_MIN of the
    drawing (an open outline, e.g. picture_2), the fill is only fragments and
    misleads, so the line drawing is used instead.
    """
    inside = inside_mask(drawing)
    lines = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY) < 128
    ys, xs = np.nonzero(lines)
    if not len(xs) or inside.sum() < FILL_MIN * (xs.max() - xs.min()) * (ys.max() - ys.min()):
        return drawing
    out = np.full_like(drawing, 255)
    out[inside] = FILL_BGR
    out[lines] = LINE_BGR
    return out


DANGLING = {"a", "an", "the", "of", "with", "and", "or", "on", "in", "at", "to", "for",
            "by", "from", "holding", "is", "are"}


def clean_guess(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"^(it'?s|it is|this is|maybe|probably|i think( it'?s)?)\s+", "", text,
                  flags=re.I)
    text = text.strip(" \"'.!?*")
    # The model sometimes runs past the 6 words asked for: allow a little more,
    # and never end on a dangling word ("a pair of hands holding a").
    words = text.split()[:8]
    while len(words) > 1 and words[-1].lower() in DANGLING:
        words.pop()
    return " ".join(words).lower() if words else "something"


def caption(guess: str) -> str:
    return f"Got it: {guess}!"


# Second step: what the subject is doing. Asked about the subject alone ("a deer"),
# not the first answer: repeating "a deer stands near a tree" anchors the model
# on "standing" (0/4 actions on the deer), the bare subject does not (4/4).
ACTION_PROMPT = ("This connect-the-dots picture shows {subject}. What is it doing, and with what? "
                 "Reply with only a short phrase of at most 8 words, starting with {subject}.")
ARTICLES = {"a", "an", "the"}
# Greedy decoding: the same puzzle always gets the same answer. Sampling
# (temperature 0.7) was as accurate on the 12 test pictures (10/12) but could
# change the answer between runs on the same input.
SAMPLE = False
BREAK = {"is", "are", "with", "at", "near", "by", "next", "in", "on", "under", "over", "and",
         "holding", "standing", "stands", "sitting", "sits", "from", "of", "to", "beside", "carrying"}


def subject_of(answer: str) -> str:
    """The noun phrase an answer starts with: 'a deer stands near a tree' -> 'a deer'."""
    out = []
    for w in answer.split():
        if out and (w.lower() in BREAK or w.lower().endswith("ing")):
            break
        out.append(w)
    return " ".join(out)


def same_subject(subject: str, answer: str) -> bool:
    """The action answer must start with the subject (it adds detail, never swaps
    it): 'a deer drinking...' for 'a deer', not 'playing with a trumpet'."""
    key = [w for w in subject.lower().split() if w not in ARTICLES]
    words = [w for w in answer.lower().split() if w not in ARTICLES]
    return bool(key) and any(k in words[:len(key) + 1] for k in key)


def fix_articles(text: str) -> str:
    """'an dragon' -> 'a dragon', 'a owl' -> 'an owl' (after dropping a word)."""
    w = text.split()
    for i in range(len(w) - 1):
        if w[i].lower() in ("a", "an"):
            w[i] = "an" if w[i + 1][0].lower() in "aeiou" else "a"
    return " ".join(w)


def with_article(answer: str, subject: str) -> str:
    first = subject.split()[0].lower() if subject else ""
    if answer.split()[0].lower() not in ARTICLES and first in ARTICLES:
        return f"{first} {answer}"
    return answer


def guess(drawing: np.ndarray, cache: Path, model_id: str, regen: bool = False,
          sample: bool = SAMPLE) -> str:
    """The VLM's guess for the finished drawing, cached in `cache`.

    Two questions on the silhouette (guess_view): the scene (PROMPT), then what
    its subject is doing (ACTION_PROMPT); the second answer is kept only if it
    starts with the same subject. Measured on 12 pictures with known answers
    (Qwen3.5-9B): sampled, 4 seeds, subject right 38/48 and the deer's action
    4/4; greedy (the default), 10/12, the deer "a deer drinking from a tree".
    """
    data = json.loads(cache.read_text()) if cache.exists() else {}
    if "guess" in data and not regen:
        return data["guess"]
    import torch
    from .vlm import QwenChat
    chat = QwenChat(model_id)
    view = guess_view(drawing)
    torch.manual_seed(0)
    first = clean_guess(chat.ask(view, PROMPT, sample=sample))
    subject = subject_of(first)
    raw = chat.ask(view, ACTION_PROMPT.format(subject=subject), max_new_tokens=80, sample=sample)
    second = clean_guess(raw)
    g = with_article(second, subject) if same_subject(subject, second) else first
    g = fix_articles(" ".join(w for w in g.split() if w.lower() != FILL_NAME))   # fill colour: not a fact
    data.update({"guess": g, "scene": first, "raw": raw, "model": model_id})   # keeps "answer"
    cache.write_text(json.dumps(data, indent=1) + "\n")
    return g


def record_answer(cache: Path, answer: str | None) -> str | None:
    """The human-given correct answer, kept in guess.json beside the AI's guess
    (`--answer` sets it, an empty string clears it; otherwise the stored one)."""
    data = json.loads(cache.read_text()) if cache.exists() else {}
    if answer is not None:
        if answer:
            data["answer"] = answer.strip().rstrip(".!")
        else:
            data.pop("answer", None)
        cache.write_text(json.dumps(data, indent=1) + "\n")
    return data.get("answer")
