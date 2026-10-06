"""Photo or phone video -> results folder. Used by run.py and app.py.

results/<sample>/
  input.<ext>                 the photo (or the video)
  keyframe.jpg                for a video: the upright frame the puzzle was solved on
  gt.json                     ground truth, if data/gt/<sample>.json exists
  drawing.png                 the finished picture as a clean line drawing
  guess_input.png             what the VLM is shown (the drawing as a filled silhouette)
  solved.jpg                  the photo with the solution drawn on it
  guess.json                  the VLM's guess (cached), and your correction if given (--answer)
  reel_<aspect>.mp4           the reel (photo or video input), see reel.py
  thumbnail_<aspect>.jpg      the reel frame with drawing + guess, for LinkedIn's custom thumbnail
  report.txt                  matching problems + accuracy against ground truth
  debug/                      intermediate images (with debug=True)
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .overlay import draw_solution
from .detect import Dots, detect_dots, draw_dots
from .guess import guess, guess_view, record_answer, render_drawing
from .match import Match, draw_match, match
from .ocr import Token, draw_tokens
from .rectify import rectify
from .recover import labels_to_check, recover_dots, recover_labels
from .reel import TAGLINE, Pace, render_reel
from .video import VIDEO_EXTS, Footage, load as load_video, rotate
from .report import Score, format_table, load_gt, score


@dataclass
class Options:
    aspects: list[str] = field(default_factory=lambda: ["9x16"])
    last: int | None = None               # highest number, if known
    corners: np.ndarray | None = None     # frame corners tl tr br bl, if detection fails
    guess: bool = True
    guess_model: str = "Qwen/Qwen3.5-9B"
    regen_guess: bool = False
    debug: bool = False
    rotation: int | str = "auto"          # degrees clockwise, or "auto" (videos: detected; photos: EXIF)
    credit: str | None = None             # video input: extra line on the reel's end card
    tagline: str = TAGLINE                # video input: line under the guess on the end card
    answer: str | None = None             # video input: what the picture really shows, if the AI is wrong
    start: float = 0.0                    # video input: seconds of footage to skip in the reel


@dataclass
class Located:
    dots: Dots
    tokens: list[Token]
    match: Match
    recovered_dots: int
    recovered_labels: int


def locate(img: np.ndarray, ocr, last: int | None = None) -> Located:
    """Dots + numbers -> ordered path, with the recovery pass (see recover.py):
    detect, read, match; look for missed dots next to labels that need one;
    re-read around dots still without a number; match again, and only then
    fill what is still missing by elimination."""
    dots = detect_dots(img)
    tokens = ocr.read(img)
    m = match(dots, tokens, last, eliminate=False)
    dots, nd = recover_dots(img, dots, tokens, labels_to_check(tokens, m.path, dots.radius))
    if nd:
        m = match(dots, tokens, last, eliminate=False)
    nl = 0
    if m.missing and m.unmatched_dots:
        tokens, nl = recover_labels(img, dots, m.unmatched_dots, ocr, tokens, m.missing)
    # Last resort, only now: place what is still missing by geometry.
    m = match(dots, tokens, last)
    return Located(dots, tokens, m, nd, nl)


@dataclass
class Result:
    name: str
    match: Match
    guess: str | None
    score: Score | None
    seconds: float


def solve(photo_path: Path, out: Path, ocr, opts: Options, log=print) -> Result:
    t0 = time.time()
    lap = lambda msg: log(f"  [{time.time() - t0:5.1f}s] {msg}")
    out.mkdir(parents=True, exist_ok=True)
    footage = None
    if photo_path.suffix.lower() in VIDEO_EXTS:
        footage = load_video(str(photo_path), opts.rotation)
        photo = footage.frames[footage.key]
        cv2.imwrite(str(out / "keyframe.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lap(f"video: {len(footage.frames)} frames, rotated {footage.rotation}°, "
            f"solving on frame {footage.key}")
    else:
        photo = cv2.imread(str(photo_path))   # applies the phone's EXIF orientation
        if photo is None:
            raise ValueError(f"cannot read {photo_path}")
        if opts.rotation != "auto":
            photo = rotate(photo, int(opts.rotation))
        # A photo is a one-frame "video": the reel holds it, then scans and draws.
        footage = Footage([photo], 30.0, 0, 0)
    shutil.copy(photo_path, out / f"input{photo_path.suffix.lower()}")
    gt = load_gt(photo_path)
    if gt:
        (out / "gt.json").write_text(json.dumps(gt, indent=1) + "\n")

    rect = rectify(photo, opts.corners)
    if not rect.found_frame:
        lap("no puzzle frame found: using the whole image (pass corners if this is wrong)")
    last = opts.last or (max(int(k) for k in gt["dots"]) if gt else None)
    loc = locate(rect.image, ocr, last)
    dots, tokens, m = loc.dots, loc.tokens, loc.match
    lap(f"{len(dots.centers)} dots (r={dots.radius:.1f}px, {loc.recovered_dots} recovered), "
        f"{len(tokens)} numbers ({loc.recovered_labels} re-read)")
    for line in m.summary().splitlines():
        log("    " + line)
    if m.missing:
        log(f"    INCOMPLETE: {len(m.missing)} numbers could not be placed, so the drawing "
            f"jumps over them: {m.missing}")

    report = [m.summary()]
    if m.missing:
        report.insert(0, f"INCOMPLETE: {len(m.missing)} numbers could not be placed: {m.missing}")
    sc = None
    if gt:
        sc = score(photo_path.stem, gt, rect, dots, tokens, m)
        report += ["", "Against ground truth:", format_table([sc])]
    (out / "report.txt").write_text("\n".join(report) + "\n")

    if opts.debug:
        d = out / "debug"
        d.mkdir(exist_ok=True)
        vis = photo.copy()
        cv2.polylines(vis, [rect.corners.astype(int)], True, (0, 0, 255), 4)
        cv2.imwrite(str(d / "0_frame.jpg"), vis)
        cv2.imwrite(str(d / "1_rectified.jpg"), rect.image)
        cv2.imwrite(str(d / "2_dots.jpg"), draw_dots(rect.image, dots))
        cv2.imwrite(str(d / "3_numbers.jpg"), draw_tokens(rect.image, tokens))
        cv2.imwrite(str(d / "4_matches.jpg"), draw_match(rect.image, dots, tokens, m))

    if len(m.path) < 2:
        lap("not enough numbers matched to draw anything")
        return Result(photo_path.stem, m, None, sc, time.time() - t0)

    path = m.drawing_path()   # closed: ends with the segment last -> first
    drawing = render_drawing(rect.image, path, dots.radius)
    cv2.imwrite(str(out / "drawing.png"), drawing)
    cv2.imwrite(str(out / "solved.jpg"), draw_solution(photo, rect, path),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    g = None
    if opts.guess:
        cv2.imwrite(str(out / "guess_input.png"), guess_view(drawing))   # what the VLM sees
        g = guess(drawing, out / "guess.json", opts.guess_model, opts.regen_guess)
        lap(f"guess: {g!r}")
    answer = record_answer(out / "guess.json", opts.answer)
    if answer:
        lap(f"correct answer (from you): {answer!r}")
    # The end card shows the drawing without digits glued to printed lines.
    clean = render_drawing(rect.image, path, dots.radius, [t.box for t in tokens])
    pts = rect.to_original(np.array([p for _, p in path]))
    for aspect in opts.aspects:
        secs = render_reel(footage.frames, footage.key, rect.corners, pts, len(m.expected),
                           m.closes, clean, g, str(out / f"reel_{aspect}.mp4"), aspect,
                           credit=opts.credit, src_fps=footage.fps, tagline=opts.tagline,
                           thumbnail_path=str(out / f"thumbnail_{aspect}.jpg"), answer=answer,
                           pace=Pace(start=opts.start))
    lap(f"reels: {', '.join(opts.aspects)} ({secs:.1f}s each)")
    return Result(photo_path.stem, m, g, sc, time.time() - t0)


def write_index(results_dir: Path, answers: dict[str, str]) -> Path:
    """results/README.md: one row per sample folder (also those from earlier runs)."""
    rows = ["| sample | numbers placed | vs ground truth | VLM guess | it actually shows |",
            "|---|---|---|---|---|"]
    for d in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        rep = (d / "report.txt").read_text().splitlines() if (d / "report.txt").exists() else []
        placed = next((l.removeprefix("numbers placed: ") for l in rep
                       if l.startswith("numbers placed: ")), "?")
        if any(l.startswith("INCOMPLETE") for l in rep):
            placed = f"⚠ {placed} (incomplete)"
        vs_gt = "–"
        for line in rep:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 5 and cells[0] == d.name:
                vs_gt = f"{cells[4]} placed correctly"
        g = json.loads((d / "guess.json").read_text())["guess"] if (d / "guess.json").exists() else "–"
        video = f"[{g}](./{d.name}/animation_1x1.mp4)" if (d / "animation_1x1.mp4").exists() else g
        rows.append(f"| [{d.name}](./{d.name}/) | {placed} | {vs_gt} | {video} | "
                    f"{answers.get(d.name) or '?'} |")
    path = results_dir / "README.md"
    path.write_text("# Results\n\nGenerated by `run.py`. Each folder holds the input, ground "
                    "truth (if any), `drawing.png`, `solved.jpg`, `guess.json`, the videos "
                    "and `report.txt`.\n\n" + "\n".join(rows) + "\n")
    return path
