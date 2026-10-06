"""Second pass: use what was found to recover what was missed.

The global dot detector uses one dot size for the whole image, and the OCR
reads the full page once. Both miss things that are easy to find once you know
where to look:

- every number read has its dot right next to it: search there for the
  thickest compact blob (handles dots smaller than usual, oval, or touching
  their digits);
- every dot left without a number has its label right next to it: crop around
  it, upscale, and read again (handles small or faint labels).

Recovered dots/labels are only candidates: the global assignment in match.py
still decides, but a recovered dot may only go to the label it was found next
to (a stray blob found for one label must not steal another label's match).
"""

import cv2
import numpy as np

from .detect import Dots, puzzle_ink
from .match import GATE, box_distance
from .ocr import Token

SEARCH = 3.0          # how far from a label to look for its dot, in dot radii
CROP = 7.0            # half-size of the re-read crop around a dot, in dot radii
UPSCALE = 3


def _stroke_halfwidth(dt: np.ndarray, mask: np.ndarray, tokens: list[Token],
                      dots: Dots) -> float:
    """Typical half-width of the digit strokes, measured inside the label boxes
    (away from the dots, which some boxes include)."""
    away = np.ones(mask.shape, bool)
    for x, y in dots.centers:
        cv2.circle(away.view(np.uint8), (int(x), int(y)), int(1.6 * dots.radius), 0, -1)
    vals = []
    for t in tokens:
        x0, y0, x1, y1 = np.round(t.box).astype(int)
        sel = (mask[y0:y1, x0:x1] > 0) & away[y0:y1, x0:x1]
        if sel.sum() > 10:
            vals.append(np.percentile(dt[y0:y1, x0:x1][sel], 90))
    return float(np.median(vals)) if vals else 0.3 * dots.radius


def _compact_peak(dt: np.ndarray, x: int, y: int) -> bool:
    """True if the ink around this distance-transform peak is a blob, not a
    stretch of line: the core (dt >= 0.7 * peak) must be small. (0.7 rather
    than lower so the core of a dot printed against bold digits stops at the
    neck between them.)"""
    rho = dt[y, x]
    s = int(4 * rho) + 2
    y0, x0 = max(0, y - s), max(0, x - s)
    win = (dt[y0:y + s + 1, x0:x + s + 1] >= 0.7 * rho).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(win)
    lab = labels[y - y0, x - x0]
    return lab > 0 and max(stats[lab, cv2.CC_STAT_WIDTH], stats[lab, cv2.CC_STAT_HEIGHT]) <= 2 * rho


def _isolated_blob(t: Token, labels, stats, cents, dt, centers, r, min_rho, window):
    """The closest isolated, round, solid ink blob near the label (a dot on its
    own), or None. Preferred over the thickest spot: with a thick pen the
    thickest spot can be part of a digit."""
    xa, ya, xb, yb = window
    ids = np.unique(labels[ya:yb, xa:xb])
    best, best_d = None, SEARCH * r
    for i in ids[ids > 0]:
        x, y, bw, bh, a = stats[i]
        if not (np.pi * (0.3 * r) ** 2 <= a <= 3 * np.pi * r * r):
            continue
        if min(bw, bh) / max(bw, bh) < 0.5 or a / (bw * bh) < 0.5:
            continue   # elongated (a "1", a stroke) or hollow (a "0")
        bx0, by0, bx1, by1 = t.box
        inside = (max(0, min(x + bw, bx1) - max(x, bx0)) * max(0, min(y + bh, by1) - max(y, by0)))
        if inside > 0.6 * bw * bh:
            continue   # mostly inside the label's own box: a glyph (a bold "1"), not a dot
        cx, cy = cents[i]
        if centers and np.min(np.hypot(*(np.array(centers) - (cx, cy)).T)) < 1.5 * r:
            continue   # already a dot
        peak = dt[labels == i].max()
        if peak < min_rho:
            continue   # as thin as a pen stroke: a digit, not a dot
        d = box_distance(np.array([[cx, cy]]), t.box)[0]
        if d < best_d:
            best, best_d = (cx, cy, peak), d
    return best


def recover_dots(img: np.ndarray, dots: Dots, tokens: list[Token],
                 orphans: list[Token]) -> tuple[Dots, int]:
    """Add the best dot candidate next to each label in `orphans` (see
    labels_to_check). `tokens` (all labels) are used to measure the digit
    stroke width."""
    r = dots.radius
    mask = puzzle_ink(img)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    hw = _stroke_halfwidth(dt, mask, tokens, dots)
    # Thicker than a digit stroke, and at least a third of a typical dot.
    min_rho = max(1.6 * hw, 0.33 * r)

    _, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centers = [tuple(c) for c in dots.centers]
    scores = list(dots.score)
    owner = list(dots.owners())
    added = 0
    h, w = mask.shape
    for t in orphans:
        x0, y0, x1, y1 = t.box
        m = SEARCH * r
        xa, ya = int(max(0, x0 - m)), int(max(0, y0 - m))
        xb, yb = int(min(w, x1 + m + 1)), int(min(h, y1 + m + 1))
        best = _isolated_blob(t, labels, stats, cents, dt, centers, r, min_rho, (xa, ya, xb, yb))
        win = dt[ya:yb, xa:xb]
        ys, xs = np.nonzero(win >= min_rho) if best is None else ([], [])
        for j in np.argsort(-win[ys, xs]):
            x, y = int(xs[j]) + xa, int(ys[j]) + ya
            if centers and np.min(np.hypot(*(np.array(centers) - (x, y)).T)) < 1.5 * r:
                continue
            if _compact_peak(dt, x, y):
                best = (x, y, dt[y, x])
                break
        if best:
            centers.append((float(best[0]), float(best[1])))
            scores.append(best[2] / r)
            owner.append(t.value)
            added += 1
    return Dots(np.array(centers, dtype=np.float64).reshape(-1, 2), r, np.array(scores),
                np.array(owner, int)), added


FAR = 2.5   # x the puzzle's median label-to-dot distance


def labels_to_check(tokens: list[Token], path, r: float) -> list[Token]:
    """Labels worth a dot search:

    - readings of a number that is missing from the path (left without a dot);
    - a number read exactly once whose assigned dot is unusually far for this
      puzzle (more than FAR x the median label-to-dot distance, within the
      matching gate): its own dot was probably missed and it took a neighbour's.

    Duplicate readings (the "3" misread from "35") are not evidence of a missed
    dot and are left alone.
    """
    placed = dict(path)
    counts: dict[int, int] = {}
    for t in tokens:
        counts[t.value] = counts.get(t.value, 0) + 1
    dist = {id(t): box_distance(np.array([placed[t.value]]), t.box)[0]
            for t in tokens if t.value in placed}
    typical = np.median(list(dist.values())) if dist else r
    limit = max(r, FAR * typical)
    out = []
    for t in tokens:
        if t.value not in placed:
            out.append(t)
        elif counts[t.value] == 1 and limit < dist[id(t)] <= GATE * r:
            out.append(t)
    return out


def recover_labels(img: np.ndarray, dots: Dots, dot_indices: list[int], ocr,
                   tokens: list[Token], missing: list[int]) -> tuple[list[Token], int]:
    """Re-read an upscaled crop around each dot that has no number. Only
    readings of numbers still `missing` are kept, so a re-read can fill a gap
    but never disturb a match that already works."""
    r = dots.radius
    h, w = img.shape[:2]
    new = []
    for i in dot_indices:
        x, y = dots.centers[i]
        s = CROP * r
        xa, ya, xb, yb = (int(max(0, x - s)), int(max(0, y - s)),
                          int(min(w, x + s)), int(min(h, y + s)))
        crop = cv2.resize(img[ya:yb, xa:xb], None, fx=UPSCALE, fy=UPSCALE,
                          interpolation=cv2.INTER_CUBIC)
        for t in ocr.read(crop):
            t.box = t.box / UPSCALE + [xa, ya, xa, ya]
            if t.value not in missing:
                continue
            if box_distance(np.array([[x, y]]), t.box)[0] > GATE * r:
                continue
            dup = any(k.value == t.value and box_distance(k.center[None], t.box)[0] < r
                      for k in tokens + new)
            if not dup:
                new.append(t)
    return tokens + new, len(new)
