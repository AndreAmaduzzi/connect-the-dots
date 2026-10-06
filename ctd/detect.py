"""Detect the printed dots in a rectified puzzle image.

Dots are small, solid, round blobs. Pipeline:
1. Estimate the dot radius r from isolated round blobs (dots form a clear size
   mode, well above paper-texture specks).
2. Morphological opening with a disk ~1.3r: thin digit strokes disappear, dots
   and thick pre-drawn strokes survive.
3. Keep blobs that are (a) as thick as a dot at their centre (distance-transform
   peak close to r) and (b) not part of a stroke: on a ring around the blob, at
   most one thick stroke leaves it (a dot may touch the start of a printed line;
   a point in the middle of a line has 2 crossings, a junction 3+). The ring is
   read on an eroded mask, so digits printed right next to a dot are ignored.
   Blobs a few dots big (dots printed touching each other) are split at their
   distance-transform peaks.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .rectify import ink_mask

MIN_THICKNESS = 0.83    # distance-transform peak / r for a solid dot
MAX_CROSSINGS = 1       # thick strokes allowed to leave a dot
THIN_STROKE = 0.35      # erosion radius / r that removes digit strokes


@dataclass
class Dots:
    centers: np.ndarray   # Nx2 float, (x, y) in rectified image coordinates
    radius: float         # estimated typical dot radius in pixels
    score: np.ndarray     # N, thickness ratio (higher = more dot-like)
    # N ints: for dots added by recover.py, the number whose label they were
    # found next to (only that label may take them); -1 for detected dots.
    owner: np.ndarray | None = None

    def owners(self) -> np.ndarray:
        return self.owner if self.owner is not None else np.full(len(self.centers), -1)


def _components(mask: np.ndarray):
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return stats[1:], centroids[1:]


def _roundness(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    aspect = np.minimum(w, h) / np.maximum(w, h)
    fill = stats[:, cv2.CC_STAT_AREA] / (w * h)   # ~0.785 for a filled disk
    return aspect, fill


def puzzle_ink(img: np.ndarray, border_frac: float = 0.012) -> np.ndarray:
    """Ink mask without the frame line, with tiny holes filled.

    Only long components touching the border are removed (frame lines, also
    when curled); dots near the edge of an unframed puzzle are kept.
    """
    mask = ink_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = mask.shape
    b = max(2, int(border_frac * max(h, w)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x0, y0 = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    x1, y1 = x0 + stats[:, cv2.CC_STAT_WIDTH], y0 + stats[:, cv2.CC_STAT_HEIGHT]
    at_border = (x0 < b) | (y0 < b) | (x1 > w - b) | (y1 > h - b)
    long_ = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) > 0.25 * min(h, w)
    drop = at_border & long_
    drop[0] = False
    mask[drop[labels]] = 0
    return mask


def estimate_dot_radius(mask: np.ndarray) -> float:
    """Typical radius of the isolated, filled, round components in the mask."""
    stats, _ = _components(mask)
    aspect, fill = _roundness(stats)
    area = stats[:, cv2.CC_STAT_AREA]
    # Paper texture produces thousands of round specks with r < ~3 px at the
    # rectified scale (long side >= 1600); dots form a separate mode above that.
    min_r = max(mask.shape) / 600
    round_ = (aspect > 0.7) & (fill > 0.6) & (area > np.pi * min_r ** 2)
    if round_.sum() < 5:
        raise RuntimeError("could not estimate dot size: too few round blobs")
    return float(np.sqrt(np.median(area[round_]) / np.pi))


def ring_crossings(mask: np.ndarray, x: float, y: float, radius: float,
                   min_arc: float, n: int = 120) -> int:
    """Number of ink runs at least `min_arc` px long on a circle around (x, y).

    The arc-length threshold ignores thin digit strokes printed next to a dot
    and only counts thick pre-drawn lines.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.clip(np.round(x + radius * np.cos(t)).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(y + radius * np.sin(t)).astype(int), 0, mask.shape[0] - 1)
    on = mask[ys, xs] > 0
    if on.all():
        return n   # buried in a solid region
    on = np.roll(on, -int(np.argmin(on)))   # start on background so runs don't wrap
    step = 2 * np.pi * radius / n
    runs, length = 0, 0
    for v in np.append(on, False):
        if v:
            length += 1
        elif length:
            runs += length * step >= min_arc
            length = 0
    return runs


def _peaks(dt: np.ndarray, region: np.ndarray, r: float) -> list[tuple[float, float]]:
    """Centres of dot-thick local maxima of the distance transform in `region`,
    at least 1.5r apart."""
    d = np.where(region, dt, 0).astype(np.float32)
    ys, xs = np.nonzero(d >= MIN_THICKNESS * r)
    order = np.argsort(-d[ys, xs])
    peaks: list[tuple[float, float]] = []
    for j in order:
        x, y = float(xs[j]), float(ys[j])
        if all((x - px) ** 2 + (y - py) ** 2 > (1.5 * r) ** 2 for px, py in peaks):
            peaks.append((x, y))
    return peaks


def detect_dots(img: np.ndarray) -> Dots:
    mask = puzzle_ink(img)
    r = estimate_dot_radius(mask)
    k = max(3, int(round(1.3 * r)) | 1)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Thick ink only: erosion wipes out thin digit strokes (which may run
    # tangent to the ring and look long) but keeps pre-drawn lines.
    e = 2 * int(round(THIN_STROKE * r)) + 1
    thick = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    dot_area = np.pi * r * r
    # Pass 1: dot-thick candidates. Pass 2: ring test, with the other solid
    # round dots blanked so that dots printed close together (dense puzzles)
    # don't count as strokes leaving each other.
    cands = []   # (x, y, thickness, round_and_solid)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not 0.3 * dot_area < a < 8.0 * dot_area:
            continue
        pts = [tuple(centroids[i])]
        split = False
        if a > 1.6 * dot_area:
            # Possibly dots printed touching each other (consecutive numbers often
            # are): one candidate per thickness peak.
            peaks = _peaks(dt, labels == i, r)
            if len(peaks) >= 2:
                pts, split = peaks, True
            elif a > 3.0 * dot_area:
                continue
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        solid = (split or (a < 1.6 * dot_area and min(bw, bh) / max(bw, bh) > 0.75
                                  and a / (bw * bh) > 0.65))
        for x, y in pts:
            xi, yi, w = int(round(x)), int(round(y)), max(1, int(r * 0.6))
            thickness = dt[yi - w:yi + w + 1, xi - w:xi + w + 1].max() / r
            if thickness >= MIN_THICKNESS:
                cands.append((x, y, thickness, solid))

    others = np.zeros_like(thick)
    for x, y, _, solid in cands:
        if solid:
            cv2.circle(others, (int(round(x)), int(round(y))), int(round(1.3 * r)), 255, -1)
    centers, scores = [], []
    for x, y, thickness, _ in cands:
        own = np.zeros_like(thick)
        cv2.circle(own, (int(round(x)), int(round(y))), int(round(1.3 * r)), 255, -1)
        ring_mask = thick & ~(others & ~own)
        if ring_crossings(ring_mask, x, y, 2.2 * r, min_arc=0.4 * r) > MAX_CROSSINGS:
            continue
        centers.append((x, y))
        scores.append(thickness)
    return Dots(np.array(centers, dtype=np.float64).reshape(-1, 2), r, np.array(scores))


def draw_dots(img: np.ndarray, dots: Dots) -> np.ndarray:
    vis = img.copy()
    for x, y in dots.centers:
        cv2.circle(vis, (int(x), int(y)), int(dots.radius * 2.5), (0, 0, 255), 2)
    return vis
