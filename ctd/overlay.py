"""Drawing the solution on the original photo, and framing the puzzle.

Lines are drawn in rectified coordinates and mapped back to the photo through
the inverse homography, so they follow the page's perspective.
"""

import cv2
import numpy as np

from .rectify import Rectified

LINE_BGR = (40, 40, 230)         # red pen


def crop_window(img_shape, corners: np.ndarray, size: tuple[int, int],
                margin: float = 0.05, caption_room: float = 0.16) -> tuple[int, int, int, int]:
    """Crop (x0, y0, w, h) of the photo with the output aspect that contains the
    puzzle plus a margin, and keeps the top `caption_room` of the frame free of
    the puzzle for the caption. May extend past the photo; the caller pads."""
    out_w, out_h = size
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    pw = (x1 - x0) * (1 + 2 * margin)
    ph = (y1 - y0) * (1 + 2 * margin) / (1 - caption_room)
    aspect = out_w / out_h
    w = max(pw, ph * aspect)
    h = w / aspect
    cx = (x0 + x1) / 2
    # Centre the puzzle in the part of the frame below the caption band.
    cy = (y0 + y1) / 2 - caption_room * h / 2
    H, W = img_shape[:2]
    # Prefer staying inside the photo: shift rather than pad when possible,
    # but never so far that the caption band would cover the puzzle.
    left = np.clip(cx - w / 2, 0, max(0, W - w)) if w <= W else cx - w / 2
    top = cy - h / 2
    if h <= H:
        top = np.clip(top, max(0, y1 + margin * (y1 - y0) - h), y0 - caption_room * h)
        top = np.clip(top, 0, H - h)
    return int(round(left)), int(round(top)), int(round(w)), int(round(h))


def path_in_photo(rect: Rectified, path) -> np.ndarray:
    return rect.to_original(np.array([p for _, p in path]))


def draw_solution(photo: np.ndarray, rect: Rectified, path) -> np.ndarray:
    """The original photo with the whole path drawn on it (full resolution)."""
    out = photo.copy()
    t = max(3, int(max(photo.shape[:2]) / 400))
    pts = np.round(path_in_photo(rect, path)).astype(int)
    for a, b in zip(pts, pts[1:]):
        cv2.line(out, tuple(a), tuple(b), LINE_BGR, t, cv2.LINE_AA)
    return out
