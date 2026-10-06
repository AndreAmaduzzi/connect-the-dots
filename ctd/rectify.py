"""Find the puzzle frame in a photo and warp it to a flat, axis-aligned image.

The "Pista Cifrata" is always printed inside a rectangular frame, so we look for
the largest convex quadrilateral made of dark ink rather than the page edge
(which is often cut off or curled in phone photos).
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Rectified images are scaled so their long side is at least this many pixels:
# small web images get upsampled, which helps OCR on tiny digits.
MIN_LONG_SIDE = 1600


@dataclass
class Rectified:
    image: np.ndarray          # warped BGR image of the puzzle area
    H: np.ndarray              # 3x3 homography: original photo -> rectified
    corners: np.ndarray        # 4x2 frame corners in the original photo (tl, tr, br, bl)
    found_frame: bool          # False if we fell back to the whole image

    def to_original(self, pts: np.ndarray) -> np.ndarray:
        """Map Nx2 points from rectified coordinates back to the original photo."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners ordered tl, tr, br, bl."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of dark ink, robust to uneven lighting."""
    # Window from the long side: on a wide, short image a window based on the
    # short side is barely wider than a dot, and solid dots lose their centre.
    block = max(15, (max(gray.shape) // 30) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 10)


def find_frame(gray: np.ndarray, work_size: int = 1200) -> np.ndarray | None:
    """Return the 4 corners of the puzzle frame, or None if not found.

    Detection runs on a copy resized to `work_size` so thresholds and kernels
    behave the same for a 350 px web image and a 4000 px phone photo.
    """
    s = work_size / max(gray.shape)
    small = cv2.resize(gray, None, fx=s, fy=s,
                       interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    corners = _find_frame(small)
    return None if corners is None else corners / s


def _find_frame(gray: np.ndarray) -> np.ndarray | None:
    h, w = gray.shape
    mask = ink_mask(gray)
    # Close small gaps in the printed frame line.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * h * w or area > 0.98 * h * w:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        # A frame is roughly rectangular: its area fills most of its min-area box.
        (_, _), (bw, bh), _ = cv2.minAreaRect(approx)
        if area / (bw * bh + 1e-6) < 0.85:
            continue
        if area > best_area:
            best, best_area = approx, area
    return None if best is None else order_corners(best)


def rectify(img: np.ndarray, corners: np.ndarray | None = None) -> Rectified:
    """Warp the puzzle area flat. `corners` (tl, tr, br, bl) overrides detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = True
    if corners is None:
        corners = find_frame(gray)
    if corners is None:
        found = False
        h, w = gray.shape
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    corners = order_corners(corners)

    tl, tr, br, bl = corners
    out_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    out_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    scale = max(1.0, MIN_LONG_SIDE / max(out_w, out_h))
    out_w, out_h = int(round(out_w * scale)), int(round(out_h * scale))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    return Rectified(warped, H, corners, found)
