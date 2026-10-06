"""Phone video input: read frames, fix the orientation, pick the frame to solve.

Phone videos are often stored sideways (the file has no rotation flag, or the
player ignores it). The orientation is found with PaddleOCR's page-orientation
classifier on a sample frame; `rotate` overrides it.

The key frame is the one the puzzle is solved on, and where the reel freezes:
the sharpest frame in which the puzzle frame is found and large.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np

from .rectify import find_frame

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass
class Footage:
    frames: list[np.ndarray]   # upright BGR frames
    fps: float
    rotation: int              # degrees clockwise applied to the raw frames
    key: int                   # index of the key frame


def read_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot read video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames, fps


def rotate(img: np.ndarray, degrees_cw: int) -> np.ndarray:
    code = ROTATIONS[degrees_cw % 360]
    return img if code is None else cv2.rotate(img, code)


def detect_rotation(frame: np.ndarray) -> int:
    """Degrees to rotate clockwise so that the text is upright."""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import DocImgOrientationClassification
    model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori",
                                            device="cpu", enable_mkldnn=False)
    label = int(model.predict(frame)[0]["label_names"][0])   # how the content is turned
    return (360 - label) % 360


def frame_score(frame: np.ndarray) -> float:
    """Puzzle frame area (fraction of the image) x sharpness inside it; 0 if no frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    quad = find_frame(gray)
    if quad is None:
        return 0.0
    h, w = gray.shape
    area = cv2.contourArea(quad.astype(np.float32)) / (h * w)
    # Fully inside the image (a frame cut by the border would be a wrong quad).
    if quad[:, 0].min() < 2 or quad[:, 1].min() < 2 or quad[:, 0].max() > w - 3 or quad[:, 1].max() > h - 3:
        return 0.0
    x0, y0 = quad.min(axis=0).astype(int)
    x1, y1 = quad.max(axis=0).astype(int)
    sharp = cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var()
    return float(area * np.sqrt(sharp))


def load(path: str, rotation: int | str = "auto", step: int = 3) -> Footage:
    raw, fps = read_frames(path)
    if rotation == "auto":
        rotation = detect_rotation(raw[len(raw) // 2])
    frames = [rotate(f, int(rotation)) for f in raw]
    scores = [frame_score(f) if i % step == 0 or i == len(frames) - 1 else 0.0
              for i, f in enumerate(frames)]
    key = int(np.argmax(scores))
    if scores[key] == 0:
        key = len(frames) - 1   # no frame found anywhere: solve on the last one
    return Footage(frames, fps, int(rotation), key)
