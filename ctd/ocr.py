"""Read the numbers printed next to the dots.

Interchangeable backends return the same `Token` list:
- "paddle": PP-OCRv6 (PaddleOCR 3.x), a classic detector + recognizer, on CPU.
- "paddle-vl": PaddleOCR-VL-1.6, a small OCR VLM in text-spotting mode (see vlm.py).
- "qwen":   Qwen3.5 VLM asked for every number with its bounding box (see vlm.py).

Raw OCR strings are noisy in predictable ways: the dot is glued onto the number
("13°", "•24"), two labels merge ("1.50"), and digits come back as look-alike
letters ("I0"). `parse_numbers` turns a raw string into clean integer tokens.
"""

import os
import re
from dataclasses import dataclass

import numpy as np

# Characters OCR returns in place of digits. Only unambiguous ones: "!" and "i"
# look like a dot with a pre-drawn stroke above it, so they are treated as noise.
LOOKALIKES = str.maketrans({"I": "1", "l": "1", "|": "1", "O": "0", "o": "0", "D": "0",
                            "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "b": "6",
                            "g": "9", "q": "9"})
MAX_NUMBER = 999


@dataclass
class Token:
    value: int
    box: np.ndarray        # (x0, y0, x1, y1) in rectified image coordinates
    raw: str               # original OCR string this token came from
    score: float = 1.0

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def parse_numbers(raw: str, box, score: float = 1.0) -> list[Token]:
    """Split a raw OCR string into integer tokens, each with its share of the box.

    Assumes horizontal text: the box is divided along x in proportion to the
    character positions of each digit group.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    text = raw.translate(LOOKALIKES)
    if not text:
        return []
    char_w = (x1 - x0) / len(text)
    tokens = []
    for m in re.finditer(r"\d+", text):
        value = int(m.group())
        if not 0 <= value <= MAX_NUMBER:   # some puzzles start at 0
            continue
        bx = (x0 + m.start() * char_w, y0, x0 + m.end() * char_w, y1)
        tokens.append(Token(value, np.array(bx), raw, score))
    return tokens


def poly_to_box(poly) -> tuple[float, float, float, float]:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


class PaddleOCRBackend:
    name = "paddle"

    def __init__(self):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
        # Numbers are printed upright, and the 180-degree line classifier would
        # risk flipping 6 <-> 9. oneDNN is off: PaddlePaddle 3.3 crashes with it
        # on PP-OCRv6 ("ConvertPirAttribute2RuntimeAttribute not support").
        self.ocr = PaddleOCR(ocr_version="PP-OCRv6", use_doc_orientation_classify=False,
                             use_doc_unwarping=False, use_textline_orientation=False,
                             device="cpu", enable_mkldnn=False)

    # Lower detection thresholds than the defaults (0.3 / 0.6) pick up faint or
    # light-gray labels; on the samples this only added correct readings.
    DET = dict(text_det_thresh=0.2, text_det_box_thresh=0.4)

    def read(self, img: np.ndarray) -> list[Token]:
        res = self.ocr.predict(img, **self.DET)[0]
        tokens = []
        for raw, poly, score in zip(res["rec_texts"], res["rec_polys"], res["rec_scores"]):
            tokens += parse_numbers(raw, poly_to_box(poly), float(score))
        return tokens


def get_backend(name: str):
    if name == "paddle":
        return PaddleOCRBackend()
    if name == "qwen":
        from .vlm import QwenOCRBackend
        return QwenOCRBackend()
    if name == "paddle-vl":
        from .vlm import PaddleOCRVLBackend
        return PaddleOCRVLBackend()
    raise ValueError(f"unknown OCR backend: {name}")


def draw_tokens(img: np.ndarray, tokens: list[Token]) -> np.ndarray:
    import cv2
    vis = img.copy()
    for t in tokens:
        x0, y0, x1, y1 = t.box.astype(int)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 120, 0), 2)
        cv2.putText(vis, str(t.value), (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 220), 2, cv2.LINE_AA)
    return vis
