"""Local vision-language models (Hugging Face transformers, one GPU).

- PaddleOCRVLBackend: PaddleOCR-VL-1.6 (0.9B) in "Spotting:" mode, which
  returns every text line with a 4-point box as <|LOC_n|> tokens (0-1000 scale).
- QwenOCRBackend: Qwen3.5 asked for {"bbox_2d", "label"} JSON per number, on
  overlapping tiles (asking for ~50 tiny objects at once makes it degenerate).
- QwenChat: free-form image + prompt -> text, used by guess.py.
"""

import json
import re
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .ocr import Token, parse_numbers, poly_to_box

QWEN_OCR_MODEL = "Qwen/Qwen3.5-9B"
GPU_RESERVE = int(2.5 * 2**30)   # bytes left free per GPU for activations
PADDLE_VL_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


@lru_cache(maxsize=2)
def _load(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    # Spread large models over the visible GPUs, using what is actually free on
    # each (other jobs may share them) minus room for the computation itself.
    max_memory = {i: max(0, torch.cuda.mem_get_info(i)[0] - GPU_RESERVE)
                  for i in range(torch.cuda.device_count())}
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", max_memory=max_memory).eval()
    return model, proc


def _generate(model_id: str, image: Image.Image, prompt: str, max_new_tokens: int,
              sample: bool, keep_special: bool = False, images_kwargs: dict | None = None,
              **template_kwargs) -> str:
    import torch
    model, proc = _load(model_id)
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]
    kw = {"images_kwargs": images_kwargs} if images_kwargs else {}
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt",
                                      **template_kwargs, **kw).to(model.device)
    # Recommended non-thinking sampling for Qwen3.5; greedy otherwise.
    gen = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20) if sample else \
        dict(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen)
    return proc.decode(out[0, inputs["input_ids"].shape[1]:],
                       skip_special_tokens=not keep_special)


class PaddleOCRVLBackend:
    name = "paddle-vl"
    _item = re.compile(r"(.*?)((?:<\|LOC_\d+\|>){8})", re.S)
    _loc = re.compile(r"<\|LOC_(\d+)\|>")

    def read(self, img: np.ndarray) -> list[Token]:
        h, w = img.shape[:2]
        # Same preprocessing as the official PaddleX pipeline for spotting.
        if w < 1500 and h < 1500:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        raw = _generate(PADDLE_VL_MODEL, to_pil(img), "Spotting:", 4096, sample=False,
                        keep_special=True,
                        images_kwargs={"size": {"shortest_edge": 112896,
                                                "longest_edge": 2048 * 28 * 28}})
        tokens = []
        for text, locs in self._item.findall(raw):
            v = list(map(int, self._loc.findall(locs)))
            pts = [(v[j] / 1000 * w, v[j + 1] / 1000 * h) for j in range(0, 8, 2)]
            text = re.sub(r"<\|[^|]*\|>", "", text).strip()
            tokens += parse_numbers(text, poly_to_box(pts))
        return tokens


class QwenOCRBackend:
    name = "qwen"
    PROMPT = ("This is part of a connect-the-dots puzzle. Detect every printed number "
              "(the labels next to the dots). Output a JSON list: "
              '[{"bbox_2d": [x1, y1, x2, y2], "label": "<number>"}]. Output only the JSON.')

    def __init__(self, model_id: str = QWEN_OCR_MODEL, grid: tuple[int, int] = (3, 3),
                 overlap: float = 0.2):
        self.model_id, self.grid, self.overlap = model_id, grid, overlap

    def tiles(self, h: int, w: int):
        rows, cols = self.grid
        th, tw = h / (rows - (rows - 1) * self.overlap), w / (cols - (cols - 1) * self.overlap)
        for i in range(rows):
            for j in range(cols):
                y0, x0 = int(i * th * (1 - self.overlap)), int(j * tw * (1 - self.overlap))
                yield x0, y0, min(w, int(x0 + tw)), min(h, int(y0 + th))

    def read(self, img: np.ndarray) -> list[Token]:
        h, w = img.shape[:2]
        tokens = []
        for x0, y0, x1, y1 in self.tiles(h, w):
            raw = _generate(self.model_id, to_pil(img[y0:y1, x0:x1]), self.PROMPT, 1500,
                            sample=False, enable_thinking=False)
            tw, th = x1 - x0, y1 - y0
            for item in _parse_json_list(raw):
                b, label = item.get("bbox_2d"), str(item.get("label", ""))
                if not (isinstance(b, list) and len(b) == 4):
                    continue
                # Qwen3-VL family: coordinates relative to the image, on a 0-1000 scale.
                bx = (x0 + b[0] / 1000 * tw, y0 + b[1] / 1000 * th,
                      x0 + b[2] / 1000 * tw, y0 + b[3] / 1000 * th)
                tokens += parse_numbers(label, bx)
        return dedupe(tokens)


def dedupe(tokens: list[Token], iou_min: float = 0.3) -> list[Token]:
    """Drop repeats of the same number found in overlapping tiles."""
    kept: list[Token] = []
    for t in tokens:
        if not any(k.value == t.value and _iou(k.box, t.box) > iou_min for k in kept):
            kept.append(t)
    return kept


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _parse_json_list(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        # Truncated output: keep the complete objects.
        data = [json.loads(o) for o in re.findall(r"\{[^{}]*\}", m.group())
                if _is_json(o)]
    return [d for d in data if isinstance(d, dict)]


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


class QwenChat:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B"):
        self.model_id = model_id

    def ask(self, img: np.ndarray, prompt: str, max_new_tokens: int = 60,
            sample: bool = True) -> str:
        return _generate(self.model_id, to_pil(img), prompt, max_new_tokens,
                         sample=sample, enable_thinking=False).strip()
