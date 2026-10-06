"""Accuracy against hand-checked ground truth (data/gt/<name>.json).

Ground truth stores, for every number, its dot position in the ORIGINAL photo,
so it stays valid if rectification changes. Metrics, per sample:
- dots found:     GT dots with a detected dot within TOL dot radii
- numbers read:   GT numbers k with an OCR token reading k next to GT dot k
                  (within the matching gate), i.e. read correctly at the right place
- wrong readings: OCR tokens whose value has no matching GT dot nearby
- numbers placed: GT numbers k whose final position is within TOL of GT dot k
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detect import Dots
from .match import GATE, Match, box_distance
from .ocr import Token
from .rectify import Rectified

TOL = 1.5   # dot radii


@dataclass
class Score:
    name: str
    n: int
    dots_found: int
    numbers_read: int
    wrong_readings: int
    tokens: int
    placed: int

    def row(self) -> list[str]:
        # One decimal unless exact, so 809/812 reads 99.6% rather than a rounded 100%.
        pct = lambda a: f"{a}/{self.n} ({'100%' if a == self.n else f'{100 * a / self.n:.1f}%'})"
        return [self.name, pct(self.dots_found), pct(self.numbers_read),
                f"{self.wrong_readings}/{self.tokens}", pct(self.placed)]


def load_gt(photo: Path) -> dict | None:
    path = photo.parent / "gt" / f"{photo.stem}.json"
    return json.loads(path.read_text()) if path.exists() else None


def score(name: str, gt: dict, rect: Rectified, dots: Dots, tokens: list[Token],
          m: Match) -> Score:
    nums = sorted(int(k) for k in gt["dots"])
    orig = np.array([gt["dots"][str(k)] for k in nums], dtype=np.float64)
    pts = cv2.perspectiveTransform(orig.reshape(-1, 1, 2), rect.H).reshape(-1, 2)
    gt_xy = dict(zip(nums, pts))
    r = dots.radius

    if len(dots.centers):
        d = np.linalg.norm(pts[:, None] - dots.centers[None], axis=2)
        dots_found = int((d.min(axis=1) <= TOL * r).sum())
    else:
        dots_found = 0

    read, wrong = set(), 0
    for t in tokens:
        g = gt_xy.get(t.value)
        if g is not None and box_distance(g[None], t.box)[0] <= GATE * r:
            read.add(t.value)
        else:
            wrong += 1

    placed = sum(1 for v, p in m.path
                 if v in gt_xy and np.linalg.norm(p - gt_xy[v]) <= TOL * r)
    return Score(name, gt["n"], dots_found, len(read), wrong, len(tokens), placed)


def format_table(scores: list[Score]) -> str:
    head = ["sample", "dots found", "numbers read", "wrong readings", "numbers placed"]
    rows = [head] + [s.row() for s in scores]
    if len(scores) > 1:
        tot = Score("TOTAL", sum(s.n for s in scores), sum(s.dots_found for s in scores),
                    sum(s.numbers_read for s in scores), sum(s.wrong_readings for s in scores),
                    sum(s.tokens for s in scores), sum(s.placed for s in scores))
        rows.append(tot.row())
    w = [max(len(r[i]) for r in rows) for i in range(len(head))]
    fmt = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
    sep = "|" + "|".join("-" * (x + 2) for x in w) + "|"
    return "\n".join([fmt(rows[0]), sep] + [fmt(r) for r in rows[1:]])
