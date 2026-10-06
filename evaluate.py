"""Accuracy report on the samples that have ground truth (data/gt/*.json).

    uv run python evaluate.py                     # PP-OCRv6 only
    uv run python evaluate.py --ocr paddle qwen paddle-vl

Prints one table per OCR backend and writes them to reports/accuracy.md.
"""

import argparse
import time
from pathlib import Path

import cv2

from ctd.ocr import get_backend
from ctd.pipeline import locate
from ctd.rectify import rectify
from ctd.report import format_table, load_gt, score

EXTS = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--ocr", nargs="+", default=["paddle"], choices=["paddle", "paddle-vl", "qwen"])
    p.add_argument("--out", type=Path, default=Path("reports/accuracy.md"))
    args = p.parse_args()

    photos = sorted(f for f in args.data.iterdir() if f.suffix.lower() in EXTS and load_gt(f))
    prepared = [(f, rectify(cv2.imread(str(f))), load_gt(f)) for f in photos]

    sections = []
    for name in args.ocr:
        backend = get_backend(name)
        scores, secs = [], 0.0
        for f, rect, gt in prepared:
            t = time.time()
            # Same steps as run.py, including the recovery pass.
            loc = locate(rect.image, backend, max(int(k) for k in gt["dots"]))
            secs += time.time() - t
            scores.append(score(f.stem, gt, rect, loc.dots, loc.tokens, loc.match))
        table = format_table(scores)
        title = f"OCR backend: {name} ({secs / len(prepared):.0f} s/image)"
        print(f"\n{title}\n{table}")
        sections.append(f"## {title}\n\n{table}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "# Accuracy on the samples\n\n"
        "- **dots found**: ground-truth dots with a detected dot within 1.5 dot radii\n"
        "- **numbers read**: numbers read correctly *next to their own dot*\n"
        "- **wrong readings**: OCR numbers with no matching dot (misreads, duplicates)\n"
        "- **numbers placed**: numbers that end up on the correct dot after matching "
        "(what the animation draws)\n\n" + "\n".join(sections))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
