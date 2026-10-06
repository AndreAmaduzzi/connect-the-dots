"""Solve connect-the-dots puzzles from photos or phone videos and render a reel.

    uv run python run.py data/06_10_26.mp4 --start 4     # -> results/06_10_26/
    uv run python run.py my_puzzle.jpg --aspect 9x16 1x1
    uv run python run.py my_videos/                      # every photo/video in a folder

Each input gets a folder in --out (see ctd/pipeline.py for its contents), and
--out/README.md indexes all of them. A video is solved on its sharpest frame.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ctd.ocr import get_backend
from ctd.pipeline import Options, solve, write_index
from ctd.reel import SIZES, TAGLINE
from ctd.video import VIDEO_EXTS

EXTS = {".jpg", ".jpeg", ".png"} | VIDEO_EXTS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=Path, nargs="+", help="photos or videos, or folders of them")
    p.add_argument("--out", type=Path, default=Path("results"), help="results folder")
    p.add_argument("--aspect", nargs="+", choices=list(SIZES), default=["9x16"],
                   help="reel formats to render (default: 9x16)")
    p.add_argument("--ocr", choices=["paddle", "paddle-vl", "qwen"], default="paddle",
                   help="number reader (default: PP-OCRv6, the most accurate on the samples)")
    p.add_argument("--n", type=int, help="highest number in the puzzle, if known")
    p.add_argument("--corners", type=float, nargs=8, metavar="XY",
                   help="puzzle frame corners tl tr br bl (x y each), if detection fails")
    p.add_argument("--no-guess", action="store_true", help="skip the VLM caption (no GPU needed)")
    p.add_argument("--guess-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--regen-guess", action="store_true", help="ignore the cached guess")
    p.add_argument("--debug", action="store_true", help="save intermediate images to debug/")
    p.add_argument("--rotate", default="auto", choices=["auto", "0", "90", "180", "270"],
                   help="video input: degrees to rotate clockwise (default: detect)")
    p.add_argument("--credit", help="video input: extra line on the reel's end card, e.g. your name")
    p.add_argument("--tagline", default=TAGLINE,
                   help=f"video input: line under the guess on the end card (default: {TAGLINE!r})")
    p.add_argument("--answer", help="video input: what the picture really shows, if the AI got it "
                                    "wrong; the reel shows the AI's guess, then 'Actually: ...' "
                                    "(stored in guess.json; --answer '' clears it)")
    p.add_argument("--start", type=float, default=0.0,
                   help="video input: start the reel this many seconds into the footage")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    photos = []
    for i in args.inputs:
        photos += sorted(f for f in i.iterdir() if f.suffix.lower() in EXTS) if i.is_dir() else [i]
    if len(photos) > 1 and (args.n or args.corners):
        sys.exit("--n and --corners apply to a single photo")

    opts = Options(aspects=args.aspect, last=args.n,
                   corners=np.array(args.corners).reshape(4, 2) if args.corners else None,
                   guess=not args.no_guess, guess_model=args.guess_model,
                   regen_guess=args.regen_guess, debug=args.debug,
                   rotation=args.rotate if args.rotate == "auto" else int(args.rotate),
                   credit=args.credit, tagline=args.tagline, answer=args.answer, start=args.start)
    ocr = get_backend(args.ocr)
    failed = []
    for photo in photos:
        print(f"{photo}")
        try:
            solve(photo, args.out / photo.stem, ocr, opts)
        except Exception as e:   # keep going on a batch; report at the end
            print(f"  FAILED: {e}")
            failed.append(photo.name)

    answers = {}
    for i in args.inputs:
        a = (i if i.is_dir() else i.parent) / "answers.json"
        if a.exists():
            answers.update(json.loads(a.read_text()))
    print(f"wrote {write_index(args.out, answers)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
