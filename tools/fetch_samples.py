"""Download extra connect-the-dots samples into data/ and build their ground truth.

    uv run python tools/fetch_samples.py

Sources (see data/SOURCES.md, written by this script):
- Wikimedia Commons, "Category:Connect-the-dots puzzles": freely licensed SVG
  puzzles (rendered to PNG by Commons) and one phone photo of a puzzle.

Ground truth for the SVGs comes from the files themselves: they were generated
by a script that writes the i-th <circle> (the dot) and the i-th <text> (its
number) in the same order, so pairing by order is exact. Files where that
doesn't hold (different counts, numbers not 1..N) get no ground truth.
"""

import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "connect-the-dots-demo/0.1 (https://github.com/; research use)"}
RENDER_WIDTH = 1600   # requested; Commons may serve a larger standard size
SVG = "{http://www.w3.org/2000/svg}"

COMMONS_FILES = [
    "File:Crow - connect the dots.svg",
    "File:Dragon - connect the dots.svg",
    "File:Hydra - connect the dots.svg",
    "File:Kiwi bird - connect the dots.svg",
    "File:Seahorse - connect the dots.svg",
    "File:Secretary bird - connect the dots.svg",
    "File:Trumpet - connect the dots.svg",
    # Not used: "File:Caesar-dot-to-dot.svg" is mostly already connected and its
    # numbers 146+ have no dots (an unfinished solution sheet, not a puzzle).
    "File:Connect the dots puzzle.jpg",
]
# What each puzzle shows, from its title / the Commons "Solution" files.
ANSWERS = {"crow": "a crow", "dragon": "a dragon", "hydra": "a hydra", "kiwi_bird": "a kiwi bird",
           "seahorse": "a seahorse", "secretary_bird": "a secretary bird", "trumpet": "a trumpet",
           "caesar": "Julius Caesar"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=60).read()


def commons_info(title: str) -> dict:
    q = urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "imageinfo", "format": "json",
        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": RENDER_WIDTH,
        "iiextmetadatafilter": "LicenseShortName|Artist"})
    page = next(iter(json.loads(get("https://commons.wikimedia.org/w/api.php?" + q))
                     ["query"]["pages"].values()))
    return page["imageinfo"][0]


def short_name(title: str) -> str:
    name = title[5:].rsplit(".", 1)[0].lower()
    name = re.sub(r"\s*-?\s*(connect the dots|dot-to-dot)", "", name)
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def flatten(png: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


# --- SVG ground truth -------------------------------------------------------

def parse_transform(s: str | None) -> np.ndarray:
    m = np.eye(3)
    for kind, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        a = [float(v) for v in re.split(r"[\s,]+", args.strip()) if v]
        if kind == "translate":
            t = np.array([[1, 0, a[0]], [0, 1, a[1] if len(a) > 1 else 0], [0, 0, 1]])
        elif kind == "scale":
            t = np.diag([a[0], a[1] if len(a) > 1 else a[0], 1])
        elif kind == "matrix":
            t = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif kind == "rotate":
            c, s_ = np.cos(np.radians(a[0])), np.sin(np.radians(a[0]))
            t = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            if len(a) == 3:
                t = (np.array([[1, 0, a[1]], [0, 1, a[2]], [0, 0, 1]]) @ t
                     @ np.array([[1, 0, -a[1]], [0, 1, -a[2]], [0, 0, 1]]))
        else:
            raise ValueError(f"unsupported transform {kind}")
        m = m @ t
    return m


def svg_ground_truth(svg: bytes, out_w: int, out_h: int) -> tuple[dict | None, str]:
    root = ET.fromstring(svg)
    vb = root.get("viewBox")
    if not vb:
        return None, "no viewBox"
    vx, vy, vw, vh = (float(v) for v in re.split(r"[\s,]+", vb.strip()))
    to_px = np.array([[out_w / vw, 0, -vx * out_w / vw], [0, out_h / vh, -vy * out_h / vh],
                      [0, 0, 1]])
    circles, numbers = [], []

    def walk(el, m):
        m = m @ parse_transform(el.get("transform"))
        if el.tag == SVG + "circle":
            p = to_px @ m @ [float(el.get("cx", 0)), float(el.get("cy", 0)), 1]
            circles.append(p[:2])
        elif el.tag == SVG + "text":
            numbers.append("".join(el.itertext()).strip())
            return
        for child in el:
            walk(child, m)

    walk(root, np.eye(3))
    if not circles or len(circles) != len(numbers):
        return None, f"{len(circles)} circles vs {len(numbers)} numbers"
    # Most count 1, 2, 3, ...; some count in steps from 0 ("0, 2, 4, ...").
    values = sorted(int(n) for n in numbers if n.isdigit())
    step = values[1] - values[0] if len(values) > 1 else 1
    start = values[0] if values else 0
    if len(values) != len(numbers) or start not in (0, step) or \
            values != [start + step * i for i in range(len(values))]:
        return None, "numbers are not an evenly spaced sequence"
    dots = {n: [round(float(x), 1), round(float(y), 1)] for n, (x, y) in zip(numbers, circles)}
    return {"n": len(numbers), "start": start, "step": step, "note": "Extracted from the SVG source (i-th "
            "circle = i-th number), in the coordinates of the rendered image.",
            "dots": dict(sorted(dots.items(), key=lambda kv: int(kv[0])))}, "ok"


# ---------------------------------------------------------------------------

def main() -> int:
    (DATA / "gt").mkdir(parents=True, exist_ok=True)
    sources = ["# Sample sources\n",
               "`06_10_26.mp4`: a phone video of La Settimana Enigmistica's \"La Pista Cifrata\" "
               "(56 dots), filmed by the author.\n",
               "Downloaded by `tools/fetch_samples.py`:\n",
               "| file | source | license | shows | ground truth |", "|---|---|---|---|---|"]
    answers = {}
    for title in COMMONS_FILES:
        info = commons_info(title)
        name = "commons_" + short_name(title)
        lic = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "?")
        if info["mime"] == "image/svg+xml":
            img = flatten(get(info["thumburl"]))
            gt, why = svg_ground_truth(get(info["url"]), *img.size)
        else:
            img, gt, why = Image.open(io.BytesIO(get(info["url"]))).convert("RGB"), None, "photo"
        out = DATA / f"{name}.jpg"
        img.save(out, quality=92)
        if gt:
            (DATA / "gt" / f"{name}.json").write_text(json.dumps(gt, indent=1))
        shows = ANSWERS.get(name.removeprefix("commons_"), "")
        answers[name] = shows
        sources.append(f"| `{out.name}` | [{title[5:]}]({info['descriptionurl']}) | {lic} | "
                       f"{shows or '?'} | {'yes (' + str(gt['n']) + ' dots)' if gt else 'no: ' + why} |")
        print(f"{out.name:40} {img.size[0]}x{img.size[1]}  {lic:12} gt: {why}")
        time.sleep(0.5)
    (DATA / "SOURCES.md").write_text("\n".join(sources) + "\n")
    # Merge: answers.json also holds the hand-written answers for our own photos.
    path = DATA / "answers.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**merged, **answers}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
