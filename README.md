# connect-the-dots

Give it a photo or a short phone video of a connect-the-dots puzzle (built for
*La Settimana Enigmistica*'s "La Pista Cifrata"). It finds the dots, reads the
numbers, draws the lines 1 → 2 → 3 … on your footage, and asks an AI what the
finished picture shows. You get a reel ready for social media, plus images.

```
photo/video ─► find puzzle ─► dots ─► numbers ─► match 1→N ─► reel (MP4) + images
                                                       └─► silhouette ─► AI: "a deer drinking from a tree"
```

## Install

Needs Linux, Python 3.12 and [uv](https://docs.astral.sh/uv/). The AI caption
needs an NVIDIA GPU with ~20 GB free (Qwen3.5-9B); everything else runs on CPU,
and `--no-guess` skips the caption.

```bash
git clone <this repo> && cd connect-the-dots
uv sync
```

The first run downloads the models: PP-OCRv6 (~135 MB) and Qwen3.5-9B (~19 GB).
`pyproject.toml` pins torch to the CUDA 12.6 wheels, which run on any 12.x
driver; with a CUDA 13 driver you can drop the `[tool.uv.sources]` block.

## Use it

**Upload page** (local web app):

```bash
uv run python app.py            # open http://127.0.0.1:7860
uv run python app.py --share    # also a temporary public link
```

Drop in a photo or a video, press *Solve*, and get the reel, the images, the
AI's answer and a report.

**Command line**:

```bash
uv run python run.py data/06_10_26.mp4 --start 4     # the sample video
uv run python run.py my_puzzle.jpg                   # a photo
uv run python run.py my_videos/                      # every photo/video in a folder
```

Each input gets a folder in `results/`:

| file | what |
|---|---|
| `reel_9x16.mp4` | the reel (1080×1920, ~20–27 s, H.264 at 30 fps, within LinkedIn's video limits) |
| `thumbnail_9x16.jpg` | a cover image: the drawing and the AI's answer |
| `solved.jpg` | your photo (or the video's sharpest frame) with the solution drawn on it |
| `drawing.png` | the finished picture as a clean line drawing |
| `guess_input.png` | what the AI looked at (the drawing as a filled silhouette) |
| `guess.json` | the AI's answer (cached: re-runs reuse it, `--regen-guess` asks again) |
| `report.txt` | what was found; **INCOMPLETE** if some numbers could not be placed |
| `keyframe.jpg` | for a video: the frame the puzzle was solved on |
| `debug/` | with `--debug`: frame, rectified page, dots, OCR boxes, matches |

Options:

| option | |
|---|---|
| `--aspect 9x16 4x5 1x1` | reel formats (default 9x16; 4x5 fills the most of a feed, 1x1 is square) |
| `--start 4` | video: start the reel 4 s into the footage |
| `--answer "a deer eating from a trough"` | if the AI is wrong: the reel shows its guess, then "Actually: …" |
| `--tagline "…"`, `--credit "Your Name"` | text on the reel's end card |
| `--rotate 90` | force the rotation (degrees clockwise); videos are turned upright automatically, photos follow their EXIF |
| `--no-guess` | no AI caption (no GPU needed) |
| `--n 56` | the highest number, if detection guesses wrong |
| `--corners x y x y x y x y` | the puzzle frame's corners (tl tr br bl), if it is not found |

**Filming tips:** end the clip on the whole puzzle and hold still for a second
(the sharpest frame there is the one that gets solved); even light, no phone
shadow on the page; the puzzle should fill most of the frame's width.

## The reel

| | |
|---|---|
| hook | your footage, upright (a photo is held for 1.2 s) |
| freeze | a camera-flash freeze on the solved frame; the view never moves |
| scan | a glowing line sweeps the puzzle and each dot pops in ("Found 56 dots") |
| draw | a glowing pen draws 1 → N with a live "23 / 56" counter |
| reveal | the page dims, "Asking the AI what it sees…", its answer types itself out |
| end card | the clean drawing glowing on a dark card, the answer, a tagline; held 3.5 s |

The path is closed (N → 1) when the last dot is next to the first, so the
outline is finished; a long closing line would cut across the picture, so it is
left out then. Numbers that cannot be placed are never hidden: the report says
INCOMPLETE and lists them, and such a reel should not be published.

## How it works

| module | step |
|---|---|
| `ctd/video.py` | Video input: read the frames, find the rotation (PaddleOCR's page-orientation model), pick the sharpest frame with the whole puzzle frame in view. |
| `ctd/rectify.py` | Find the puzzle's printed frame (largest dark convex quadrilateral) and warp it flat; without a frame, the whole image is used. |
| `ctd/detect.py` | Dots = solid round blobs: as thick as a dot at their centre (rejects digit fragments and lines) and left by at most one thick stroke (rejects points along printed lines and their junctions). Touching dots are split at their thickness peaks. |
| `ctd/ocr.py` | PP-OCRv6 reads the numbers, then clean-up: drop the glued-on dot ("13°"), split merged labels ("1.50" → 1, 50), map look-alikes ("I0" → 10). |
| `ctd/match.py` | Infer the sequence (1, 2, 3… or 0, 2, 4…), then a Hungarian assignment of numbers to dots by distance; duplicates are resolved by fit with their neighbours. |
| `ctd/recover.py` | Second pass: look for a missed dot next to each label left without one, and re-read an upscaled crop around each dot left without a number. |
| `ctd/guess.py` | Fill the drawing's outline as a silhouette and ask Qwen3.5-9B two questions: what the scene shows, then what its subject is doing and with what. |
| `ctd/reel.py` | The reel (timeline above), drawn frame by frame with OpenCV and Pillow, encoded with ffmpeg. Text in Poppins (`assets/fonts`, SIL Open Font License). |
| `ctd/pipeline.py` | `locate()` (detect → read → match → recover) and `solve()` (input → results folder), used by `run.py`, `app.py` and `evaluate.py`. |
| `ctd/vlm.py` | Loads the vision-language models; also the VLM number readers compared below (`--ocr paddle-vl`, `--ocr qwen`). |

## Accuracy

**Placing the numbers.** On 10 puzzles with ground truth (our own phone photos
and a 351 px web image, not included here, and 7 Wikimedia Commons puzzles),
801 of 812 numbers end up on the right dot (98.6%): 100% on 9 of them, 78% on
the low-resolution web image. `tools/fetch_samples.py` downloads the Commons
puzzles with exact ground truth; `evaluate.py` scores them
(`reports/accuracy.md`).

**Reading the numbers: a small OCR model beats a big VLM.** On the first three
test puzzles (147 numbers):

| reader | numbers on the right dot | time / image |
|---|---|---|
| PP-OCRv6 (34M parameters, CPU) | 135/147 (91.8%) | 18 s |
| PaddleOCR-VL-1.6 (0.9B, GPU) | 128/147 (87.1%) | 183 s |
| Qwen3.5-9B (bounding boxes, GPU) | 34/147 (23.1%) | 120 s |

Qwen reads most labels, but its boxes are off by 3–4 dot radii on labels this
small, so the numbers land on the wrong dots.

**Naming the picture: show the AI a silhouette, not lines.** On 12 drawings
with known answers, Qwen3.5-9B named the subject in 6 when shown the line
drawing, and in 10 when shown the filled silhouette and asked the follow-up
question (greedy decoding, so the same puzzle always gets the same answer).
The deer went from "a dog attacking a tower" (9 of 9 tries on lines) to "a deer
drinking from a tree". A larger model (Qwen3.6-27B, 8-bit) did worse (12/24 vs
17/24 on the silhouettes). Still wrong: a crow and a secretary bird come out as
"a bird"; the follow-up question sometimes invents an action.

**Known limits.** Very low-resolution images lose labels and dots; hand-drawn
dots of very different sizes are partly missed (the dot size is estimated once
per image). Printed puzzles with uniform dots work best.

## Layout

```
app.py          upload page (Gradio)
run.py          command line: photos or videos -> results/
evaluate.py     accuracy report on samples with ground truth
ctd/            the pipeline (see How it works)
tools/          fetch_samples.py: test puzzles from Wikimedia Commons
assets/fonts/   Poppins (SIL Open Font License)
data/           06_10_26.mp4 (the sample video); fetched samples land here
reports/        accuracy.md
```

## License

MIT (see `LICENSE`). The models are downloaded from Hugging Face / PaddlePaddle
under their own licenses. The sample video shows a page of *La Settimana
Enigmistica*; the puzzle belongs to its publisher.
