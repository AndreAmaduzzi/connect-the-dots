"""Upload a photo or a phone video of a connect-the-dots puzzle; get the reel and images.

    uv run python app.py            # then open http://127.0.0.1:7860
    uv run python app.py --share    # also a temporary public link (gradio.live)

Everything runs on this machine; results are also saved in results/<name>/.
"""

import argparse
import time
from pathlib import Path

import gradio as gr

from ctd.ocr import get_backend
from ctd.pipeline import Options, solve
from ctd.reel import SIZES

_ocr = None


def run(file, aspect, start, answer, guess):
    global _ocr
    if not file:
        raise gr.Error("Upload a photo or a video of the puzzle first.")
    if _ocr is None:
        _ocr = get_backend("paddle")
    src = Path(file)
    out = Path("results") / f"{src.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    log = []
    opts = Options(aspects=[aspect], start=float(start or 0), answer=answer.strip() or None,
                   guess=guess)
    try:
        res = solve(src, out, _ocr, opts, log=log.append)
    except Exception as e:   # shown in the page instead of a stack trace
        raise gr.Error(f"Could not solve this one: {e}")
    reel = out / f"reel_{aspect}.mp4"
    images = [(str(out / f), cap) for f, cap in [
        ("solved.jpg", "the solution on your photo"),
        ("drawing.png", "the finished drawing"),
        ("guess_input.png", "what the AI looked at"),
        (f"thumbnail_{aspect}.jpg", "a cover image for social posts")] if (out / f).exists()]
    said = f"The AI says: {res.guess}" if res.guess else "No AI guess (turned off)."
    report = (out / "report.txt").read_text() if (out / "report.txt").exists() else ""
    return (str(reel) if reel.exists() else None, images, said,
            report + "\n" + "\n".join(log) + f"\n\nSaved in {out}/")


with gr.Blocks(title="Connect the dots") as demo:
    gr.Markdown("# Connect the dots\nUpload a photo or a short phone video of a connect-the-dots "
                "puzzle. The code finds the dots, reads the numbers, draws the lines in order "
                "and asks an AI what the picture shows.")
    with gr.Row():
        with gr.Column():
            file = gr.File(label="Photo or video", file_types=["image", "video"], type="filepath")
            aspect = gr.Radio(list(SIZES), value="9x16", label="Video format")
            start = gr.Number(value=0, label="Video only: start the reel at (seconds)")
            answer = gr.Textbox(label="Optional: what it really shows, if the AI gets it wrong")
            guess = gr.Checkbox(value=True, label="Ask the AI what it shows (needs a GPU)")
            go = gr.Button("Solve", variant="primary")
        with gr.Column():
            reel = gr.Video(label="Your reel")
            said = gr.Textbox(label="Answer", interactive=False)
    images = gr.Gallery(label="Images", columns=4, height="auto")
    report = gr.Textbox(label="Report", lines=8, interactive=False)
    go.click(run, [file, aspect, start, answer, guess], [reel, images, said, report],
             concurrency_limit=1)   # one puzzle at a time: the models share one GPU

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="also create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.launch(server_name="127.0.0.1", server_port=a.port, share=a.share)
