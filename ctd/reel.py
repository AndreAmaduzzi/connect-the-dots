"""The LinkedIn reel: phone footage -> freeze -> scan -> glowing drawing -> AI guess.

Timeline (seconds, at the default pace):

    hook    the phone footage plays (up to the key frame)
    freeze  camera-flash freeze on the key frame (the view never moves or zooms)
    scan    a glowing line sweeps the puzzle; each dot pops in as it passes
    draw    a glowing pen draws 1 -> N (and the closing segment), with a counter
    reveal  the photo dims, "Asking the AI...", the guess types itself out
            (+ "Actually: ..." when a human correction is given)
    outro   end card: the clean drawing glowing on a dark background, guess + tagline

Everything is drawn in output-canvas pixels (crisp at 1080 wide); the footage
is placed with ONE 2x3 affine (photo -> canvas) for the whole reel, so the
puzzle stays where it was filmed. Glows are blurred on a quarter-size layer and
added back.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .overlay import crop_window

FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"
# 9:16 for the vertical video feed, 4:5 takes the most room in the main feed,
# 1:1 is the safe square. All within LinkedIn's limits (aspect 1:2.4-2.4:1,
# up to 4096x2304, 10-60 fps).
SIZES = {"9x16": (1080, 1920), "4x5": (1080, 1352), "1x1": (1080, 1080)}   # 1352: the encoder needs a multiple of 8
TAGLINE = "Filmed with a phone · solved by AI"

# Colours (BGR for OpenCV, RGB for PIL).
INK = (54, 90, 255)          # pen: hot orange-red
SCAN = (245, 214, 61)        # scan line / dot rings: cyan
TIP = (235, 250, 255)        # pen tip core
BG_TOP, BG_BOTTOM = (32, 16, 11), (58, 31, 27)   # end card gradient (dark navy)
DRAW_DIM = 0.35              # how much the page dims while the pen draws (reveal: 1.0)


@dataclass
class Pace:
    fps: float = 30.0
    flash: float = 0.25
    scan: float = 1.8
    draw: float = 7.5
    dim: float = 0.7
    thinking: float = 1.3
    type_cps: float = 28.0      # caption typing speed, characters/second
    hold: float = 2.6
    end_hold: float = 3.5       # final card fully visible (after its fade-in)
    start: float = 0.0          # seconds of footage to skip at the beginning
    still: float = 1.2          # how long a photo (one-frame footage) is shown first


def _font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS / f"Poppins-{weight}.ttf"
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def ease(t: float) -> float:
    """Smooth ease-in-out on [0, 1]."""
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def ease_out_back(t: float) -> float:
    """Overshoot then settle: used for dots popping in."""
    t = min(1.0, max(0.0, t))
    c = 1.7
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


# --- geometry ----------------------------------------------------------------

def cover_affine(img_shape, size) -> np.ndarray:
    """Photo -> canvas affine that fills the canvas (centred crop)."""
    h, w = img_shape[:2]
    W, H = size
    s = max(W / w, H / h)
    return np.array([[s, 0, (W - w * s) / 2], [0, s, (H - h * s) / 2]])


def fit_affine(img_shape, size) -> np.ndarray:
    """Photo -> canvas affine showing the whole photo (letterboxed), centred."""
    h, w = img_shape[:2]
    W, H = size
    s = min(W / w, H / h)
    return np.array([[s, 0, (W - w * s) / 2], [0, s, (H - h * s) / 2]])


def full_view(img_shape, size) -> np.ndarray:
    """Fill the canvas when the footage has about the canvas's shape; otherwise
    show it whole on a blurred copy of itself (portrait footage in a square)."""
    h, w = img_shape[:2]
    W, H = size
    close = abs(np.log((w / h) / (W / H))) < np.log(1.15)
    return cover_affine(img_shape, size) if close else fit_affine(img_shape, size)


def window_affine(win, size) -> np.ndarray:
    x0, y0, w, _ = win
    s = size[0] / w
    return np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])


def apply(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ A[:, :2].T + A[:, 2]


def place(img: np.ndarray, A: np.ndarray, size, fill: np.ndarray) -> np.ndarray:
    """Warp the photo onto the canvas; outside the photo, show `fill`."""
    out = cv2.warpAffine(img, A, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0))
    inside = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), A, size,
                            flags=cv2.INTER_NEAREST, borderValue=0)
    if inside.min() < 255:
        out = np.where(inside[..., None] > 0, out, fill)
    return out


def blurred_fill(img: np.ndarray, size) -> np.ndarray:
    W, H = size
    bg = cv2.resize(img, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    return (cv2.resize(bg, (W, H)) * 0.55).astype(np.uint8)


# --- drawing helpers -----------------------------------------------------------

def add_glow(base: np.ndarray, layer: np.ndarray, sigma: float, gain: float) -> np.ndarray:
    """Additive glow of `layer` (same size, black background) onto `base`."""
    h, w = layer.shape[:2]
    small = cv2.resize(layer, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, sigma / 4))
    glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.add(base, cv2.convertScaleAbs(glow, alpha=gain))


def polyline(layer: np.ndarray, pts: np.ndarray, progress: float, color, thick: int) -> tuple | None:
    """Draw the path up to `progress` segments (fractional). Returns the pen tip."""
    full = int(np.floor(progress))
    ip = np.round(pts).astype(int)
    for i in range(min(full, len(pts) - 1)):
        cv2.line(layer, tuple(ip[i]), tuple(ip[i + 1]), color, thick, cv2.LINE_AA)
    if full >= len(pts) - 1:
        return None
    head = pts[full] + (progress - full) * (pts[full + 1] - pts[full])
    cv2.line(layer, tuple(ip[full]), tuple(np.round(head).astype(int)), color, thick, cv2.LINE_AA)
    return tuple(np.round(head).astype(int))


def dim(img: np.ndarray, amount: float) -> np.ndarray:
    """Desaturate and darken by `amount` in [0, 1]."""
    if amount <= 0:
        return img
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    mixed = cv2.addWeighted(img, 1 - 0.7 * amount, gray, 0.7 * amount, 0)
    return cv2.convertScaleAbs(mixed, alpha=1 - 0.62 * amount)


class Text:
    """Text overlays rendered with PIL (cached per string) and alpha-blended."""

    def __init__(self, size):
        self.W, self.H = size
        self.cache = {}

    def layer(self, key, draw_fn) -> tuple[np.ndarray, np.ndarray]:
        if key not in self.cache:
            im = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
            draw_fn(ImageDraw.Draw(im))
            a = np.asarray(im)
            self.cache[key] = (cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR),
                               a[..., 3:].astype(np.float32) / 255)
        return self.cache[key]

    def blend(self, frame: np.ndarray, key, draw_fn, alpha: float = 1.0) -> np.ndarray:
        if alpha <= 0:
            return frame
        rgb, a = self.layer(key, draw_fn)
        a = a * alpha
        return (frame * (1 - a) + rgb * a).astype(np.uint8)

    # Layouts -------------------------------------------------------------
    def wrap(self, d, text, font, max_w):
        """Greedy wrap to find the number of lines, then balance them: the
        split that minimises the longest line ("It's a dog / attacking a tower!"
        rather than "It's a dog attacking a / tower!")."""
        words, lines = text.split(), [""]
        for w in words:
            trial = (lines[-1] + " " + w).strip()
            if d.textlength(trial, font=font) <= max_w or not lines[-1]:
                lines[-1] = trial
            else:
                lines.append(w)
        if len(lines) == 2:
            width = lambda ws: d.textlength(" ".join(ws), font=font)
            k = min(range(1, len(words)), key=lambda i: max(width(words[:i]), width(words[i:])))
            if max(width(words[:k]), width(words[k:])) <= max_w:
                lines = [" ".join(words[:k]), " ".join(words[k:])]
        return lines

    def pill(self, lines_spec, y_top, pad=0.04, fill=(14, 16, 26, 215), outline=None):
        """A rounded panel with lines [(text, font, rgb)] centred, top at y_top."""
        def draw(d):
            W = self.W
            px, py = int(W * pad), int(W * 0.025)
            widths = [d.textlength(t, font=f) for t, f, _ in lines_spec]
            heights = [f.size * 1.25 for _, f, _ in lines_spec]
            bw, bh = max(widths) + 2 * px, sum(heights) + 2 * py
            x0 = (W - bw) / 2
            d.rounded_rectangle([x0, y_top, x0 + bw, y_top + bh], radius=int(min(bh, W * 0.06) * 0.45),
                                fill=fill, outline=outline, width=3 if outline else 0)
            y = y_top + py
            for (t, f, rgb), hgt in zip(lines_spec, heights):
                d.text((W / 2, y + hgt / 2), t, font=f, fill=rgb, anchor="mm")
                y += hgt
        return draw


# --- the reel --------------------------------------------------------------------

def render_reel(frames: list[np.ndarray], key: int, corners: np.ndarray, path_pts: np.ndarray,
                n_numbers: int, closes: bool, drawing: np.ndarray, guess: str | None,
                out_path: str, aspect: str = "9x16", pace: Pace = Pace(),
                credit: str | None = None, src_fps: float = 30.0,
                tagline: str = TAGLINE, thumbnail_path: str | None = None,
                answer: str | None = None) -> float:
    """Render the reel. `path_pts`: the drawn path in key-frame pixels (the
    closing point repeated at the end if `closes`). `thumbnail_path`: also save
    the frame with the drawing and the full guess (for LinkedIn's custom
    thumbnail). `answer`: what the picture really shows, if the AI got it
    wrong; shown after the AI's guess as "Actually: ...", never in its place.
    Returns the duration (s)."""
    size = SIZES[aspect]
    W, H = size
    fps = pace.fps
    key_img = frames[key]
    fill = blurred_fill(key_img, size)
    tall = H / W > 1.5
    # ONE fixed view for the whole reel, so the puzzle never moves: the footage
    # as filmed when it has the canvas's shape (portrait video in 9:16),
    # otherwise a fixed crop around the puzzle (4:5, 1:1).
    A_view = cover_affine(key_img.shape, size)
    if not np.allclose(full_view(key_img.shape, size), A_view):
        A_view = window_affine(crop_window(key_img.shape, corners, size, margin=0.04,
                                           caption_room=0.17), size)
    dots = path_pts[:-1] if closes else path_pts   # each dot once
    n_seg = len(path_pts) - 1
    txt = Text(size)
    f_big = _font("Bold", int(W * 0.062))
    f_mid = _font("SemiBold", int(W * 0.042))
    f_small = _font("Medium", int(W * 0.034))
    f_num = _font("Bold", int(W * 0.05))
    WHITE, SOFT, ORANGE, CYAN = (255, 255, 255), (200, 206, 222), (255, 110, 70), (90, 220, 245)
    # Captions go where there is more free room: above the puzzle or below it
    # (in portrait footage the page often has more room below, over the comics).
    q = apply(A_view, corners)
    above, below = q[:, 1].min(), H - q[:, 1].max()
    top = H * 0.045 if above >= below else q[:, 1].max() + H * 0.03

    writer = imageio_ffmpeg.write_frames(out_path, size, fps=fps, codec="libx264",
                                         pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "18", "-movflags", "+faststart"],
                                         macro_block_size=8)
    writer.send(None)
    count = [0]

    def emit(frame):
        writer.send(np.ascontiguousarray(frame))
        count[0] += 1

    def frames_for(seconds):
        return max(1, int(round(seconds * fps)))

    # Sizes on the canvas.
    dot_r = max(5, int(W * 0.011))
    line_t = max(4, int(W * 0.0075))

    # 1. hook: the footage as filmed ----------------------------------------
    src_step = src_fps / fps
    first = min(key, int(round(pace.start * src_fps)))   # never start after the key frame
    n_hook = int((key - first) / src_step) + 1
    if n_hook == 1:   # a photo, or a start at the key frame: hold it a moment
        n_hook = frames_for(pace.still)
    for k in range(n_hook):
        src = frames[min(key, first + int(round(k * src_step)))]
        emit(place(src, A_view, size, blurred_fill(src, size)))

    # 2. freeze: a camera flash on the key frame, in place --------------------------
    base = place(key_img, A_view, size, fill)
    for k in range(frames_for(pace.flash)):
        a = 1 - k / frames_for(pace.flash)
        emit(cv2.addWeighted(base, 1 - a * 0.85, np.full_like(base, 255), a * 0.85, 0))

    P = apply(A_view, path_pts)
    D = apply(A_view, dots)
    Q = apply(A_view, corners)

    # 3. scan: a glowing line sweeps down; dots pop in as it passes ---------------
    y0, y1 = Q[:, 1].min(), Q[:, 1].max()
    x0, x1 = Q[:, 0].min(), Q[:, 0].max()
    pop_time = 0.35
    scan_caption = txt.pill([(f"Found {len(dots)} dots", f_mid, WHITE),
                             ("reading the numbers…", f_small, CYAN)], top)
    nf = frames_for(pace.scan)
    for k in range(nf + frames_for(pop_time)):
        s = min(1.0, k / nf)
        y = y0 + (y1 - y0) * ease(s)
        f = base.copy()
        glow = np.zeros_like(f)
        if s < 1:
            cv2.line(glow, (int(x0), int(y)), (int(x1), int(y)), SCAN, max(3, line_t // 2), cv2.LINE_AA)
        rings = np.zeros_like(f)
        for (dx, dy) in D:
            if dy > y + 1:
                continue
            age = (k - ((dy - y0) / max(1, y1 - y0)) * nf) / (pop_time * fps)
            rad = dot_r * (0.4 + 1.1 * ease_out_back(age))
            cv2.circle(rings, (int(dx), int(dy)), int(rad), SCAN, max(2, line_t // 2), cv2.LINE_AA)
        f = add_glow(f, glow, sigma=W * 0.02, gain=2.2)
        f = add_glow(f, rings, sigma=W * 0.008, gain=1.4)
        f = cv2.add(f, rings)
        f = cv2.add(f, glow)
        emit(txt.blend(f, "scan", scan_caption, ease(k / (0.3 * fps))))

    # 4. draw: the glowing pen, with a live counter ----------------------------------
    nf = frames_for(pace.draw)
    rings_static = np.zeros_like(base)
    for (dx, dy) in D:
        cv2.circle(rings_static, (int(dx), int(dy)), int(dot_r * 1.5), SCAN, max(2, line_t // 3), cv2.LINE_AA)
    counter_cache = {}
    # The page dims a little as the pen starts, so the glow reads on white paper.
    for k in range(nf + 1):
        # Ease the progress so the pen starts and ends gently.
        prog = n_seg * (0.5 - 0.5 * np.cos(np.pi * k / nf))
        f = dim(base, DRAW_DIM * ease(k / (0.6 * fps)))
        f = cv2.add(f, cv2.convertScaleAbs(rings_static, alpha=0.55))
        layer = np.zeros_like(f)
        tip = polyline(layer, P, prog, INK, line_t)
        f = add_glow(f, layer, sigma=W * 0.012, gain=1.8)
        mask = layer.any(axis=2)
        f[mask] = layer[mask]
        if tip is not None:
            tipl = np.zeros_like(f)
            cv2.circle(tipl, tip, int(line_t * 1.8), INK, -1, cv2.LINE_AA)
            f = add_glow(f, tipl, sigma=W * 0.02, gain=3.0)
            cv2.circle(f, tip, int(line_t * 1.1), TIP, -1, cv2.LINE_AA)
        reached = min(n_numbers, int(prog) + 1)
        if reached not in counter_cache:
            counter_cache[reached] = txt.pill([("Connecting the dots", f_small, SOFT),
                                               (f"{reached} / {n_numbers}", f_num, WHITE)], top)
        emit(txt.blend(f, f"count{reached}", counter_cache[reached]))

    # 5. reveal: dim the photo, glow the drawing, the AI's guess types itself out -----
    lines = np.zeros_like(base)
    polyline(lines, P, n_seg, INK, line_t)
    lmask = lines.any(axis=2)

    def finished(amount, pulse=0.0):
        f = dim(base, DRAW_DIM + (1 - DRAW_DIM) * amount)
        f = add_glow(f, lines, sigma=W * (0.012 + 0.01 * pulse), gain=1.8 + 1.2 * pulse)
        f[lmask] = lines[lmask]
        return f

    for k in range(frames_for(pace.dim)):
        emit(finished(ease(k / frames_for(pace.dim))))
    for k in range(frames_for(pace.thinking)):
        dots_txt = "." * (1 + (k // int(fps * 0.3)) % 3)
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * k / (fps * 0.9))
        emit(txt.blend(finished(1, pulse), f"think{dots_txt}",
                       txt.pill([(f"Asking the AI what it sees{dots_txt}", f_mid, CYAN)], top)))
    def type_out(label, label_rgb, text, key, hold, thumbnail=None):
        """A caption whose text types itself out, then holds."""
        d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        wrapped = txt.wrap(d, text, f_big, W * 0.84)
        n_chars = len(text)
        typing = int(n_chars / pace.type_cps * fps)
        for k in range(typing + frames_for(hold)):
            shown = min(n_chars, int(k * pace.type_cps / fps) + 1)
            # Fill the final line breaks progressively, so the text doesn't jump.
            spec, used = [(label, f_small, label_rgb)], 0
            for line in wrapped:
                part = text[used:used + len(line)][:max(0, shown - used)]
                spec.append((part if part else " ", f_big, WHITE))
                used += len(line) + 1
            frame = txt.blend(finished(1), f"{key}{shown}", txt.pill(spec, top))
            if thumbnail and k == typing:
                cv2.imwrite(thumbnail, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            emit(frame)

    if guess:
        # With a correction, the AI's answer holds briefly, then "Actually" types
        # out the right one (the thumbnail shows the correction).
        type_out("The AI says", CYAN, f"It's {guess}!", "guess",
                 pace.hold * (0.6 if answer else 1.0), None if answer else thumbnail_path)
        if answer:
            type_out("Actually", ORANGE, f"It's {answer}.", "answer", pace.hold, thumbnail_path)
    else:
        for _ in range(frames_for(pace.hold)):
            emit(finished(1))

    # 6. outro: the clean drawing glowing on a dark card --------------------------------
    grad = np.linspace(0, 1, H)[:, None, None]
    card = (np.array(BG_TOP) * (1 - grad) + np.array(BG_BOTTOM) * grad).astype(np.uint8)
    card = np.repeat(card, W, axis=1)
    # Clean drawing (black on white) -> white strokes on the card, orange glow.
    ink = 255 - cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    bw, bh = int(W * 0.8), int(H * (0.48 if tall else 0.55))
    s = min(bw / ink.shape[1], bh / ink.shape[0])
    ink = cv2.resize(ink, (int(ink.shape[1] * s), int(ink.shape[0] * s)), interpolation=cv2.INTER_AREA)
    canvas_ink = np.zeros((H, W), np.uint8)
    ox, oy = (W - ink.shape[1]) // 2, int(H * (0.2 if tall else 0.08))
    canvas_ink[oy:oy + ink.shape[0], ox:ox + ink.shape[1]] = ink
    glow_layer = np.zeros((H, W, 3), np.uint8)
    glow_layer[canvas_ink > 60] = INK
    card_drawn = add_glow(card, glow_layer, sigma=W * 0.015, gain=2.2)
    card_drawn[canvas_ink > 60] = (240, 245, 255)
    ty = oy + ink.shape[0] + H * 0.035
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    head = f"It's {answer}." if answer else f"It's {guess}!" if guess else "Solved!"
    end_spec = [(line, f_big, WHITE) for line in txt.wrap(d, head, f_big, W * 0.86)]
    if answer and guess:
        end_spec += [(line, f_small, SOFT) for line in
                     txt.wrap(d, f"The AI said: {guess}", f_small, W * 0.86)]
    end_spec += [(" ", f_small, SOFT), (tagline, f_mid, ORANGE)]
    if credit:
        end_spec.append((credit, f_small, CYAN))
    end = txt.pill(end_spec, ty, fill=(0, 0, 0, 0))
    last = finished(1)
    fade = frames_for(0.6)
    # The card fades in, then its text (fully visible at 1.6 x fade); then hold.
    for k in range(int(1.6 * fade) + frames_for(pace.end_hold)):
        a = ease(k / fade)
        f = cv2.addWeighted(last, 1 - a, card_drawn, a, 0)
        emit(txt.blend(f, "end", end, ease((k - fade * 0.6) / fade)))
    writer.close()
    return count[0] / fps
