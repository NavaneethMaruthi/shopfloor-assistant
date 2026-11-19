"""Video: describe a short clip and answer questions about it.

1. ffmpeg samples frames at VIDEO_FPS.
2. Keep only frames where the scene changed (plus first and last), max 9.
3. Tile them into ONE grid image, each frame labelled with its number and time.
   Ollama gives every image roughly the same token budget, so one grid costs about
   as much as a single photo instead of 9x as much (faster, fits the context window).
4. Qwen3-VL writes one short note per frame, then 'Answer: ...'.
"""
import glob
import io
import math
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
import prompts
from handlers import llm


def extract_frames(path, tmpdir):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-vf", f"fps={config.VIDEO_FPS},scale={config.VIDEO_FRAME_WIDTH}:-2",
         "-q:v", "3", os.path.join(tmpdir, "f_%05d.jpg")],
        check=True,
    )
    files = sorted(glob.glob(os.path.join(tmpdir, "f_*.jpg")))
    return [(i / config.VIDEO_FPS, f) for i, f in enumerate(files)]


def _small(path):
    return np.asarray(Image.open(path).convert("RGB").resize((96, 54)), dtype=np.int16)


def keyframes(frames):
    if not frames:
        return []
    kept = [frames[0]]
    last = _small(frames[0][1])
    for t, f in frames[1:]:
        cur = _small(f)
        changed = (np.abs(cur - last).max(axis=2) > 30).mean()
        if changed > config.VIDEO_CHANGE_FRAC:
            kept.append((t, f))
            last = cur
    if kept[-1] != frames[-1]:
        kept.append(frames[-1])
    if len(kept) > config.VIDEO_MAX_FRAMES:
        idx = sorted(set(np.linspace(0, len(kept) - 1, config.VIDEO_MAX_FRAMES).round().astype(int)))
        kept = [kept[i] for i in idx]
    return kept


MOTION_RE = re.compile(r"\b(move|moves|moving|moved|direction|go|goes|going|travel|travels|heading|motion|slide|slides)\b", re.I)


def motion_hint(frames):
    """Track the centre of whatever differs from the static background (median of all frames).
    Classic computer vision, no answers involved. Returns a sentence or ''."""
    if len(frames) < 3:
        return ""
    stack = np.stack([_small(f) for _, f in frames])            # (n, 54, 96, 3)
    background = np.median(stack, axis=0)
    pts = []
    for t, img in zip([t for t, _ in frames], stack):
        mask = np.abs(img - background).max(axis=2) > 40
        if mask.sum() >= 4:
            ys, xs = np.nonzero(mask)
            pts.append((t, xs.mean() / mask.shape[1], ys.mean() / mask.shape[0]))
    if len(pts) < 2:
        return ""
    (t0, x0, y0), (t1, x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    parts = []
    if abs(dy) > 0.1:
        parts.append("down" if dy > 0 else "up")
    if abs(dx) > 0.1:
        parts.append("right" if dx > 0 else "left")
    direction = " and ".join(parts) if parts else "barely moved"
    return (f"Motion tracker measurement: the moving object's centre went from "
            f"{x0:.0%} across, {y0:.0%} down at t={t0:.1f}s to {x1:.0%} across, {y1:.0%} down at "
            f"t={t1:.1f}s, so it moved {direction} (from the viewer's point of view).")


def _font(size):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_grid(kept):
    """Tile frames left-to-right, top-to-bottom with a '#N t=X.Xs' label on each."""
    imgs = [Image.open(f).convert("RGB") for _, f in kept]
    w, h = imgs[0].size
    n = len(imgs)
    cols = n if n <= 3 else 3
    rows = math.ceil(n / cols)
    gap = 8
    grid = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (60, 60, 60))
    draw = ImageDraw.Draw(grid)
    font = _font(max(18, h // 10))
    for i, (im, (t, _)) in enumerate(zip(imgs, kept)):
        x, y = (i % cols) * (w + gap), (i // cols) * (h + gap)
        grid.paste(im.resize((w, h)), (x, y))
        label = f"#{i + 1}  t={t:.1f}s"
        box = draw.textbbox((x + 6, y + 6), label, font=font)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    grid.save(buf, "JPEG", quality=90)
    return buf.getvalue(), grid.size


def handle(question, path, mode="General"):
    q = (question or "").strip() or "Describe what happens in this video."
    tmpdir = tempfile.mkdtemp()
    try:
        frames = extract_frames(path, tmpdir)
        if not frames:
            return {"answer": "I couldn't read any frames from this video.", "details": "",
                    "models": config.VL_MODEL}
        kept = keyframes(frames)
        grid, _size = make_grid(kept)
        duration = frames[-1][0]
        hint = motion_hint(frames) if MOTION_RE.search(q) else ""
        prompt = (
            f"The image is a grid of {len(kept)} frames from a {duration:.1f}-second video, in "
            "chronological order, read left to right then top to bottom. Each frame has a small black label "
            "with its number and time (for example '#1 t=0.0s'). Frames were captured whenever the "
            "scene changed.\n"
            "Step 1: write one short line per frame, 'Frame N: ...', saying what is NEW or DIFFERENT "
            "compared with the previous frame (for Frame 1, say what is visible). Ignore the labels.\n"
            "Step 2: on the last line write 'Answer: ' followed by the answer to the question as one "
            "short, complete sentence.\n"
            "Rules: an object that stays visible across frames is the same object, so count each "
            "distinct object only once. For 'how many' questions the Answer must start with the number "
            "as a digit. For movement, compare positions between the first and last frames. "
            "For text or signs, copy the words exactly.\n"
            + (f"{hint} Trust this measurement for direction.\n" if hint else "")
            + f"Question: {q}"
        )
        reply = llm.chat(config.VL_MODEL, prompt, system=prompts.system_for(mode),
                         images=[grid], think=config.THINK_VIDEO, max_tokens=400)
        m = re.search(r"Answer:\s*(.+)", reply, re.S | re.I)
        answer = m.group(1).strip() if m else reply.strip()
        notes = reply[:m.start()].strip() if m else ""
        details = f"🎞️ Analysed {len(kept)} key frames out of {len(frames)} sampled"
        if hint:
            details += "\n\n" + hint
        if notes:
            details += "\n\n" + notes
        return {"answer": answer, "details": details, "models": config.VL_MODEL}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
