"""Video: describe a short clip and answer questions about it.

ffmpeg samples frames at VIDEO_FPS -> keep only frames where the scene changed
(plus first and last) -> send them in order, with timestamps, to Qwen3-VL.
Keeping change-frames catches each new object, movement and sign change
without sending dozens of near-identical frames.
"""
import glob
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

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


def handle(question, path, mode="General"):
    q = (question or "").strip() or "Describe what happens in this video."
    tmpdir = tempfile.mkdtemp()
    try:
        frames = extract_frames(path, tmpdir)
        if not frames:
            return {"answer": "I couldn't read any frames from this video.", "details": "",
                    "models": config.VL_MODEL}
        kept = keyframes(frames)
        duration = frames[-1][0]
        listing = "\n".join(f"Frame {i + 1}: t={t:.2f}s" for i, (t, _) in enumerate(kept))
        prompt = (
            f"The images are {len(kept)} frames from a {duration:.1f}-second video, in chronological "
            "order. Frames were captured whenever the scene changed, so each one shows a new moment.\n"
            f"{listing}\n\n"
            "Answer the question about the video. Pay attention to what changes between frames "
            "(what appears, disappears, moves, and in what order). For 'how many' questions, count "
            "distinct objects without counting the same object twice, and START with the number as a "
            "digit. For movement, compare positions across frames.\n"
            f"Question: {q}"
        )
        answer = llm.chat(
            config.VL_MODEL, prompt,
            system=prompts.system_for(mode),
            images=[llm.image_bytes(f, max_side=config.VIDEO_FRAME_WIDTH) for _, f in kept],
            think=config.THINK_VIDEO,
        )
        return {"answer": answer,
                "details": f"🎞️ Analysed {len(kept)} key frames out of {len(frames)} sampled",
                "models": config.VL_MODEL}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
