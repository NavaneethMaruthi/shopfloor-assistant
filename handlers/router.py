"""Routes each request to the right handler by file type (or code intent)."""
import os
import re
import time

import config
import prompts
from handlers import audio, code, image, llm, text, video

EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".heif"},
    "audio": {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"},
    "video": {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"},
    "text": {".txt", ".pdf", ".md"},
}
HANDLERS = {"image": image, "audio": audio, "video": video, "text": text, "code": code}

CODE_RE = re.compile(
    r"(^\s*(def|class|from \w+ import|import)\s)"
    r"|\b(python|function|script|implement|algorithm|program|write (a |some |the )?code)\b",
    re.I | re.M,
)


def detect(path):
    ext = os.path.splitext(path)[1].lower()
    for modality, exts in EXTENSIONS.items():
        if ext in exts:
            return modality
    return None


def is_code_request(question):
    return bool(question and CODE_RE.search(question))


def handle(question, path=None, mode="General", context=""):
    t0 = time.time()
    if path:
        modality = detect(path)
        if modality is None:
            return {"answer": f"I can't read {os.path.splitext(path)[1]} files yet. Try a photo, "
                              "audio recording, video, PDF or TXT file.",
                    "details": "", "modality": "unsupported", "latency": 0.0, "models": ""}
    elif is_code_request(question):
        modality = "code"
    else:
        modality = "chat"

    if modality == "chat":
        result = {"answer": llm.chat(config.VL_MODEL, prompts.context_block(context) + (question or "Hello"),
                                     system=prompts.system_for(mode), think=False),
                  "details": "", "models": config.VL_MODEL}
    else:
        result = HANDLERS[modality].handle(question, path, mode, context=context)

    result["modality"] = modality
    result["latency"] = time.time() - t0
    return result


def warmup():
    """Load every model once so the first real request isn't slow."""
    t0 = time.time()
    llm.chat(config.VL_MODEL, "Reply with OK.", think=False)
    llm.chat(config.CODE_MODEL, "Reply with OK.")
    audio.warmup()
    return time.time() - t0
