"""Routes each request to the right handler by file type (or code intent)."""
import os
import re
import tempfile
import time
import zipfile

import config
import prompts
from handlers import audio, code, image, llm, text, video

EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".heif"},
    "audio": {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"},
    "video": {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"},
    "text": {".txt", ".pdf", ".md", ".docx"},
}
KIND_NAME = {"image": "image", "audio": "audio", "video": "video", "text": "document"}

ZIP_MAX_FILES = 50
ZIP_MAX_BYTES = 200 * 1024 * 1024   # refuse archives that unpack to more than this
_zip_cache = {}
HANDLERS = {"image": image, "audio": audio, "video": video, "text": text, "code": code}

CODE_RE = re.compile(
    r"(^\s*(def|class|from \w+ import|import)\s)"
    r"|\b(python|function|script|implement|algorithm|program|write (a |some |the )?code)\b",
    re.I | re.M,
)


def unpack_zip(path):
    """Safely extract a zip: readable file types only, no path tricks, size and count limits.
    Returns ([(name inside zip, extracted path)], [names skipped])."""
    if path in _zip_cache:
        return _zip_cache[path]
    out_dir = tempfile.mkdtemp(prefix="zip_")
    files, skipped, total = [], [], 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/") or os.path.basename(name).startswith("."):
                continue
            if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
                skipped.append(name)
                continue
            if detect(name) in (None, "archive") or len(files) >= ZIP_MAX_FILES:
                skipped.append(name)
                continue
            total += info.file_size
            if total > ZIP_MAX_BYTES:
                skipped.append(name)
                continue
            target = os.path.join(out_dir, f"{len(files):02d}_{os.path.basename(name)}")
            with z.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            files.append((name, target))
    _zip_cache[path] = (files, skipped)
    return files, skipped


def _pick_from_zip(question, files):
    q = (question or "").lower()
    named = [f for f in files if os.path.basename(f[0]).lower() in q
             or os.path.splitext(os.path.basename(f[0]))[0].lower() in q]
    if len(named) == 1:
        return named[0]
    if len(files) == 1:
        return files[0]
    return None


def handle_zip(question, path, mode, context):
    try:
        files, skipped = unpack_zip(path)
    except zipfile.BadZipFile:
        return {"answer": "This zip file is damaged or not a real zip, so I can't open it.",
                "details": "", "models": ""}
    listing = ", ".join(f"{os.path.basename(n)} ({KIND_NAME[detect(n)]})" for n, _ in files) or "no files I can read"
    skipped_note = f" Skipped {len(skipped)} file(s) I can't read." if skipped else ""
    if not files:
        return {"answer": "This zip has no files I can read. I can read images, audio, video, and PDF, "
                          f"Word or text documents.{skipped_note}",
                "details": "", "models": ""}
    choice = _pick_from_zip(question, files)
    if choice is None:
        return {"answer": f"This zip contains {len(files)} file(s) I can read: {listing}.{skipped_note} "
                          "Ask about one by name, for example: What does "
                          f"{os.path.basename(files[0][0]) if files else 'notes.txt'} say?",
                "details": "", "models": ""}
    name, inner = choice
    kind = detect(inner)
    result = HANDLERS[kind].handle(question, inner, mode, context=context)
    note = f"🗜️ From {name} inside the zip ({len(files)} readable file(s): {listing}).{skipped_note}"
    result["details"] = note + ("\n\n" + result["details"] if result.get("details") else "")
    return result


def detect(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zip":
        return "archive"
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

    if modality == "archive":
        result = handle_zip(question, path, mode, context)
    elif modality == "chat":
        note = ""
        if (mode or "").lower().startswith("tech"):
            note = ("_⚠️ No document loaded: this is general guidance, not from your manual. "
                    "Attach the manual for answers with sources._")
        result = {"answer": llm.chat(config.VL_MODEL, prompts.context_block(context) + (question or "Hello"),
                                     system=prompts.system_for(mode), think=False),
                  "details": note, "models": config.VL_MODEL}
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
