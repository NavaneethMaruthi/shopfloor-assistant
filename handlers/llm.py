"""Thin wrapper around the local Ollama server."""
import re
import time
from io import BytesIO

import ollama
from PIL import Image, ImageOps

import config

try:  # iPhone photos (HEIC/HEIF)
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def chat(model, prompt, system=None, images=None, think=None, json_mode=False,
         num_ctx=None, temperature=0.0, max_tokens=None):
    """Send one prompt (optionally with images) and return the reply text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user = {"role": "user", "content": prompt}
    if images:
        user["images"] = images
    messages.append(user)

    kwargs = dict(model=model, messages=messages,
                  options={"num_ctx": num_ctx or config.NUM_CTX, "temperature": temperature,
                           "num_predict": max_tokens or config.MAX_TOKENS})
    if json_mode:
        kwargs["format"] = "json"
    if think is not None:
        kwargs["think"] = think

    t0 = time.time()
    try:
        resp = ollama.chat(**kwargs)
    except Exception as e:  # some models / library versions don't accept the think flag
        if "think" in kwargs and ("think" in str(e).lower() or isinstance(e, TypeError)):
            kwargs.pop("think")
            resp = ollama.chat(**kwargs)
        else:
            raise
    msg = resp["message"]
    content = msg["content"] or ""
    thinking = getattr(msg, "thinking", None) or (msg.get("thinking") if isinstance(msg, dict) else None)
    if config.DEBUG:
        print(f"[llm] {model:<18} {time.time() - t0:5.1f}s  think={kwargs.get('think')}  "
              f"thinking_chars={len(thinking or '')}  images={len(images or [])}  "
              f"in_tokens={_get(resp, 'prompt_eval_count')}  out_tokens={_get(resp, 'eval_count')}")
    return _THINK_RE.sub("", content).strip()


def _get(resp, key):
    try:
        return resp[key]
    except Exception:
        return getattr(resp, key, None)


def image_bytes(path, max_side=None):
    """Load an image, fix phone rotation, shrink it, return JPEG bytes."""
    max_side = max_side or config.IMAGE_MAX_SIDE
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()
