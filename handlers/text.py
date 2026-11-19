"""Text: answer questions about a document, and say so when the answer isn't there.

1. The model must return JSON: {"answer": ..., "quote": exact supporting sentence}.
2. Grounding check: the quote must really appear in the document (fuzzy match).
3. Verification: a second short call checks the quote answers THIS question
   (catches near-miss traps like a different year or place).
Any failure -> "The document does not say."
"""
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache

from rapidfuzz import fuzz

import config
import prompts
from handlers import llm

SYSTEM = (
    "You answer questions using ONLY the document provided.\n"
    "Rules:\n"
    "- Use only facts stated in the document. Do not use outside knowledge.\n"
    "- Every detail in the question (names, dates, numbers, places) must match the document. "
    "If the document discusses something similar but not exactly what was asked "
    "(for example a different year, person or place), the answer is NOT in the document.\n"
    "- If several causes, steps or items answer the question, include ALL of them, with one quote "
    "for each. In a troubleshooting table, every row listed under a problem heading is a possible "
    "cause of that problem, so list every one of those rows.\n"
    '  Example: if the document says "Engine Stalls\nNo fuel Refill tank\nClogged filter Replace filter", '
    'then for "The engine stalls, what should I check?" reply {"answer": "1) No fuel: refill the tank. '
    '2) Clogged filter: replace the filter.", "quotes": ["No fuel Refill tank", "Clogged filter Replace filter"]}\n'
    '- Reply with JSON only: {"answer": "<short answer>", "quotes": ["<exact sentence from the '
    'document that states the answer>", ...]}\n'
    '- If the document does not contain the answer, reply: {"answer": "NOT_FOUND", "quotes": []}'
)


@lru_cache(maxsize=16)
def load_document(path):
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages).strip()
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def render_pdf_pages(path, max_pages=2):
    """Render the first pages of a PDF to PNG files (for scanned PDFs with no text layer)."""
    tmp = tempfile.mkdtemp()
    try:
        import fitz  # PyMuPDF, if installed
        doc = fitz.open(path)
        out = []
        for i in range(min(max_pages, len(doc))):
            p = os.path.join(tmp, f"page{i + 1}.png")
            doc[i].get_pixmap(dpi=150).save(p)
            out.append(p)
        return out
    except ImportError:
        pass
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "150", "-l", str(max_pages), path, os.path.join(tmp, "page")],
                       check=True)
        return sorted(glob.glob(os.path.join(tmp, "page*.png")))
    if shutil.which("sips"):  # built into macOS (first page only)
        p = os.path.join(tmp, "page1.png")
        subprocess.run(["sips", "-s", "format", "png", path, "--out", p], check=True, capture_output=True)
        return [p] if os.path.exists(p) else []
    return []


def answer_scanned(question, path, mode):
    """No text layer: read the page images with the vision model instead."""
    pages = render_pdf_pages(path)
    if not pages:
        return None
    q = (question or "").strip() or "Summarise this document in a few sentences."
    answer = llm.chat(
        config.VL_MODEL,
        "The image(s) are scanned pages of a document. Read them carefully and answer the question "
        "using only what is written on the pages. If the pages do not contain the answer, reply exactly: "
        f"{prompts.not_found_message(mode)}\nQuestion: {q}",
        system=prompts.system_for(mode),
        images=[llm.image_bytes(p, max_side=1280) for p in pages],
        think=False,
    )
    return {"answer": answer,
            "details": f"_(Scanned PDF with no text layer: read {len(pages)} page(s) as images.)_",
            "models": config.VL_MODEL}


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def _parse(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None, None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, None
    quotes = data.get("quotes", data.get("quote", []))
    if isinstance(quotes, str):
        quotes = [quotes]
    quotes = [str(q).strip() for q in (quotes or []) if str(q).strip()]
    return str(data.get("answer", "")).strip(), quotes


def _quote_in_doc(quote, doc):
    q = _norm(quote)
    return bool(q) and fuzz.partial_ratio(q, _norm(doc)) >= config.QUOTE_MATCH_MIN


def _verify(question, answer, quote):
    reply = llm.chat(
        config.VL_MODEL,
        f"Question: {question}\nProposed answer: {answer}\nQuote from the document: \"{quote}\"\n\n"
        "Does the quote support this answer to the question? Ignore spelling mistakes and "
        "rewording in the question. Answer NO only if the quote does not support the answer, or "
        "it is about a different event, person, date, number or place than the question asks "
        "about. Reply with only YES or NO.",
        think=False,
    )
    return reply.strip().upper().startswith("YES")


def handle(question, path, mode="General", context=""):
    q = (question or "").strip() or "Summarise this document in a few sentences."
    doc = load_document(path)
    truncated_note = ""
    if len(doc) > config.MAX_DOC_CHARS:
        doc = doc[:config.MAX_DOC_CHARS]
        truncated_note = ("\n\n_(Long document: only the first part was read. "
                          "For best results, upload just the relevant section.)_")
    if not doc:
        if path.lower().endswith(".pdf"):
            try:
                scanned = answer_scanned(question, path, mode)
                if scanned:
                    return scanned
            except Exception as e:
                print(f"[text] scanned-PDF fallback failed: {e}")
        return {"answer": "I couldn't read any text from this document (it may be a scanned image).",
                "details": "", "models": config.VL_MODEL}

    if not (question or "").strip():  # no question: plain summary
        summary = llm.chat(config.VL_MODEL, f"DOCUMENT:\n<<<\n{doc}\n>>>\n\n{q}",
                           system=prompts.system_for(mode), think=False)
        return {"answer": summary, "details": truncated_note.strip(), "models": config.VL_MODEL}

    raw = llm.chat(
        config.VL_MODEL,
        f"DOCUMENT:\n<<<\n{doc}\n>>>\n"
        + (f"\nSESSION NOTES (from other files, NOT part of the document; use them only to understand "
           f"what the user refers to, e.g. a model number):\n{context}\n" if context else "")
        + f"\nQUESTION: {q}",
        system=SYSTEM,
        json_mode=True,
        think=False,
    )
    answer, quotes = _parse(raw)
    not_found = prompts.not_found_message(mode)

    if not answer or answer.upper().startswith("NOT_FOUND"):
        return {"answer": not_found, "details": truncated_note.strip(), "grounded": False, "models": config.VL_MODEL}

    quotes = [qt for qt in quotes if _quote_in_doc(qt, doc)]
    if not quotes:
        # none of the model's quotes is really in the document -> don't trust the answer
        return {"answer": not_found, "details": f"_(Rejected candidate “{answer}”: its quote was not found in the document.)_",
                "grounded": False, "models": config.VL_MODEL}
    quote = " | ".join(quotes)

    if config.TEXT_VERIFY and not _verify(q, answer, quote):
        return {"answer": not_found, "details": f"_(Rejected candidate “{answer}” with quote “{quote}”: the check said it doesn't answer this question.)_",
                "grounded": False, "models": config.VL_MODEL}

    sources = "\n".join(f"📄 “{qt}”" for qt in quotes)
    return {"answer": answer, "details": f"Sources:\n{sources}" + truncated_note, "grounded": True,
            "models": config.VL_MODEL}
