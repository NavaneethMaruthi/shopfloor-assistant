"""Text: answer questions about a document, and say so when the answer isn't there.

1. The model must return JSON: {"answer": ..., "quote": exact supporting sentence}.
2. Grounding check: the quote must really appear in the document (fuzzy match).
3. Verification: a second short call checks the quote answers THIS question
   (catches near-miss traps like a different year or place).
Any failure -> "The document does not say."
"""
import json
import re
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
    '- Reply with JSON only: {"answer": "<short answer>", "quote": "<the exact sentence from the '
    'document that states the answer>"}\n'
    '- If the document does not contain the answer, reply: {"answer": "NOT_FOUND", "quote": ""}'
)


@lru_cache(maxsize=16)
def load_document(path):
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages).strip()
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().strip()


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
    return str(data.get("answer", "")).strip(), str(data.get("quote", "")).strip()


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


def handle(question, path, mode="General"):
    q = (question or "").strip() or "Summarise this document in a few sentences."
    doc = load_document(path)
    truncated_note = ""
    if len(doc) > config.MAX_DOC_CHARS:
        doc = doc[:config.MAX_DOC_CHARS]
        truncated_note = ("\n\n_(Long document: only the first part was read. "
                          "For best results, upload just the relevant section.)_")
    if not doc:
        return {"answer": "I couldn't read any text from this document (it may be a scanned image).",
                "details": "", "models": config.VL_MODEL}

    if not (question or "").strip():  # no question: plain summary
        summary = llm.chat(config.VL_MODEL, f"DOCUMENT:\n<<<\n{doc}\n>>>\n\n{q}",
                           system=prompts.system_for(mode), think=False)
        return {"answer": summary, "details": truncated_note.strip(), "models": config.VL_MODEL}

    raw = llm.chat(
        config.VL_MODEL,
        f"DOCUMENT:\n<<<\n{doc}\n>>>\n\nQUESTION: {q}",
        system=SYSTEM,
        json_mode=True,
        think=False,
    )
    answer, quote = _parse(raw)
    not_found = prompts.not_found_message(mode)

    if not answer or answer.upper().startswith("NOT_FOUND"):
        return {"answer": not_found, "details": truncated_note.strip(), "grounded": False, "models": config.VL_MODEL}

    if not _quote_in_doc(quote, doc):
        # model's quote isn't really in the document -> don't trust the answer
        return {"answer": not_found, "details": f"_(Rejected candidate “{answer}”: its quote was not found in the document.)_",
                "grounded": False, "models": config.VL_MODEL}

    if config.TEXT_VERIFY and not _verify(q, answer, quote):
        return {"answer": not_found, "details": f"_(Rejected candidate “{answer}” with quote “{quote}”: the check said it doesn't answer this question.)_",
                "grounded": False, "models": config.VL_MODEL}

    return {"answer": answer, "details": f"📄 Source: “{quote}”" + truncated_note, "grounded": True,
            "models": config.VL_MODEL}
