"""Score the app on the practice pack.

Usage (from the project folder, with Ollama running):
  python eval/run_eval.py                          # all 100 items
  python eval/run_eval.py --modality image --limit 5
  python eval/run_eval.py --ids p_txt_004 p_vid_001

Writes:
  eval/results/results_<time>.csv   every item, with prediction + reference (keep local, has answers)
  eval/summary.md                   accuracy + latency table (safe to commit, no answers)

Free-text scoring is approximate: look through the CSV rows marked wrong by hand.
"""
import argparse
import csv
import datetime
import json
import os
import re
import statistics
import string
import subprocess
import sys
import tempfile
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from handlers import llm, router  # noqa: E402

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
STOP = {"a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "at", "and", "it",
        "this", "that", "there", "with", "by", "for", "as", "be", "its", "i", "image", "video"}
REFUSAL_RE = re.compile(
    r"(does not say|doesn'?t say|does not (contain|mention|state|specify|provide)|"
    r"not (mentioned|stated|specified|provided|found|in the document)|no information|cannot be determined)",
    re.I)


# ---------------- scoring helpers ----------------
def normalize(s):
    s = (s or "").lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_tokens(s):
    return [stem(w) for w in normalize(s).split() if w not in STOP]


def recall(pred, ref):
    """Fraction of the reference's content words found in the prediction."""
    r = content_tokens(ref)
    if not r:
        return 0.0
    p = Counter(content_tokens(pred))
    hit = 0
    for w in r:
        if p[w] > 0:
            hit += 1
            p[w] -= 1
    return hit / len(r)


def first_number(s):
    for tok in normalize(s).split():
        if tok.isdigit():
            return int(tok)
        if tok in NUMBER_WORDS:
            return NUMBER_WORDS[tok]
    return None


def wer(pred, ref):
    p, r = normalize(pred).split(), normalize(ref).split()
    if not r:
        return 0.0 if not p else 1.0
    d = list(range(len(p) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(p) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != p[j - 1]))
            prev = cur
    return d[len(p)] / len(r)


def judge(question, ref, pred):
    """Local LLM grader for free-text answers that word overlap can't score fairly
    (e.g. 'OPEN, CLOSED' vs 'The sign says OPEN at the start and CLOSED at the end')."""
    reply = llm.chat(
        config.VL_MODEL,
        f"Question: {question}\nReference answer: {ref}\nModel answer: {pred}\n\n"
        "Is the model answer correct? It is correct if it gives the same key facts the question asks "
        "for as the reference (wording, length and extra detail don't matter). It is wrong if any key "
        "fact differs or is missing. Reply with only YES or NO.",
        max_tokens=3,
    )
    return reply.strip().upper().startswith("YES")


def run_code_tests(code, test_path):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "run.py")
        with open(f, "w") as fh:
            fh.write(code + "\n\n" + open(test_path).read())
        try:
            r = subprocess.run([sys.executable, f], capture_output=True, text=True, timeout=15, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, (r.stderr or "")[-300:]


def score(item, ans, result, data_dir):
    """Return (correct: bool, metric: float, note: str)."""
    m, q = item["modality"], item["question"]
    pred, ref = result.get("answer", ""), ans.get("reference_answer", "")

    if m == "code":
        test = os.path.join(data_dir, "code", "tests", f"{item['id']}_test.py")
        ok, err = run_code_tests(result.get("code", ""), test)
        return ok, float(ok), ("passed" if ok else "failed: " + err.strip().splitlines()[-1][:120] if err.strip() else "failed")

    if m == "text":
        refused = bool(REFUSAL_RE.search(pred))
        if ans.get("answerable") is False:
            return refused, float(refused), "unanswerable"
        rc = recall(pred, ref)
        if refused:
            return False, rc, "answerable"
        if rc >= 0.6:
            return True, rc, "answerable"
        ok = judge(q, ref, pred)
        return ok, rc, "answerable (judge)"

    if m == "audio" and "transcri" in q.lower():
        w = wer(pred, ans.get("transcript") or ref)
        return w <= 0.15, w, "WER"

    if q.lower().startswith("how many"):
        a, b = first_number(pred), first_number(ref)
        return (a is not None and a == b), float(a == b), f"pred={a} ref={b}"

    rc = recall(pred, ref)
    if rc >= 0.6:
        return True, rc, "recall"
    ok = judge(q, ref, pred)
    return ok, rc, "judge"


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=config.DATA_DIR)
    ap.add_argument("--modality", nargs="*", default=None, help="image audio text code video")
    ap.add_argument("--limit", type=int, default=None, help="max items per modality")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--mode", default="General")
    args = ap.parse_args()

    items = json.load(open(os.path.join(args.data, "items.json")))
    answers = {a["id"]: a for a in json.load(open(os.path.join(args.data, "answers.json")))}

    if args.ids:
        items = [it for it in items if it["id"] in set(args.ids)]
    if args.modality:
        items = [it for it in items if it["modality"] in set(args.modality)]
    if args.limit:
        seen = Counter()
        kept = []
        for it in items:
            if seen[it["modality"]] < args.limit:
                kept.append(it)
                seen[it["modality"]] += 1
        items = kept

    out_dir = os.path.join(ROOT, "eval", "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"results_{stamp}.csv")
    fields = ["id", "modality", "difficulty", "question", "reference", "prediction",
              "correct", "metric", "note", "latency_s"]

    print("Warming up…")
    router.warmup()

    rows = []
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for n, it in enumerate(items, 1):
            ans = answers[it["id"]]
            path = os.path.join(args.data, it["file"]) if it.get("file") else None
            t0 = time.time()
            try:
                result = router.handle(it["question"], path, args.mode)
                correct, metric, note = score(it, ans, result, args.data)
            except Exception as e:  # keep going if one item crashes
                import traceback
                traceback.print_exc()
                result, correct, metric, note = {"answer": f"ERROR: {e}"}, False, 0.0, "error"
            latency = time.time() - t0
            pred = result.get("code") if it["modality"] == "code" else result.get("answer", "")
            row = {"id": it["id"], "modality": it["modality"], "difficulty": ans.get("difficulty", ""),
                   "question": it["question"][:300], "reference": ans.get("reference_answer", ""),
                   "prediction": (pred or "")[:1500], "correct": int(correct),
                   "metric": round(metric, 3), "note": note, "latency_s": round(latency, 2)}
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"[{n}/{len(items)}] {it['id']:<12} {'✓' if correct else '✗'}  {latency:5.1f}s  {note}")

    # ---------- summary ----------
    lines = ["# Practice pack results", "",
             f"Run: {stamp} · mode: {args.mode} · models: {config.VL_MODEL}, {config.CODE_MODEL}, "
             f"whisper-{config.WHISPER_SIZE}, AST", "",
             "| Modality | Items | Correct | Accuracy | Median latency (s) | Max latency (s) |",
             "|---|---|---|---|---|---|"]
    for m in ["image", "audio", "text", "code", "video"]:
        rs = [r for r in rows if r["modality"] == m]
        if not rs:
            continue
        c = sum(r["correct"] for r in rs)
        lat = [r["latency_s"] for r in rs]
        lines.append(f"| {m} | {len(rs)} | {c} | {100 * c / len(rs):.0f}% | "
                     f"{statistics.median(lat):.1f} | {max(lat):.1f} |")
    total = sum(r["correct"] for r in rows)
    lines.append(f"| **all** | {len(rows)} | {total} | {100 * total / max(len(rows), 1):.0f}% | | |")

    txt = [r for r in rows if r["modality"] == "text"]
    if txt:
        un = [r for r in txt if r["note"] == "unanswerable"]
        an = [r for r in txt if r["note"].startswith("answerable")]
        lines += ["", f"Text: answerable {sum(r['correct'] for r in an)}/{len(an)} correct, "
                      f"unanswerable correctly refused {sum(r['correct'] for r in un)}/{len(un)}."]

    lines += ["", "Scoring: counts by exact number, transcription by word error rate (<=15%), code by the "
              "official unit tests, other free-text answers by word overlap, with a local LLM grader for "
              "answers worded differently from the reference. Mismatches spot-checked by hand."]
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "eval", "summary.md"), "w") as fh:
        fh.write(summary)
    print("\n" + summary)
    print(f"Per-item details: {csv_path}")


if __name__ == "__main__":
    main()
