"""Code: write working Python from a prompt.

Self-repair loop: run the generated code in a subprocess, check it against any
>>> examples written in the user's own prompt (doctest), and if it crashes or an
example fails, send the error back to the model and retry.
Uses only the prompt itself, never hidden tests or answers.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import config
from handlers import llm

SYSTEM = (
    "You are an expert Python programmer. Write correct, complete Python 3 code. "
    "Reply with a single ```python code block containing the full implementation, "
    "including the function signature and any imports. "
    "No explanations, no example usage, no tests, no input() calls."
)

HARNESS = r'''
import doctest, io, json, sys
g = {"__name__": "candidate"}
src = open(sys.argv[1]).read()
exec(compile(src, "candidate.py", "exec"), g)
text = open(sys.argv[2]).read()
# closing triple quotes would otherwise be read as part of the expected output
text = text.replace('"' * 3, "\n").replace("'" * 3, "\n")
try:
    test = doctest.DocTestParser().get_doctest(text, g, "prompt_examples", "prompt", 0)
except ValueError:
    test = None
if test is None or not test.examples:
    print("RESULT " + json.dumps({"attempted": 0, "failed": 0}))
    sys.exit(0)
buf = io.StringIO()
runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
res = runner.run(test, out=buf.write)
print(buf.getvalue()[-2000:])
print("RESULT " + json.dumps({"attempted": res.attempted, "failed": res.failed}))
'''

TYPING_RE = re.compile(r"\b(List|Dict|Tuple|Optional|Any|Union|Set)\[")


def extract_code(reply):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
    code = max(blocks, key=len) if blocks else reply
    code = code.strip()
    if TYPING_RE.search(code) and "from typing" not in code and "import typing" not in code:
        code = "from typing import *\n\n" + code
    return code


def self_check(code, prompt):
    """Return (ok, report, n_examples)."""
    with tempfile.TemporaryDirectory() as d:
        cpath, ppath, hpath = (os.path.join(d, n) for n in ("candidate.py", "prompt.txt", "harness.py"))
        open(cpath, "w").write(code)
        open(ppath, "w").write(prompt)
        open(hpath, "w").write(HARNESS)
        try:
            r = subprocess.run([sys.executable, hpath, cpath, ppath], capture_output=True,
                               text=True, timeout=config.CODE_TIMEOUT_S, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "The code timed out (possible infinite loop).", 0
    if r.returncode != 0:
        return False, (r.stderr or r.stdout)[-1500:], 0
    m = re.search(r"RESULT (\{.*\})", r.stdout)
    stats = json.loads(m.group(1)) if m else {"attempted": 0, "failed": 0}
    if stats["failed"]:
        return False, r.stdout.split("RESULT")[0][-1500:], stats["attempted"]
    return True, "", stats["attempted"]


def generate(prompt, previous=None, error=None):
    if previous is None:
        user = prompt
    else:
        user = (f"{prompt}\n\nYour previous code:\n```python\n{previous}\n```\n"
                f"It failed with:\n{error}\n\nFix the code. Reply with a single python code block.")
    return extract_code(llm.chat(config.CODE_MODEL, user, system=SYSTEM))


def handle(question, path=None, mode="General"):
    prompt = (question or "").strip()
    code = generate(prompt)
    ok, report, n_ex = self_check(code, prompt)
    repairs = 0
    while not ok and repairs < config.CODE_MAX_REPAIRS:
        repairs += 1
        code = generate(prompt, previous=code, error=report)
        ok, report, n_ex = self_check(code, prompt)

    if ok and n_ex:
        status = f"✅ Passed all {n_ex} examples from your prompt"
    elif ok:
        status = "✅ Code runs without errors (no examples in the prompt to test against)"
    else:
        status = "⚠️ Still failing a check after retries — review before using"
    if repairs:
        status += f" · fixed itself {repairs}× after a failed check"

    return {"answer": f"```python\n{code}\n```", "details": status, "code": code,
            "passed_self_check": ok, "repairs": repairs, "models": config.CODE_MODEL}
