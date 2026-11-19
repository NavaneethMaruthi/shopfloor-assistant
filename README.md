# Shopfloor Assistant

An offline, multimodal AI assistant for **maintenance technicians in small and mid-size plants**, built for the IEEE AI Hackathon 2026 (Multimodal LLM Challenge). Everything runs locally on a laptop with open-weight models, with no internet connection and no cloud API.

## The problem

When a machine fails mid-shift, a technician has to piece together answers from a long manual, a faded nameplate, an odd noise and a log file. In many plants the machines sit on networks that are cut off from the internet on purpose (operational-technology networks are isolated for safety and security), so cloud AI assistants can't be used on the shop floor. Offline industrial AI exists, but it's sold as enterprise server platforms that a 40-person shop can't justify.

US manufacturing may need up to 3.8 million new workers between 2024 and 2033, and about 1.9 million of those jobs could go unfilled; roughly 2.8 million of the openings come from retirements (Deloitte and The Manufacturing Institute, 2024). Experienced technicians are leaving, newer ones need help, and that help can't come from the cloud.

## What it does

One chat interface handles five kinds of input. Attach a file and ask about it; follow-up questions reuse the last file.

| Input | What it does | Technician example |
|---|---|---|
| Image | Describes photos, counts objects, reads text | "What's the rated current on this nameplate?" |
| Audio | Transcribes speech, answers questions about it, identifies sounds | "What's that sound?" (vacuum cleaner, 0.75 confidence) |
| Document (PDF/TXT) | Answers from the document only, cites the source line, and says so when the answer isn't there | "The motor is humming loudly. What should I check?" |
| Code | Writes Python, runs it, and fixes it if it fails | "Flag a motor as overloaded above 115% of full-load amps" |
| Video | Describes short clips: counting, motion, order of events, signs | "What's happening in this video?" |

**Technician mode** adds a shop-floor persona: reads labels exactly, never invents specifications or safety steps, points to the manual or a supervisor when information is missing, and reminds about lockout/tagout. **General mode** (default) answers everyday questions.

## Architecture

```
Browser (Gradio chat UI, one input box for every file type)
   │
app.py ── router (by file type, or code intent)
   ├─ Image  → Qwen3-VL 4B Instruct
   ├─ Video  → ffmpeg 8 fps → keep changed frames (max 9) → ONE labelled grid image → Qwen3-VL
   │            + measurements: motion tracker (direction, bounces), start/end colours,
   │              audio track → sound classifier / Whisper
   ├─ Audio  → AST sound classifier → speech? → faster-whisper → Qwen3-VL answers from transcript
   │                                  → sound? → Qwen3-VL names the source from the top-10 labels
   ├─ Text   → whole document in context → JSON {answer, quotes} → quotes checked against the
   │            document (fuzzy match) → second check that the quotes answer this question
   └─ Code   → Qwen2.5-Coder 3B → run in a sandboxed subprocess against the prompt's own
                examples (>>> and "f(x) == y" style) → feed errors back, retry up to 2 times
```

All models are served by **Ollama** (Qwen) or run in-process (Whisper, AST). Offline flags are set at startup so nothing is downloaded at run time.

### Models and why

| Model | Role | Why this one |
|---|---|---|
| `qwen3-vl:4b-instruct` | Images, video, documents, audio Q&A | Strong OCR and visual reasoning at 4B parameters. The **instruct** build answers directly; the default reasoning build spent 10–30 s generating hidden reasoning per question. |
| `qwen2.5-coder:3b` | Code | The 7B version plus the vision model didn't fit in 16 GB alongside the audio models, causing heavy swapping (70 s per image). The 3B scored the same on our first code items and is faster. |
| faster-whisper `small` (int8, CPU) | Speech-to-text | Accurate on clear speech, small, fast on Apple Silicon CPU. |
| AST fine-tuned on AudioSet | Sound identification | Whisper only transcribes speech; AST recognises 527 sound types, including machines and alarms. |

### Design decisions that mattered

- **Instruct instead of reasoning build:** document questions went from about 30 s to under 9 s (first question) and under 1 s (follow-ups, thanks to Ollama's prompt caching).
- **Video as one grid image:** Ollama gives each image roughly the same token budget (about 1,000 tokens), so 10 separate frames overflowed the context window. One labelled grid of change-frames costs about 1,300–1,700 tokens total. Video went from 25% at 90–130 s to 100% at about 5 s on the first four items.
- **Measurements instead of guessing:** a simple motion tracker (median background + centroid) answers direction and bounce questions, and colour measurement answers background questions. A grid can't show motion between frames; the tracker can.
- **Grounded documents:** every answer must quote the document; quotes are checked against the text, and a second check rejects quotes that don't answer the exact question. This is what makes "the document does not say" reliable.
- **Self-repairing code:** generated code is executed against the examples written in the prompt itself (never hidden tests), and failures are fed back to the model.
- **Output length cap:** stops occasional repetition loops in the small model.

## Hardware

MacBook Air M4, 16 GB unified memory, fanless. All results below were measured on this machine.

## Test results (practice pack, 100 items: 92/100)

| Modality | Items | Accuracy | Median time |
|---|---|---|---|
| Image | 25 | 96% | 6.8 s |
| Audio | 20 | 100% | 2.6 s |
| Text | 20 | 95% | 2.3 s |
| Code | 20 | 85% | 2.8 s |
| Video | 15 | 80% | 16.1 s |

Text: all 15 answerable questions correct; 4 of 5 unanswerable questions correctly refused. The latest full run is written to `eval/summary.md`.

Scoring: counts by exact number, transcription by word error rate (≤15%), code by the official unit tests, other free-text answers by word overlap, with a local LLM grader for answers worded differently from the reference. Mismatches were spot-checked by hand. The final run was measured with Wi-Fi off.

## Setup

Requirements: macOS or Linux, Python 3.11, ffmpeg, Ollama.

```bash
brew install ffmpeg python@3.11 ollama
ollama pull qwen3-vl:4b-instruct
ollama pull qwen2.5-coder:3b

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# one-time download of the speech and sound models (needs internet once)
HF_HUB_OFFLINE=0 python -c "from faster_whisper import WhisperModel; WhisperModel('small')"
HF_HUB_OFFLINE=0 python -c "from transformers import pipeline; pipeline('audio-classification', model='MIT/ast-finetuned-audioset-10-10-0.4593')"
```

## Run

```bash
# terminal 1
export OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=2
ollama serve

# terminal 2
source .venv/bin/activate
python app.py          # then open http://localhost:7860
```

Evaluate on the practice pack (place it in `data/`, which is git-ignored):

```bash
python eval/run_eval.py                     # all items
python eval/run_eval.py --modality video    # one type
```

## Limits

- **Short inputs:** documents are read up to about 24,000 characters (longer ones are cut, with a note); video works best under 30 s; audio sound labels are general (e.g. "squeal"), not fault diagnoses.
- **Sound is triage, not diagnosis:** the classifier can say "whine", not "bearing failure"; the manual lookup does the rest.
- **Counting in crowded scenes** (overlapping animals) and **counting many objects in video** are the weakest areas.
- **Code:** the 3B model misses some harder problems (17/20 on the practice pack).
- **Fanless laptop:** after about 15 minutes of continuous load, video slows from about 5–10 s to 15–20 s.
- **Desktop browser only:** the interface is a web app, but phone access was not verified on our hardware.
- **English** prompts and documents.

## Repository layout

```
app.py              Gradio interface
config.py           models and settings (offline flags are set here)
prompts.py          General and Technician system prompts
handlers/           image, audio, text, code, video, router, llm (Ollama wrapper)
eval/run_eval.py    scores the practice pack, writes eval/summary.md
```

The practice-pack data and demo files are kept local and never committed, as the dataset rules require.