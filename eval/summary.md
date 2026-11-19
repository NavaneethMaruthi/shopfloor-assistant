# Practice pack results

Run: 20261119_134402 · mode: General · models: qwen3-vl:4b-instruct, qwen2.5-coder:3b, whisper-small, AST

| Modality | Items | Correct | Accuracy | Median latency (s) | Max latency (s) |
|---|---|---|---|---|---|
| text | 20 | 19 | 95% | 2.2 | 5.8 |
| **all** | 20 | 19 | 95% | | |

Text: answerable 15/15 correct, unanswerable correctly refused 4/5.

Scoring: counts by exact number, transcription by word error rate (<=15%), code by the official unit tests, other free-text answers by word overlap, with a local LLM grader for answers worded differently from the reference. Mismatches spot-checked by hand.
