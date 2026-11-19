# Practice pack results

Run: 20261119_161912 · mode: General · models: qwen3-vl:4b-instruct, qwen2.5-coder:3b, whisper-small, AST

| Modality | Items | Correct | Accuracy | Median latency (s) | Max latency (s) |
|---|---|---|---|---|---|
| image | 25 | 24 | 96% | 7.0 | 8.2 |
| audio | 20 | 20 | 100% | 2.4 | 7.4 |
| text | 20 | 19 | 95% | 2.4 | 7.1 |
| code | 20 | 17 | 85% | 2.5 | 9.9 |
| video | 15 | 12 | 80% | 14.7 | 20.0 |
| **all** | 100 | 92 | 92% | | |

Text: answerable 15/15 correct, unanswerable correctly refused 4/5.

Scoring: counts by exact number, transcription by word error rate (<=15%), code by the official unit tests, other free-text answers by word overlap, with a local LLM grader for answers worded differently from the reference. Mismatches spot-checked by hand.
