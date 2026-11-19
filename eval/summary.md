# Practice pack results

Run: 20261119_125628 · mode: General · models: qwen3-vl:4b-instruct, qwen2.5-coder:3b, whisper-small, AST

| Modality | Items | Correct | Accuracy | Median latency (s) | Max latency (s) |
|---|---|---|---|---|---|
| image | 25 | 24 | 96% | 7.0 | 10.2 |
| audio | 20 | 18 | 90% | 3.8 | 8.6 |
| text | 20 | 18 | 90% | 2.8 | 7.0 |
| code | 20 | 17 | 85% | 3.0 | 11.2 |
| video | 15 | 12 | 80% | 16.9 | 20.4 |
| **all** | 100 | 89 | 89% | | |

Text: answerable 14/15 correct, unanswerable correctly refused 4/5.

Scoring: counts by exact number, transcription by word error rate (<=15%), code by the official unit tests, other free-text answers by word overlap, with a local LLM grader for answers worded differently from the reference. Mismatches spot-checked by hand.
