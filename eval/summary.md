# Practice pack results

Run: 20261119_131156 · mode: General · models: qwen3-vl:4b-instruct, qwen2.5-coder:3b, whisper-small, AST

| Modality | Items | Correct | Accuracy | Median latency (s) | Max latency (s) |
|---|---|---|---|---|---|
| audio | 20 | 20 | 100% | 2.1 | 4.5 |
| video | 15 | 12 | 80% | 19.5 | 27.8 |
| **all** | 35 | 32 | 91% | | |

Scoring: counts by exact number, transcription by word error rate (<=15%), code by the official unit tests, other free-text answers by word overlap, with a local LLM grader for answers worded differently from the reference. Mismatches spot-checked by hand.
