# Practice pack results

Run: 20261119_121916 · mode: General · models: qwen3-vl:4b-instruct, qwen2.5-coder:7b, whisper-small, AST

| Modality | Items | Correct | Accuracy | Median latency (s) | Max latency (s) |
|---|---|---|---|---|---|
| video | 4 | 4 | 100% | 5.3 | 8.6 |
| **all** | 4 | 4 | 100% | | |

Scoring: counts by exact number, transcription by word error rate (<=15%), code by the official unit tests, other free-text answers by word overlap, with a local LLM grader for answers worded differently from the reference. Mismatches spot-checked by hand.
