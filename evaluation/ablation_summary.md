# Content-Based Retrieval Recall

Evaluated 55 questions.
Metric: Recall@10 — at least 40% of ground-truth answer
keywords must appear in any one retrieved chunk.

| Method | Recall@10 | Rate | MRR |
|---|---:|---:|---:|
| bm25 only | 47/55 | 85.5% | 0.654 |
| vector only | 48/55 | 87.3% | 0.714 |
| hybrid rrf | 49/55 | 89.1% | 0.705 |

Note: A question is counted as hit if any of the top-10 retrieved chunks
contains the key clinical facts from the ground truth answer.
This is a content-level check, not a section metadata match.
