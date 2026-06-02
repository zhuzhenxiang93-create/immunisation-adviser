# Section-Level Retrieval Ablation

Evaluated 35 hand-crafted questions with ground-truth sections.
Metric: Section Hit@8 and MRR.

| Method | Section Hit@8 | Rate | MRR |
|---|---:|---:|---:|
| bm25 only | 27/35 | 77.1% | 0.564 |
| vector only | 31/35 | 88.6% | 0.655 |
| hybrid rrf | 28/35 | 80.0% | 0.657 |
| hybrid rrf section rerank | 30/35 | 85.7% | 0.666 |

Note: This ablation is retrieval-only. It does not judge final answer correctness or whether generated citations fully support the answer.
