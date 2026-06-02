# Section-Level Retrieval Ablation

Evaluated 35 hand-crafted questions with ground-truth sections.
Metric: Section Hit@11 and MRR.

| Method | Section Hit@8 | Rate | MRR |
|---|---:|---:|---:|
| bm25 only | 29/35 | 82.9% | 0.623 |
| vector only | 31/35 | 88.6% | 0.656 |
| hybrid rrf | 31/35 | 88.6% | 0.701 |

Note: This ablation is retrieval-only. It does not judge final answer correctness or whether generated citations fully support the answer.
