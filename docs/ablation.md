# Retrieval ablation

Generated 2026-08-05 · k=5 · 20 labelled queries · no API key required.

Retrieval quality is a ranking question, so it is measured against labelled relevance rather than an LLM judge. Every number here is reproducible offline with `rag ablate`.

| Configuration | recall@k | precision@k | MRR | nDCG@k | leaked | superseded | ms |
|---|---|---|---|---|---|---|---|
| `semantic + hybrid+rerank` | 1.000 | 0.246 | 1.000 | 1.000 | 0 | 0 | 156 |
| `fixed + hybrid+rerank` | 1.000 | 0.211 | 1.000 | 0.996 | 0 | 0 | 195 |
| `recursive + hybrid+rerank` | 1.000 | 0.211 | 1.000 | 0.996 | 0 | 0 | 107 |
| `fixed + bm25` | 0.944 | 0.339 | 0.944 | 0.938 | 0 | 0 | 0 |
| `recursive + bm25` | 0.944 | 0.339 | 0.944 | 0.938 | 0 | 0 | 0 |
| `semantic + bm25` | 0.944 | 0.365 | 0.944 | 0.938 | 0 | 0 | 1 |
| `fixed + hybrid` | 0.944 | 0.200 | 0.807 | 0.837 | 0 | 0 | 34 |
| `recursive + hybrid` | 0.944 | 0.200 | 0.807 | 0.837 | 0 | 0 | 31 |
| `semantic + hybrid` | 0.889 | 0.213 | 0.824 | 0.836 | 0 | 0 | 28 |
| `semantic + dense` | 0.833 | 0.206 | 0.833 | 0.829 | 0 | 0 | 28 |
| `fixed + dense` | 0.778 | 0.167 | 0.750 | 0.753 | 0 | 0 | 90 |
| `recursive + dense` | 0.778 | 0.167 | 0.750 | 0.753 | 0 | 0 | 31 |

## By query type — recall@k

Aggregate scores hide the interesting behaviour. These columns are why hybrid retrieval exists.

| Configuration | acronym | identifier | mixed | paraphrase | versioned |
|---|---|---|---|---|---|
| `semantic + hybrid+rerank` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `fixed + hybrid+rerank` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `recursive + hybrid+rerank` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `fixed + bm25` | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 |
| `recursive + bm25` | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 |
| `semantic + bm25` | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 |
| `fixed + hybrid` | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 |
| `recursive + hybrid` | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 |
| `semantic + hybrid` | 1.000 | 0.750 | 1.000 | 0.750 | 1.000 |
| `semantic + dense` | 1.000 | 0.500 | 1.000 | 0.750 | 1.000 |
| `fixed + dense` | 1.000 | 0.500 | 1.000 | 0.750 | 0.500 |
| `recursive + dense` | 1.000 | 0.500 | 1.000 | 0.750 | 0.500 |

## Findings

**Best configuration: `semantic + hybrid+rerank`** — nDCG@5 1.000, recall@5 1.000, against `fixed + dense` at nDCG 0.753.

Read the per-kind table rather than the aggregate: the whole argument for a second index is that dense and lexical retrieval fail on *different* queries. If one strategy dominated every column, the other leg would be dead weight.

**Access control.** The `leaked` column counts chunks returned to a caller whose roles do not permit them. It is not a quality metric — any value above zero is a defect, because filtering happens inside the retrieval query rather than after generation.

**Versioning.** The `superseded` column counts retrieved chunks from superseded document revisions. Answering from a withdrawn policy is a correctness bug, not a staleness annoyance.

## Method

- 20 labelled queries across five kinds: identifier, paraphrase, acronym-collision, versioned, and mixed.
- Relevance is labelled per document; multiple chunks of the same document count once, so precision is not inflated by chunk granularity.
- Queries whose correct answer is *nothing* (restricted content, wrong role) are excluded from recall and scored by leakage instead — averaging a fabricated 1.0 into recall would mask real misses.
- Reranked configurations retrieve `candidate_k` and narrow to `k`.

