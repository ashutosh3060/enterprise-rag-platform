# enterprise-rag-platform

**Month 2 · Sep 2026** — Accurate, cited answers over large document collections, with the evaluation to prove it.

> Depends on [`ai-core`](https://github.com/ashutosh3060/ai-core) — the shared provider
> gateway, versioned cost accounting, and shared types used across this portfolio.

**The retrieval half runs with no API key.** Embeddings are local
(`sentence-transformers`), so ingestion, hybrid search, reranking, permission
filtering, and the full evaluation ablation work offline. Only answer *generation*
needs a provider.

---

## 1. Problem

Employees and customers cannot find answers buried in thousands of PDFs, wikis, and
contracts. Naive RAG demos work on a curated ten-document corpus and fall apart on
real enterprise data: part numbers, error codes, near-duplicate policy revisions,
acronyms that mean different things in different departments, and access control
that must actually hold.

## 2. Business Value

Support deflection and internal search are the two highest-ROI enterprise LLM use
cases, and the two where a wrong answer is most expensive. The value is not "a
chatbot over documents" — it is a system whose retrieval quality you can measure,
whose answers cite their sources, which never returns a withdrawn policy, and which
cannot leak a document the asker is not entitled to see.

## 3. Architecture

```
Documents (PDF · DOCX · Markdown · Web)
           |
  Ingestion — parse · normalise · version · attach metadata + ACLs
           |
     Chunking — fixed | recursive | semantic        (pluggable, ablated)
           |
   Local embeddings (sentence-transformers, CPU)
           |
     Vector store  ─────────────┐
           |                    |
       Retriever  ←──────── BM25 lexical
           |
     RRF fusion (rank-based, no score calibration)
           |
  Permission + version filter   ← inside the query, never after generation
           |
   Cross-encoder reranker       ← retrieve wide for recall, rerank narrow for precision
           |
      [ LLM generation + citations ]   ← the only stage needing an API key
```

## 4. Technology Choices

| Technology | Why this one |
|---|---|
| **sentence-transformers** (local) | Embedding runs on every chunk at ingest and every query at serve. An API round-trip per query is latency and cost nobody wants — and it makes the whole retrieval half runnable with no key. |
| **NumPy store (default) + Chroma** | The default store is exact brute-force. An ablation comparing retrieval strategies must not have ANN recall error mixed into the measurement. Chroma is there for scale; Qdrant is the production answer but needs a running service. |
| **rank-bm25** | The lexical leg. Exact matching is the only thing that separates `RMA-4471` from `RMA-4472`. |
| **RRF fusion** | Combines *ranks*, so no calibration is needed between a cosine similarity and a BM25 score — two quantities on incomparable scales. Weighted blending needs a per-corpus coefficient; RRF has one constant. |
| **Cross-encoder reranker** | Bi-encoder retrieval is fast because query and document never interact, which is also why it loses precision. A cross-encoder reads the pair together. |

> ⚠️ **Platform pins are load-bearing on x86 macOS.** PyTorch stopped publishing Intel
> Mac wheels after 2.2.2, and current `transformers` calls torch APIs that do not exist
> there. Unpinned installs fail with a misleading `NameError: name 'torch' is not
> defined`. See `pyproject.toml`.

## 5. Design Decisions

### Permissions filter inside the retrieval query, never after generation
Post-filtering means restricted content has already entered the model's context and
can leak through paraphrase. Both the vector store and BM25 apply the visibility
mask *before* ranking, so a filtered chunk cannot occupy a top-k slot.

### Documents are versioned, not overwritten
Enterprise policies supersede each other. Answering from a superseded revision is a
correctness bug, not a staleness annoyance — so `superseded` is a retrieval filter
and the ablation reports a `superseded` column that must stay at zero.

### Relevance is labelled per document, not per chunk
Several chunks of the same document would each count as a separate hit and inflate
precision. Only the first occurrence of a document counts.

### Queries whose correct answer is *nothing* are scored by leakage, not recall
Three queries ask for restricted content with the wrong role. Averaging a fabricated
1.0 into recall would mask real misses; they are excluded from recall and counted in
the `leaked` column instead, where any value above zero is a defect.

### The corpus contains 184 generated distractors
The twelve hand-written documents are the labelled gold set. On their own they are
not a benchmark — at k=5 you retrieve 42% of the corpus and **recall@5 is 1.000 for
every strategy**. That was the first result this project produced, and it measured
the corpus rather than the retrievers. The distractors are near misses: identifiers
differing by a digit, policies on adjacent topics.

## 6. Prior Art

LlamaIndex, LangChain, Haystack, and Vectara all do enterprise RAG, and every managed
offering (Azure AI Search, Vertex AI Search, Amazon Kendra) does it with more
polish. **If you need this in production, use one of them.**

What this repo has that a framework tutorial does not is the *measurement*: a
labelled query set built to expose where each retrieval strategy fails, and an
ablation that reports a result contradicting the usual advice (below). Frameworks
give you the components; they do not tell you which combination works on your corpus.

## 7. Trade-offs

- **Synthetic corpus.** Authored fiction, so the absolute numbers do not transfer.
  The *relative* ordering of strategies is the finding.
- **No knowledge graph.** Multi-hop questions are out of scope; query decomposition
  would be the cheaper next step.
- **Exact search by default.** Correct up to ~10⁵ chunks. Beyond that, ANN — and the
  recall/latency trade-off becomes another thing to measure.
- **Reranking triples query latency** (31ms → 107ms here). Worth it on these results;
  measure before assuming it is worth it on yours.

## 8. Evaluation Results

**196 documents · 20 labelled queries · k=5 · no API key required.** Reproduce with
`rag ablate -k 5`.

| Configuration | recall@5 | precision@5 | MRR | nDCG@5 | leaked | superseded | ms |
|---|---|---|---|---|---|---|---|
| `semantic + hybrid+rerank` | **1.000** | 0.246 | **1.000** | **1.000** | 0 | 0 | 156 |
| `fixed + hybrid+rerank` | 1.000 | 0.211 | 1.000 | 0.996 | 0 | 0 | 195 |
| `recursive + hybrid+rerank` | 1.000 | 0.211 | 1.000 | 0.996 | 0 | 0 | 107 |
| `fixed + bm25` | 0.944 | 0.339 | 0.944 | 0.938 | 0 | 0 | 0 |
| `recursive + bm25` | 0.944 | 0.339 | 0.944 | 0.938 | 0 | 0 | 0 |
| `semantic + bm25` | 0.944 | **0.365** | 0.944 | 0.938 | 0 | 0 | 1 |
| `fixed + hybrid` | 0.944 | 0.200 | 0.807 | 0.837 | 0 | 0 | 34 |
| `recursive + hybrid` | 0.944 | 0.200 | 0.807 | 0.837 | 0 | 0 | 31 |
| `semantic + hybrid` | 0.889 | 0.213 | 0.824 | 0.836 | 0 | 0 | 28 |
| `semantic + dense` | 0.833 | 0.206 | 0.833 | 0.829 | 0 | 0 | 28 |
| `fixed + dense` | 0.778 | 0.167 | 0.750 | 0.753 | 0 | 0 | 90 |
| `recursive + dense` | 0.778 | 0.167 | 0.750 | 0.753 | 0 | 0 | 31 |

### recall@5 by query kind — where the aggregate hides the story

| Configuration | acronym | identifier | mixed | paraphrase | versioned |
|---|---|---|---|---|---|
| `* + hybrid+rerank` | 1.000 | **1.000** | **1.000** | **1.000** | 1.000 |
| `* + bm25` | 1.000 | **1.000** | 0.833 | 1.000 | 1.000 |
| `fixed/recursive + hybrid` | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 |
| `semantic + dense` | 1.000 | **0.500** | 1.000 | 0.750 | 1.000 |
| `fixed/recursive + dense` | 1.000 | **0.500** | 1.000 | 0.750 | **0.500** |

## Findings

**1. Dense retrieval misses half the identifier queries — recall 0.500 vs 1.000 for
anything with a lexical leg.** This is the thesis, and it held. An embedding places
`RMA-4471` and `RMA-4472` in nearly the same region; only exact matching separates
them. On a corpus dense with part numbers and error codes, pure vector search
silently fails on exactly the queries a support engineer actually types.

**2. Hybrid without reranking was *worse* than BM25 alone — nDCG 0.837 vs 0.938.**
This contradicts the usual "hybrid beats both" advice, and it is the most useful
result here. RRF promoted dense's plausible-but-wrong neighbours into slots BM25 had
correctly filled, dragging MRR from 0.944 to 0.807. Fusion is not free: adding a
weaker retriever to a stronger one can dilute it. **Reranking is what recovers it**
— the cross-encoder re-sorts the fused candidates and lifts nDCG to 0.996–1.000.

The practical reading: *hybrid retrieval and reranking are one decision, not two.*
Shipping hybrid without a reranker, on a corpus like this, would have been a
regression against the simpler BM25 baseline.

**3. Dense retrieval returned superseded policy revisions on half the versioned
queries** (recall 0.500 with fixed/recursive chunking). It cannot distinguish v1 from
v2 of the same policy — the texts are near-identical in embedding space. Only the
explicit `superseded` filter prevents answering from a withdrawn policy.

**4. Access control held everywhere — zero leaked chunks across all 12
configurations.** Because filtering is inside the retrieval query rather than after
it, the invariant does not depend on which strategy is used.

**5. Reranking costs 3.5× query latency** (31ms → 107ms) for +0.16 nDCG. Cheap here;
worth re-measuring on a corpus where retrieval is not already near-perfect.

### Limitations

- Synthetic corpus of 196 documents. Absolute numbers do not transfer; the ordering
  is the finding.
- 20 labelled queries is small. Per-kind cells rest on 2–6 queries each, so treat
  individual cells as directional.
- BM25 scored 1.000 on paraphrase queries, higher than expected. The synthetic
  paraphrases likely retain more vocabulary overlap than genuine user phrasing would.
  A real corpus would probably widen dense's advantage there.
- No statistical significance testing at this sample size.

## 9. Demo

> _To be recorded._

## 10. Future Improvements

- **Generation with citations** — the last stage, needs an API key. Answers must cite
  chunk IDs; uncited claims get flagged.
- **Faithfulness and hallucination scoring** (Ragas) — also needs a key.
- **Query decomposition** for multi-hop questions.
- **Learned fusion weights** instead of vanilla RRF, given finding 2.
- **Incremental re-indexing** on document update rather than full re-ingest.

---

## Quickstart

```bash
git clone https://github.com/ashutosh3060/enterprise-rag-platform.git
cd enterprise-rag-platform
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

rag corpus                      # inspect the corpus and labelled queries
rag query "RMA-4471"            # hybrid retrieval
rag query "what do I do if production credentials leak" --roles engineering
rag ablate -k 5 -o docs/ablation.md
```

The second query returns **nothing** — that document is security-restricted, and
withholding it is the correct behaviour.

## Repository Layout

```
src/rag_platform/
  models.py       Document, Chunk, ScoredChunk — versioning and ACLs
  chunking.py     fixed | recursive | semantic
  embedding.py    local sentence-transformers
  store.py        NumpyStore (exact) + ChromaStore
  retrieval.py    dense, BM25, RRF hybrid, cross-encoder rerank
  corpus.py       synthetic gold set + 184 distractors + 20 labelled queries
  evaluation.py   recall/precision/MRR/nDCG + leakage, LLM-free
  pipeline.py     indexing and the ablation runner
tests/            20 tests, all offline
docs/             ablation.md
```

---

Part of a [6-month Product AI Engineer portfolio](https://github.com/ashutosh3060).
