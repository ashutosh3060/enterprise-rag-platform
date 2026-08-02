# enterprise-rag-platform

**Month 2 · Sep 2026** — Accurate, cited answers over large document collections — with the evaluation to prove it.

> Depends on [`ai-core`](https://github.com/ashutosh3060/ai-core) — the shared provider gateway, cost accounting, and evaluation primitives used across this portfolio.

> **Status:** 🚧 Scaffolded. Implementation begins Sep 2026.
> Sections 7 (Evaluation Results) and 8 (Demo) are filled in as the work lands — they are
> the point of the repository, not an afterthought.

---

## 1. Problem

Employees and customers cannot find answers buried in thousands of PDFs, wikis, and contracts. Naive RAG demos work on a curated ten-document corpus and fall apart on real enterprise data: acronyms, near-duplicate versions, access control, and questions whose answer spans three documents.

## 2. Business Value

Support deflection and internal search are the two highest-ROI enterprise LLM use cases — and the two where a wrong answer is most expensive. The value is not 'a chatbot over documents'; it is a system whose faithfulness you can measure, whose answers cite their sources, and which cannot leak a document the asker is not entitled to see.

## 3. Architecture

```
Documents (PDF · DOCX · Markdown · Web)
           |
 Ingestion pipeline  ── parse · normalize · version · attach metadata + ACLs
           |
      Chunking       ── fixed | recursive | semantic  (pluggable)
           |
  Embedding model
           |
   Vector database  ────────┐
           |                |
       Retriever   ←── BM25 (hybrid, RRF fusion)
           |
 Metadata + permission filter   (applied at retrieval, not after)
           |
      Reranker (cross-encoder)
           |
         LLM
           |
 Answer + inline citations  →  feedback loop → labelled eval set
```

## 4. Technology Choices

| Technology | Why this one |
|---|---|
| **LlamaIndex** | Best-in-class document ingestion and node/metadata model. Used for the pipeline; retrieval orchestration stays explicit rather than hidden in an abstraction. |
| **Qdrant** | Native hybrid search, payload filtering, and named vectors. Payload filters are what make pre-retrieval permission enforcement possible. |
| **Cross-encoder reranker** | Bi-encoder retrieval optimizes for recall; a cross-encoder restores precision on the top-k. This is consistently the single largest quality win in the pipeline. |
| **Ragas** | Standard RAG metrics — context precision/recall, faithfulness, answer relevancy — so results are comparable to published baselines instead of bespoke. |

## 5. Design Decisions

### 1. Permissions filter at retrieval time, never after generation

Post-filtering means restricted content has already entered the model's context and can leak through paraphrase. Filtering in the vector query is the only version that is actually a security boundary.

### 2. Hybrid retrieval (dense + BM25, fused with RRF) is the default

Enterprise corpora are dense with identifiers, acronyms, error codes, and part numbers — exactly the tokens dense embeddings handle worst and lexical search handles best. Pure-dense retrieval underperforms badly on real queries.

### 3. Answers must cite chunk IDs; uncited claims are flagged

An answer without provenance cannot be audited or trusted. Citations also give the faithfulness scorer something concrete to verify against.

### 4. Documents are versioned, not overwritten

Enterprise policies supersede each other. Answering from a superseded revision is a correctness bug, not a staleness annoyance — so version is a first-class retrieval filter.

## 6. Trade-offs

What this project deliberately does **not** do, and why:

- Semantic chunking costs an extra embedding pass at ingest. Accepted: ingest is a batch cost paid once, retrieval quality is paid on every query.
- Reranking adds latency to every query. Mitigated by reranking only the top-k (k≈25 → 5) rather than the full candidate set.
- No knowledge graph. Multi-hop questions are handled with query decomposition instead — far less infrastructure for most of the benefit on this corpus type.

## 7. Evaluation Results

> _To be populated during Sep 2026._
> Real, measured numbers only — no estimates. See [`docs/evaluation.md`](docs/evaluation.md)
> for methodology and [`docs/cost-analysis.md`](docs/cost-analysis.md) for the cost breakdown.

## 8. Demo

> _2–4 minute walkthrough — to be recorded at the end of Sep 2026._

## 9. Future Improvements

- Query decomposition and multi-hop retrieval for questions spanning several documents.
- Incremental re-indexing on document update instead of full re-ingest.
- Automatic hard-negative mining from thumbs-down feedback to fine-tune the reranker.

---

## Quickstart

```bash
git clone https://github.com/ashutosh3060/enterprise-rag-platform.git
cd enterprise-rag-platform

python -m venv .venv && source .venv/bin/activate
make install

cp .env.example .env      # add ANTHROPIC_API_KEY (the only required key)
python -m ai_core.probe   # confirm which providers are reachable
```

Everything except Anthropic is optional — the gateway registers a provider only when its key
is present, and each view renders whatever is available.

## Repository Layout

```
src/rag_platform/    application code
tests/             unit + integration tests
docs/              architecture · design-decisions · evaluation · cost-analysis · future-roadmap
```

## Documentation

- [Architecture](docs/architecture.md)
- [Design Decisions](docs/design-decisions.md)
- [Evaluation](docs/evaluation.md)
- [Cost Analysis](docs/cost-analysis.md)
- [Future Roadmap](docs/future-roadmap.md)

---

Part of a [6-month Product AI Engineer portfolio](https://github.com/ashutosh3060) —
`ai-core` · `llm-engineering-playground` · `enterprise-rag-platform` ·
`multi-agent-ai-platform` · `llm-evaluation-platform` · `production-ai-assistant` ·
`ai-model-router`
