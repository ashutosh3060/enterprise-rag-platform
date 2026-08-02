# Architecture — enterprise-rag-platform

## System Diagram

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

## Components

_One subsection per box above: responsibility, inputs, outputs, failure modes,
and what happens when its dependency is unavailable._

## Data Flow

_End-to-end trace of a single representative request, with the data shape at each hop._

## Scaling Considerations

_Where this design breaks under load, and the first thing that would need to change._
