# Design Decisions — enterprise-rag-platform

Architecture decision records. Each captures the context, the decision, and — once
implemented — what the decision actually cost.

## ADR-001: Permissions filter at retrieval time, never after generation

**Status:** Accepted

**Context & Decision**

Post-filtering means restricted content has already entered the model's context and can leak through paraphrase. Filtering in the vector query is the only version that is actually a security boundary.

**Consequences**

_To be recorded once implemented — including anything this decision made harder._

## ADR-002: Hybrid retrieval (dense + BM25, fused with RRF) is the default

**Status:** Accepted

**Context & Decision**

Enterprise corpora are dense with identifiers, acronyms, error codes, and part numbers — exactly the tokens dense embeddings handle worst and lexical search handles best. Pure-dense retrieval underperforms badly on real queries.

**Consequences**

_To be recorded once implemented — including anything this decision made harder._

## ADR-003: Answers must cite chunk IDs; uncited claims are flagged

**Status:** Accepted

**Context & Decision**

An answer without provenance cannot be audited or trusted. Citations also give the faithfulness scorer something concrete to verify against.

**Consequences**

_To be recorded once implemented — including anything this decision made harder._

## ADR-004: Documents are versioned, not overwritten

**Status:** Accepted

**Context & Decision**

Enterprise policies supersede each other. Answering from a superseded revision is a correctness bug, not a staleness annoyance — so version is a first-class retrieval filter.

**Consequences**

_To be recorded once implemented — including anything this decision made harder._
