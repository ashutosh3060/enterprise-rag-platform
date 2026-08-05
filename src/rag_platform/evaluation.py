"""Retrieval evaluation.

Deliberately **LLM-free**. Retrieval quality is a ranking question — did the
relevant document appear, and how high — and answering it needs labelled
relevance, not a language model. That is why the Month 2 headline result can be
produced with no API key at all.

Answer-level metrics (faithfulness, hallucination rate) are a different problem,
require generation, and live in `generation.py` behind a provider.

Metrics chosen for what they each expose:

- **recall@k** — did we retrieve the answer at all? A generator cannot recover
  from a miss here, so this is the ceiling on end-to-end quality.
- **precision@k** — how much irrelevant text is in the context window? Costs
  money and dilutes attention.
- **MRR** — how high did the first relevant result land? Position matters
  because context is read with a recency and primacy bias.
- **nDCG@k** — rank-sensitive overall quality.
- **leakage** — how many restricted chunks reached a caller who should not see
  them. Not a quality metric: any value above zero is a defect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .corpus import EvalQuery
from .models import ScoredChunk

__all__ = ["EvalResult", "Retriever", "evaluate", "ndcg_at_k", "precision_at_k", "recall_at_k"]


class Retriever(Protocol):
    name: str

    def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        roles: set[str] | None = None,
        include_superseded: bool = False,
    ) -> list[ScoredChunk]: ...


def _relevant_hits(results: list[ScoredChunk], relevant: set[str]) -> list[bool]:
    """Per-rank relevance, deduplicated by document.

    Chunks are the retrieval unit but relevance is labelled per document, so
    several chunks of the same document would otherwise each count as a separate
    hit and inflate precision. Only the first occurrence of a document counts.
    """
    seen: set[str] = set()
    hits: list[bool] = []
    for scored in results:
        doc_id = scored.chunk.doc_id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        hits.append(doc_id in relevant)
    return hits


def recall_at_k(results: list[ScoredChunk], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents retrieved in the top k.

    Undefined when nothing is relevant — those queries are scored by leakage
    instead, and averaging a fabricated 1.0 into recall would mask real misses.
    """
    if not relevant:
        return float("nan")
    found = {s.chunk.doc_id for s in results[:k]} & relevant
    return len(found) / len(relevant)


def precision_at_k(results: list[ScoredChunk], relevant: set[str], k: int) -> float:
    hits = _relevant_hits(results, relevant)[:k]
    return (sum(hits) / len(hits)) if hits else 0.0


def reciprocal_rank(results: list[ScoredChunk], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")
    for rank, hit in enumerate(_relevant_hits(results, relevant), start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[ScoredChunk], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    hits = _relevant_hits(results, relevant)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(hits) if hit)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


@dataclass
class EvalResult:
    """Aggregate scores for one retrieval configuration."""

    config: str
    n_queries: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    leaked_chunks: int
    leaked_queries: int
    superseded_hits: int
    by_kind: dict[str, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0

    def as_row(self) -> dict[str, object]:
        return {
            "config": self.config,
            "recall@k": round(self.recall_at_k, 4),
            "precision@k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@k": round(self.ndcg_at_k, 4),
            "leaked": self.leaked_chunks,
            "superseded": self.superseded_hits,
            "ms": round(self.mean_latency_ms, 1),
            **{f"recall[{kind}]": round(v, 4) for kind, v in sorted(self.by_kind.items())},
        }


def evaluate(
    retriever: Retriever,
    queries: list[EvalQuery],
    *,
    k: int = 5,
    config_name: str | None = None,
    rerank: object | None = None,
    candidate_k: int = 30,
) -> EvalResult:
    """Score a retrieval configuration over a labelled query set.

    When `rerank` is supplied the retriever fetches `candidate_k` and the
    reranker narrows to `k` — retrieve wide for recall, rerank narrow for
    precision.
    """
    import statistics
    import time

    recalls: list[float] = []
    precisions: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    latencies: list[float] = []
    by_kind: dict[str, list[float]] = {}

    leaked_chunks = 0
    leaked_queries = 0
    superseded_hits = 0

    for query in queries:
        started = time.perf_counter()
        fetch = candidate_k if rerank is not None else k
        results = retriever.retrieve(query.text, k=fetch, roles=query.roles)
        if rerank is not None:
            results = rerank.rerank(query.text, results, k=k)  # type: ignore[attr-defined]
        results = results[:k]
        latencies.append((time.perf_counter() - started) * 1000)

        # Access control is not a quality metric — it is a correctness invariant.
        # Any chunk the caller's roles do not permit is a defect regardless of
        # how well the ranking scored.
        leaks = [s for s in results if not s.chunk.visible_to(query.roles)]
        leaked_chunks += len(leaks)
        leaked_queries += 1 if leaks else 0
        superseded_hits += sum(1 for s in results if s.chunk.superseded)

        if not query.relevant_doc_ids:
            # Nothing should be returned. Scored only by leakage above.
            continue

        r = recall_at_k(results, query.relevant_doc_ids, k)
        recalls.append(r)
        precisions.append(precision_at_k(results, query.relevant_doc_ids, k))
        rrs.append(reciprocal_rank(results, query.relevant_doc_ids))
        ndcgs.append(ndcg_at_k(results, query.relevant_doc_ids, k))
        by_kind.setdefault(query.kind, []).append(r)

    return EvalResult(
        config=config_name or retriever.name,
        n_queries=len(queries),
        recall_at_k=statistics.fmean(recalls) if recalls else 0.0,
        precision_at_k=statistics.fmean(precisions) if precisions else 0.0,
        mrr=statistics.fmean(rrs) if rrs else 0.0,
        ndcg_at_k=statistics.fmean(ndcgs) if ndcgs else 0.0,
        k=k,
        leaked_chunks=leaked_chunks,
        leaked_queries=leaked_queries,
        superseded_hits=superseded_hits,
        by_kind={kind: statistics.fmean(v) for kind, v in by_kind.items()},
        mean_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
    )
