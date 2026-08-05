"""Ingestion and retrieval pipeline, plus the ablation runner.

The ablation is the point of this project: the same corpus and the same labelled
queries, run across every combination of chunking strategy and retrieval
strategy, so "hybrid plus reranking is better" becomes a number rather than an
assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import get_chunker
from .corpus import EvalQuery, build_full_corpus, build_queries
from .embedding import Embedder, get_embedder
from .evaluation import EvalResult, evaluate
from .models import Chunk, Document
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Reranker
from .store import NumpyStore, VectorStore

__all__ = ["Index", "build_index", "run_ablation"]


@dataclass
class Index:
    """An ingested corpus, ready to query."""

    chunks: list[Chunk]
    store: VectorStore
    embedder: Embedder
    chunker_name: str

    def dense(self) -> DenseRetriever:
        return DenseRetriever(self.store, self.embedder)

    def bm25(self) -> BM25Retriever:
        return BM25Retriever(self.chunks)

    def hybrid(self, candidate_k: int = 30) -> HybridRetriever:
        return HybridRetriever(self.dense(), self.bm25(), candidate_k=candidate_k)


def build_index(
    documents: list[Document],
    *,
    chunker: str = "recursive",
    target_tokens: int = 200,
    overlap_tokens: int = 40,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> Index:
    """Chunk, embed, and index a document set."""
    emb = embedder or get_embedder()

    kwargs: dict[str, object] = {
        "target_tokens": target_tokens,
        "overlap_tokens": overlap_tokens,
    }
    if chunker == "semantic":
        # The semantic chunker needs an embedder to find topic boundaries; the
        # others do not, and passing one would be an unused-argument error.
        kwargs["embedder"] = emb
        kwargs["overlap_tokens"] = 0

    splitter = get_chunker(chunker, **kwargs)
    chunks = [c for doc in documents for c in splitter.chunk(doc)]

    vectors = emb.encode([c.text for c in chunks])
    vstore = store if store is not None else NumpyStore()
    vstore.add(chunks, vectors)

    return Index(chunks=chunks, store=vstore, embedder=emb, chunker_name=chunker)


def run_ablation(
    documents: list[Document] | None = None,
    queries: list[EvalQuery] | None = None,
    *,
    chunkers: tuple[str, ...] = ("fixed", "recursive", "semantic"),
    strategies: tuple[str, ...] = ("dense", "bm25", "hybrid", "hybrid+rerank"),
    k: int = 5,
    candidate_k: int = 30,
    progress: object | None = None,
) -> list[EvalResult]:
    """Every chunking x retrieval combination against the labelled query set."""
    docs = documents if documents is not None else build_full_corpus()
    qs = queries if queries is not None else build_queries()
    embedder = get_embedder()
    reranker: Reranker | None = None

    results: list[EvalResult] = []
    total = len(chunkers) * len(strategies)
    done = 0

    for chunker in chunkers:
        index = build_index(docs, chunker=chunker, embedder=embedder)
        for strategy in strategies:
            if strategy == "dense":
                retriever, rr = index.dense(), None
            elif strategy == "bm25":
                retriever, rr = index.bm25(), None
            elif strategy == "hybrid":
                retriever, rr = index.hybrid(candidate_k=candidate_k), None
            elif strategy == "hybrid+rerank":
                if reranker is None:
                    reranker = Reranker()
                retriever, rr = index.hybrid(candidate_k=candidate_k), reranker
            else:
                raise ValueError(f"Unknown strategy {strategy!r}")

            results.append(
                evaluate(
                    retriever,
                    qs,
                    k=k,
                    config_name=f"{chunker} + {strategy}",
                    rerank=rr,
                    candidate_k=candidate_k,
                )
            )
            done += 1
            if progress:
                progress(done, total)  # type: ignore[operator]

    results.sort(key=lambda r: -r.ndcg_at_k)
    return results


def write_ablation_markdown(results: list[EvalResult], path: Path, k: int) -> None:
    """Render the ablation table, findings first."""
    from datetime import date

    best = results[0]
    worst = min(results, key=lambda r: r.ndcg_at_k)

    lines = [
        "# Retrieval ablation",
        "",
        f"Generated {date.today().isoformat()} · k={k} · "
        f"{best.n_queries} labelled queries · no API key required.",
        "",
        "Retrieval quality is a ranking question, so it is measured against labelled "
        "relevance rather than an LLM judge. Every number here is reproducible offline "
        "with `rag ablate`.",
        "",
        "| Configuration | recall@k | precision@k | MRR | nDCG@k | leaked | superseded | ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.config}` | {r.recall_at_k:.3f} | {r.precision_at_k:.3f} | "
            f"{r.mrr:.3f} | {r.ndcg_at_k:.3f} | {r.leaked_chunks} | "
            f"{r.superseded_hits} | {r.mean_latency_ms:.0f} |"
        )

    lines += [
        "",
        "## By query type — recall@k",
        "",
        "Aggregate scores hide the interesting behaviour. These columns are why "
        "hybrid retrieval exists.",
        "",
    ]
    kinds = sorted({kind for r in results for kind in r.by_kind})
    lines.append("| Configuration | " + " | ".join(kinds) + " |")
    lines.append("|---" * (len(kinds) + 1) + "|")
    for r in results:
        cells = " | ".join(
            f"{r.by_kind[kind]:.3f}" if kind in r.by_kind else "—" for kind in kinds
        )
        lines.append(f"| `{r.config}` | {cells} |")

    lines += [
        "",
        "## Findings",
        "",
        f"**Best configuration: `{best.config}`** — nDCG@{k} {best.ndcg_at_k:.3f}, "
        f"recall@{k} {best.recall_at_k:.3f}, against `{worst.config}` at "
        f"nDCG {worst.ndcg_at_k:.3f}.",
        "",
        "Read the per-kind table rather than the aggregate: the whole argument for a "
        "second index is that dense and lexical retrieval fail on *different* queries. "
        "If one strategy dominated every column, the other leg would be dead weight.",
        "",
        "**Access control.** The `leaked` column counts chunks returned to a caller "
        "whose roles do not permit them. It is not a quality metric — any value above "
        "zero is a defect, because filtering happens inside the retrieval query rather "
        "than after generation.",
        "",
        "**Versioning.** The `superseded` column counts retrieved chunks from "
        "superseded document revisions. Answering from a withdrawn policy is a "
        "correctness bug, not a staleness annoyance.",
        "",
        "## Method",
        "",
        f"- {best.n_queries} labelled queries across five kinds: identifier, "
        "paraphrase, acronym-collision, versioned, and mixed.",
        "- Relevance is labelled per document; multiple chunks of the same document "
        "count once, so precision is not inflated by chunk granularity.",
        "- Queries whose correct answer is *nothing* (restricted content, wrong role) "
        "are excluded from recall and scored by leakage instead — averaging a "
        "fabricated 1.0 into recall would mask real misses.",
        "- Reranked configurations retrieve `candidate_k` and narrow to `k`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
