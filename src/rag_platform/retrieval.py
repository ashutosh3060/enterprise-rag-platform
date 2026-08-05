"""Retrieval: dense, lexical, hybrid, and reranked.

The claim this module exists to test is that **hybrid retrieval beats pure dense
retrieval on enterprise corpora**, and specifically because of the tokens dense
embeddings handle worst: part numbers, error codes, ticket IDs, acronyms. An
embedding maps `RMA-4471` into roughly the same region as every other identifier
it has seen; BM25 matches it exactly.

Fusion is Reciprocal Rank Fusion rather than score normalisation. RRF combines
*ranks*, so it needs no calibration between a cosine similarity and a BM25 score
— two quantities on incomparable scales whose relative magnitudes shift with
corpus size. Weighted score blending requires tuning a coefficient per corpus;
RRF has one constant that works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .embedding import Embedder, get_embedder
from .models import Chunk, ScoredChunk
from .store import VectorStore

__all__ = ["BM25Retriever", "DenseRetriever", "HybridRetriever", "Reranker", "rrf_fuse"]

# The standard RRF constant. Damps the influence of top ranks so a single
# retriever cannot dominate the fused ordering on its own.
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping internal hyphens and digits.

    Keeping `rma-4471` as one token rather than splitting on the hyphen is the
    whole reason the lexical leg earns its place — splitting it would discard
    exactly the signal dense retrieval already lacks.
    """
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class DenseRetriever:
    """Vector similarity over an embedding store."""

    name = "dense"

    def __init__(self, store: VectorStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()

    def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        roles: set[str] | None = None,
        include_superseded: bool = False,
    ) -> list[ScoredChunk]:
        vector = self.embedder.encode(query)[0]
        return self.store.search(
            vector, k=k, roles=roles, include_superseded=include_superseded
        )


class BM25Retriever:
    """Lexical retrieval over the same chunk set."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._chunks]) if chunks else None

    def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        roles: set[str] | None = None,
        include_superseded: bool = False,
    ) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))

        # Same discipline as the vector store: apply visibility before ranking,
        # so a filtered chunk never consumes a top-k slot.
        eligible = [
            i
            for i, c in enumerate(self._chunks)
            if c.visible_to(roles) and (include_superseded or not c.superseded)
        ]
        eligible.sort(key=lambda i: -scores[i])

        # Drop non-matches — but only when there is something positive to compare
        # against. Robertson IDF evaluates to exactly 0 when a term appears in
        # roughly half a very small corpus, so an unconditional `> 0` filter
        # discards genuine single-document matches on tiny indexes. Real corpora
        # never hit this; test fixtures and cold-start indexes do.
        best = max((scores[i] for i in eligible), default=0.0)
        keep = eligible[:k] if best <= 0 else [i for i in eligible[:k] if scores[i] > 0]

        return [
            ScoredChunk(
                chunk=self._chunks[i], score=float(scores[i]), source_ranks={"bm25": rank}
            )
            for rank, i in enumerate(keep)
        ]


def rrf_fuse(
    result_sets: dict[str, list[ScoredChunk]], k: int = 10, rrf_k: int = RRF_K
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion across named retrievers.

    score(d) = sum over retrievers of 1 / (rrf_k + rank(d))

    Rank-based, so no score normalisation is needed between retrievers on
    different scales. A document found by both legs outranks one found by either
    alone, which is the behaviour that makes hybrid worth the second index.
    """
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    ranks: dict[str, dict[str, int]] = {}

    for retriever_name, results in result_sets.items():
        for rank, scored in enumerate(results):
            cid = scored.id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks[cid] = scored.chunk
            ranks.setdefault(cid, {})[retriever_name] = rank

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [
        ScoredChunk(chunk=chunks[cid], score=score, source_ranks=ranks[cid])
        for cid, score in ordered
    ]


@dataclass
class HybridRetriever:
    """Dense + BM25, fused with RRF.

    `candidate_k` over-fetches from each leg before fusion. Fusing only the top-k
    of each would discard exactly the documents that rank modestly in both — which
    are the ones fusion is supposed to promote.
    """

    dense: DenseRetriever
    bm25: BM25Retriever
    candidate_k: int = 30
    name: str = "hybrid"

    def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        roles: set[str] | None = None,
        include_superseded: bool = False,
    ) -> list[ScoredChunk]:
        kwargs = {"roles": roles, "include_superseded": include_superseded}
        return rrf_fuse(
            {
                "dense": self.dense.retrieve(query, k=self.candidate_k, **kwargs),
                "bm25": self.bm25.retrieve(query, k=self.candidate_k, **kwargs),
            },
            k=k,
        )


class Reranker:
    """Cross-encoder reranking of a candidate set.

    Bi-encoder retrieval embeds query and document independently, which is what
    makes it fast enough to search a whole corpus — and also what costs it
    precision, since the two never interact. A cross-encoder reads the pair
    together and is far more accurate, but too slow to run over everything.

    So it runs over the top candidates only: retrieve wide for recall, rerank
    narrow for precision. This is consistently the single largest quality win in
    the pipeline, and the reason `candidate_k` is much larger than `k`.
    """

    name = "rerank"
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: object | None = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int = 10) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._load().predict(  # type: ignore[attr-defined]
            [(query, c.text) for c in candidates], show_progress_bar=False
        )
        order = np.argsort(-np.asarray(scores))[:k]
        return [
            ScoredChunk(
                chunk=candidates[i].chunk,
                score=float(scores[i]),
                # Preserve where it came from, plus its pre-rerank position —
                # the delta is what tells you whether reranking is earning its
                # latency on this corpus.
                source_ranks={**candidates[i].source_ranks, "pre_rerank": int(i)},
            )
            for i in order
        ]
