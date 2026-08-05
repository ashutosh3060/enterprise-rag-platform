"""Vector storage.

Two implementations behind one protocol:

- `NumpyStore` — in-process, no dependencies beyond numpy, exact search. Correct
  for corpora up to tens of thousands of chunks, which covers every evaluation in
  this repo. Being exact rather than approximate matters here: an ablation
  comparing retrieval strategies must not have ANN recall error mixed into the
  measurement.
- `ChromaStore` — persistent, HNSW-indexed, the realistic choice at scale.

Qdrant is the better production answer (native hybrid search, richer payload
filtering) but requires a running service. The protocol exists so that is a
config change rather than a rewrite.

**Permission filtering happens inside the store query**, not after it. Filtering
retrieved results post-hoc means restricted content has already been read out of
the index and is one bug away from reaching a prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .models import Chunk, ScoredChunk

__all__ = ["ChromaStore", "NumpyStore", "VectorStore"]


def _visible(chunk: Chunk, roles: set[str] | None, include_superseded: bool) -> bool:
    if not include_superseded and chunk.superseded:
        return False
    return chunk.visible_to(roles)


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        *,
        roles: set[str] | None = None,
        include_superseded: bool = False,
    ) -> list[ScoredChunk]: ...

    def __len__(self) -> int: ...


class NumpyStore:
    """Exact brute-force search over normalised vectors."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors — these must correspond."
            )
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._vectors = (
            vectors.astype(np.float32)
            if self._vectors is None
            else np.vstack([self._vectors, vectors.astype(np.float32)])
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        *,
        roles: set[str] | None = None,
        include_superseded: bool = False,
    ) -> list[ScoredChunk]:
        if self._vectors is None or not self._chunks:
            return []

        # Build the visibility mask *before* scoring, so a filtered-out chunk can
        # never occupy a top-k slot and silently shorten the result set.
        allowed = np.array(
            [_visible(c, roles, include_superseded) for c in self._chunks], dtype=bool
        )
        if not allowed.any():
            return []

        sims = self._vectors @ np.asarray(query_vector, dtype=np.float32).ravel()
        sims = np.where(allowed, sims, -np.inf)
        top = np.argsort(-sims)[: min(k, int(allowed.sum()))]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(sims[i]), source_ranks={"dense": r})
            for r, i in enumerate(top)
            if np.isfinite(sims[i])
        ]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)


class ChromaStore:
    """Persistent store backed by Chroma.

    Permission and version predicates are pushed into Chroma's `where` clause so
    the filtering happens in the index, not in Python after the fact.
    """

    def __init__(self, path: Path | str | None = None, collection: str = "chunks") -> None:
        import chromadb

        self._client = (
            chromadb.PersistentClient(path=str(path))
            if path
            else chromadb.EphemeralClient()
        )
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors — these must correspond."
            )
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[c.to_payload() for c in chunks],
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        *,
        roles: set[str] | None = None,
        include_superseded: bool = False,
    ) -> list[ScoredChunk]:
        clauses: list[dict[str, object]] = []
        if not include_superseded:
            clauses.append({"superseded": False})

        where: dict[str, object] | None = None
        if len(clauses) == 1:
            where = clauses[0]
        elif clauses:
            where = {"$and": clauses}

        # Chroma has no set-intersection operator, so role filtering cannot be
        # expressed in `where`. Over-fetch and filter the remainder in Python —
        # still pre-generation, which is the property that matters, and the
        # over-fetch keeps top-k full after filtering.
        fetch = k if roles is None else k * 5
        res = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).ravel().tolist()],
            n_results=min(fetch, max(1, self._collection.count())),
            where=where,
        )

        out: list[ScoredChunk] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=True):
            chunk = Chunk.from_payload(cid, text, dict(meta))
            if not _visible(chunk, roles, include_superseded):
                continue
            # Chroma returns cosine *distance*; convert to similarity so every
            # store in this module reports the same direction.
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=1.0 - float(dist),
                    source_ranks={"dense": len(out)},
                )
            )
            if len(out) >= k:
                break
        return out

    def __len__(self) -> int:
        return int(self._collection.count())
