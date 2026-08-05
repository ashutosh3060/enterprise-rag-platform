"""Core domain types.

Two decisions are encoded here and everything else follows from them:

1. **Documents are versioned, not overwritten.** Enterprise policies supersede each
   other, and answering from a superseded revision is a correctness bug rather
   than a staleness annoyance. `version` and `superseded` are retrieval filters,
   not metadata.

2. **Access control travels with the chunk.** `allowed_roles` is on the chunk
   because that is what retrieval filters on. Enforcing permissions after
   generation is not a security boundary — the restricted text has already
   entered the model's context and can leak through paraphrase.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any

__all__ = ["Chunk", "Document", "ScoredChunk"]


@dataclass
class Document:
    """A source document, before chunking."""

    id: str
    text: str
    title: str = ""
    source: str = ""
    doc_type: str = "text"
    version: int = 1
    superseded: bool = False
    effective_date: date | None = None
    allowed_roles: tuple[str, ...] = ("*",)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    """A retrievable unit.

    Carries its parent document's provenance and ACL so that retrieval can filter
    and cite without a second lookup.
    """

    id: str
    doc_id: str
    text: str
    ordinal: int
    title: str = ""
    source: str = ""
    version: int = 1
    superseded: bool = False
    allowed_roles: tuple[str, ...] = ("*",)
    metadata: dict[str, Any] = field(default_factory=dict)

    def visible_to(self, roles: set[str] | None) -> bool:
        """Whether a caller holding `roles` may see this chunk.

        `None` means no access control is being applied — used by ingestion and
        evaluation, never by a user-facing query path.
        """
        if roles is None:
            return True
        if "*" in self.allowed_roles:
            return True
        return bool(roles & set(self.allowed_roles))

    def to_payload(self) -> dict[str, Any]:
        """Flat metadata for a vector store payload.

        Chroma (and most stores) accept only scalars in metadata, so the role
        tuple is serialised to a delimited string. The delimiters on both ends
        prevent `admin` from matching `superadmin` in a substring filter.
        """
        return {
            "doc_id": self.doc_id,
            "ordinal": self.ordinal,
            "title": self.title,
            "source": self.source,
            "version": self.version,
            "superseded": self.superseded,
            "roles": "|" + "|".join(self.allowed_roles) + "|",
            **{k: v for k, v in self.metadata.items() if isinstance(v, str | int | float | bool)},
        }

    @classmethod
    def from_payload(cls, chunk_id: str, text: str, payload: dict[str, Any]) -> Chunk:
        roles = tuple(r for r in str(payload.get("roles", "|*|")).split("|") if r)
        return cls(
            id=chunk_id,
            doc_id=str(payload.get("doc_id", "")),
            text=text,
            ordinal=int(payload.get("ordinal", 0)),
            title=str(payload.get("title", "")),
            source=str(payload.get("source", "")),
            version=int(payload.get("version", 1)),
            superseded=bool(payload.get("superseded", False)),
            allowed_roles=roles or ("*",),
        )


@dataclass
class ScoredChunk:
    """A chunk with a retrieval score and its provenance."""

    chunk: Chunk
    score: float
    # Which retriever produced it, and its rank there. Kept because the whole
    # point of hybrid retrieval is knowing *why* something surfaced — a result
    # found only by BM25 tells you something different from one both agree on.
    source_ranks: dict[str, int] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.text
