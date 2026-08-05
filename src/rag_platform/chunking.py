"""Chunking strategies behind one interface.

Chunking is the highest-leverage and least-examined decision in a RAG pipeline.
It is also the one most often made once, by accident, and never revisited — which
is why all three strategies live behind the same interface and the evaluation
harness can ablate across them.

The trade-off in one line: smaller chunks retrieve precisely but lose the context
that makes an answer correct; larger chunks carry context but dilute the
embedding and drag in irrelevant text.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .models import Chunk, Document

__all__ = [
    "CHUNKERS",
    "Chunker",
    "FixedChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "get_chunker",
]

# Split points in descending order of how much structure they preserve.
_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def _approx_tokens(text: str) -> int:
    """~4 chars/token. Crude, but chunk sizing does not need better."""
    return max(1, len(text) // 4)


class Chunker(ABC):
    """Splits a document into retrievable chunks."""

    name: str = "base"

    def __init__(self, target_tokens: int = 200, overlap_tokens: int = 40) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be < target_tokens "
                f"({target_tokens}), otherwise chunking cannot make progress."
            )
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    @abstractmethod
    def split(self, text: str) -> list[str]: ...

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a document, carrying provenance and ACL onto every chunk."""
        pieces = [p.strip() for p in self.split(doc.text) if p.strip()]
        return [
            Chunk(
                id=f"{doc.id}::{self.name}::{i}",
                doc_id=doc.id,
                text=piece,
                ordinal=i,
                title=doc.title,
                source=doc.source,
                version=doc.version,
                superseded=doc.superseded,
                allowed_roles=doc.allowed_roles,
                metadata={"chunker": self.name, **doc.metadata},
            )
            for i, piece in enumerate(pieces)
        ]


class FixedChunker(Chunker):
    """Fixed-width windows with overlap, ignoring structure.

    The baseline. Fast, predictable, and blind — it will cut mid-sentence and
    mid-table. Included because it is what most pipelines actually ship, and an
    ablation needs an honest baseline rather than a strawman.
    """

    name = "fixed"

    def split(self, text: str) -> list[str]:
        window = self.target_tokens * 4
        stride = max(1, window - self.overlap_tokens * 4)
        return [text[i : i + window] for i in range(0, len(text), stride)]


class RecursiveChunker(Chunker):
    """Split on the most structural separator that fits, descending.

    Tries paragraph breaks before line breaks before sentences before words, so a
    chunk boundary lands where the document already had one. This is the sensible
    default for prose.
    """

    name = "recursive"

    def split(self, text: str) -> list[str]:
        # Overlap is applied once, here — not inside `_split`. Applying it at
        # every recursion level compounds it, so a deeply-split document ends up
        # with chunks that are mostly duplicated preceding text.
        return self._apply_overlap(self._split(text, _SEPARATORS))

    def _split(self, text: str, seps: list[str]) -> list[str]:
        if _approx_tokens(text) <= self.target_tokens:
            return [text]
        if not seps:
            # No separator left: hard-cut. Rare, but a 50k-token line with no
            # whitespace must not recurse forever.
            window = self.target_tokens * 4
            return [text[i : i + window] for i in range(0, len(text), window)]

        sep, rest = seps[0], seps[1:]
        parts = text.split(sep)
        if len(parts) == 1:
            return self._split(text, rest)

        out: list[str] = []
        buf = ""
        for part in parts:
            candidate = f"{buf}{sep}{part}" if buf else part
            if _approx_tokens(candidate) <= self.target_tokens:
                buf = candidate
                continue
            if buf:
                out.append(buf)
            # The part alone may still be too large — recurse with finer separators.
            too_big = _approx_tokens(part) > self.target_tokens
            out.extend(self._split(part, rest) if too_big else [part])
            buf = ""
        if buf:
            out.append(buf)
        return out

    def _apply_overlap(self, parts: list[str]) -> list[str]:
        """Prefix each chunk with the tail of its predecessor.

        Overlap exists so a fact split across a boundary is retrievable from
        either side. It costs storage and some duplicate retrieval, which the
        reranker then has to collapse.
        """
        if self.overlap_tokens <= 0 or len(parts) < 2:
            return parts
        chars = self.overlap_tokens * 4
        out = [parts[0]]
        # `parts[:-1]` and `parts[1:]` are equal-length; zipping `parts` against
        # `parts[1:]` under strict=True always raises.
        for prev, cur in zip(parts[:-1], parts[1:], strict=True):
            out.append((prev[-chars:] + " " + cur).strip())
        return out


class SemanticChunker(Chunker):
    """Group adjacent sentences while they stay on-topic.

    Embeds each sentence, then starts a new chunk where consecutive-sentence
    similarity drops below a threshold — the assumption being that a topic shift
    is a better boundary than a character count.

    Costs an extra embedding pass at ingest. That is a batch cost paid once,
    against retrieval quality paid on every query, which is why it is usually
    worth it. Requires an embedder; falls back to recursive without one.
    """

    name = "semantic"

    def __init__(
        self,
        target_tokens: int = 200,
        overlap_tokens: int = 0,
        embedder: object | None = None,
        breakpoint_percentile: float = 25.0,
    ) -> None:
        super().__init__(target_tokens, overlap_tokens)
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile

    def split(self, text: str) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) < 3 or self.embedder is None:
            return RecursiveChunker(self.target_tokens, self.overlap_tokens).split(text)

        import numpy as np

        vectors = self.embedder.encode(sentences)  # type: ignore[attr-defined]
        sims = np.array(
            [float(np.dot(vectors[i], vectors[i + 1])) for i in range(len(sentences) - 1)]
        )
        # A boundary is a similarity in the bottom quartile — i.e. the biggest
        # topic shifts, relative to this document rather than an absolute cutoff.
        threshold = float(np.percentile(sims, self.breakpoint_percentile))

        chunks: list[str] = []
        buf = [sentences[0]]
        for i, sentence in enumerate(sentences[1:]):
            too_long = _approx_tokens(" ".join([*buf, sentence])) > self.target_tokens
            if sims[i] < threshold or too_long:
                chunks.append(" ".join(buf))
                buf = [sentence]
            else:
                buf.append(sentence)
        if buf:
            chunks.append(" ".join(buf))
        return chunks


CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def get_chunker(name: str, **kwargs: object) -> Chunker:
    try:
        cls = CHUNKERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown chunker {name!r}. Available: {', '.join(sorted(CHUNKERS))}"
        ) from None
    return cls(**kwargs)  # type: ignore[arg-type]
