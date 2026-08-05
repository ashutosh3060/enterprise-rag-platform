"""Embeddings — local by default.

Runs `sentence-transformers` on CPU, so the entire retrieval half of this project
works with no API key and no network at query time. That is not a compromise:
local embedding models are what a large share of production RAG systems actually
use, because embedding is called on every chunk at ingest and every query at
serve time, and an API round-trip per query is a latency and cost line nobody
wants.

**Platform constraint.** This machine is an Intel Mac, where PyTorch stopped
publishing x86 wheels after 2.2.2. Current `transformers` calls torch APIs that
do not exist there, and the failure mode is a confusing
`NameError: name 'torch' is not defined` rather than a version conflict. The pins
in `pyproject.toml` are load-bearing on x86 macOS.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["DEFAULT_MODEL", "Embedder", "get_embedder"]

# 384 dimensions, ~80MB, fast on CPU. Big enough to be useful, small enough that
# ingesting a few hundred documents on a laptop is not an overnight job.
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Wraps a sentence-transformers model with normalised output.

    Vectors are L2-normalised at encode time, which makes cosine similarity a
    plain dot product everywhere downstream — one less place to get the metric
    wrong, and it matches what the vector store expects.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: object | None = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers is not installed. On x86 macOS the pinned "
                    'versions are required: pip install "torch==2.2.2" '
                    '"transformers==4.40.2" "sentence-transformers==2.7.0" "numpy<2"'
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())  # type: ignore[attr-defined]

    def encode(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._load().encode(  # type: ignore[attr-defined]
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=4)
def get_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    """Cached per model name — loading the weights takes seconds."""
    return Embedder(model_name)
