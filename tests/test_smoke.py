"""Tests — all offline. Embeddings run locally, so no API key is needed."""

from __future__ import annotations

import numpy as np
import pytest

from rag_platform.chunking import FixedChunker, RecursiveChunker, get_chunker
from rag_platform.corpus import build_corpus, build_full_corpus, build_queries
from rag_platform.evaluation import ndcg_at_k, precision_at_k, recall_at_k
from rag_platform.models import Chunk, Document, ScoredChunk
from rag_platform.retrieval import BM25Retriever, rrf_fuse
from rag_platform.store import NumpyStore


def _chunk(cid: str, doc: str, text: str = "x", roles: tuple[str, ...] = ("*",),
           superseded: bool = False) -> Chunk:
    return Chunk(id=cid, doc_id=doc, text=text, ordinal=0,
                 allowed_roles=roles, superseded=superseded)


# --- models -----------------------------------------------------------------

def test_wildcard_role_is_visible_to_everyone() -> None:
    assert _chunk("a", "d").visible_to({"engineering"})
    assert _chunk("a", "d").visible_to(None)


def test_restricted_chunk_hidden_from_wrong_role() -> None:
    c = _chunk("a", "d", roles=("finance",))
    assert not c.visible_to({"engineering"})
    assert c.visible_to({"finance"})


def test_role_payload_roundtrip_does_not_substring_match() -> None:
    """`admin` must not match inside `superadmin` — hence the delimiters."""
    c = _chunk("a", "d", roles=("superadmin",))
    payload = c.to_payload()
    assert payload["roles"] == "|superadmin|"
    assert "|admin|" not in payload["roles"]
    assert Chunk.from_payload("a", "x", payload).allowed_roles == ("superadmin",)


# --- chunking ---------------------------------------------------------------

def test_overlap_must_be_smaller_than_target() -> None:
    with pytest.raises(ValueError, match="must be <"):
        FixedChunker(target_tokens=100, overlap_tokens=100)


def test_chunks_inherit_document_acl_and_version() -> None:
    doc = Document(id="d1", text="a. " * 300, allowed_roles=("finance",),
                   version=3, superseded=True)
    chunks = RecursiveChunker(target_tokens=50).chunk(doc)
    assert len(chunks) > 1
    assert all(c.allowed_roles == ("finance",) for c in chunks)
    assert all(c.version == 3 and c.superseded for c in chunks)


def test_recursive_chunker_terminates_on_unsplittable_text() -> None:
    """A long run with no separator must hard-cut, not recurse forever."""
    chunks = RecursiveChunker(target_tokens=20, overlap_tokens=0).split("x" * 5000)
    assert len(chunks) > 1


def test_unknown_chunker_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="Available:"):
        get_chunker("magic")


# --- store: permissions enforced pre-ranking --------------------------------

def test_store_filters_restricted_chunks_before_ranking() -> None:
    store = NumpyStore()
    chunks = [_chunk("a", "d1", roles=("finance",)), _chunk("b", "d2")]
    store.add(chunks, np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))

    hits = store.search(np.array([1.0, 0.0]), k=5, roles={"engineering"})
    assert [h.id for h in hits] == ["b"], "the finance chunk must never surface"


def test_store_excludes_superseded_by_default() -> None:
    store = NumpyStore()
    store.add([_chunk("old", "d", superseded=True), _chunk("new", "d")],
              np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32))
    assert [h.id for h in store.search(np.array([1.0, 0.0]), k=5)] == ["new"]
    assert len(store.search(np.array([1.0, 0.0]), k=5, include_superseded=True)) == 2


def test_store_rejects_mismatched_chunk_and_vector_counts() -> None:
    with pytest.raises(ValueError, match="must correspond"):
        NumpyStore().add([_chunk("a", "d")], np.zeros((2, 3), dtype=np.float32))


# --- retrieval --------------------------------------------------------------

def test_bm25_matches_identifiers_exactly() -> None:
    """The reason the lexical leg exists: embeddings cannot separate RMA codes."""
    chunks = [
        _chunk("a", "d1", "RMA-4471 covers thermal throttling on X200 units."),
        _chunk("b", "d2", "RMA-4472 covers fan noise on X300 units."),
    ]
    hits = BM25Retriever(chunks).retrieve("RMA-4471", k=2)
    assert hits[0].chunk.doc_id == "d1"


def test_bm25_respects_roles() -> None:
    chunks = [_chunk("a", "d1", "secret incident runbook", roles=("security",))]
    assert BM25Retriever(chunks).retrieve("incident", k=5, roles={"engineering"}) == []


def test_rrf_promotes_documents_found_by_both_retrievers() -> None:
    """The property that makes a second index worth maintaining."""
    both = _chunk("both", "d-both")
    dense_only = _chunk("dense", "d-dense")
    bm25_only = _chunk("bm25", "d-bm25")

    fused = rrf_fuse({
        "dense": [ScoredChunk(dense_only, 0.9), ScoredChunk(both, 0.8)],
        "bm25": [ScoredChunk(bm25_only, 5.0), ScoredChunk(both, 4.0)],
    }, k=3)
    assert fused[0].id == "both"
    assert set(fused[0].source_ranks) == {"dense", "bm25"}


# --- evaluation metrics -----------------------------------------------------

def test_recall_is_nan_when_nothing_is_relevant() -> None:
    """Restricted-content queries are scored by leakage, not by a fabricated 1.0."""
    import math
    assert math.isnan(recall_at_k([], set(), k=5))


def test_duplicate_document_chunks_count_once_for_precision() -> None:
    results = [ScoredChunk(_chunk(f"c{i}", "d1"), 1.0) for i in range(3)]
    assert precision_at_k(results, {"d1"}, k=3) == 1.0


def test_ndcg_rewards_higher_placement() -> None:
    high = [ScoredChunk(_chunk("a", "good"), 1.0), ScoredChunk(_chunk("b", "bad"), 0.9)]
    low = [ScoredChunk(_chunk("b", "bad"), 1.0), ScoredChunk(_chunk("a", "good"), 0.9)]
    assert ndcg_at_k(high, {"good"}, k=2) > ndcg_at_k(low, {"good"}, k=2)


# --- corpus -----------------------------------------------------------------

def test_corpus_has_a_superseded_pair() -> None:
    docs = {d.id: d for d in build_corpus()}
    assert docs["pol-leave-v1"].superseded
    assert not docs["pol-leave-v2"].superseded


def test_distractors_make_the_corpus_discriminative() -> None:
    """12 documents at k=5 gives recall 1.0 for everything — a useless benchmark."""
    assert len(build_corpus()) < 20
    assert len(build_full_corpus()) > 150


def test_distractor_ids_never_collide_with_gold_ids() -> None:
    gold = {d.id for d in build_corpus()}
    full = build_full_corpus()
    assert len(full) == len({d.id for d in full}), "duplicate document ids"
    assert gold <= {d.id for d in full}


def test_some_queries_expect_no_results() -> None:
    """Restricted queries where the only correct answer is nothing."""
    empties = [q for q in build_queries() if not q.relevant_doc_ids]
    assert empties, "need negative cases to measure leakage"
    assert all(q.roles for q in empties)
