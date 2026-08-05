"""enterprise-rag-platform — accurate, cited answers over document collections.

The retrieval half runs entirely offline with local embeddings:

    from rag_platform.pipeline import build_index, run_ablation
    from rag_platform.corpus import build_corpus, build_queries

    index = build_index(build_corpus(), chunker="recursive")
    hits = index.hybrid().retrieve("RMA-4471", k=5, roles={"engineering"})

CLI: `rag ingest | query | ablate`
"""

__version__ = "0.1.0"
