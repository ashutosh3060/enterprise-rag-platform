"""Command line interface.

    rag ablate                 run the full chunking x retrieval ablation
    rag query "RMA-4471"       query the synthetic corpus
    rag corpus                 show the corpus and labelled queries
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .corpus import build_corpus, build_full_corpus, build_queries
from .pipeline import build_index, run_ablation, write_ablation_markdown

app = typer.Typer(add_completion=False, help="Enterprise RAG Platform")
console = Console(width=125)


@app.command()
def corpus() -> None:
    """Show the synthetic corpus and its labelled queries."""
    docs = build_corpus()
    table = Table(title=f"Corpus — {len(docs)} documents")
    for col in ("id", "type", "v", "superseded", "roles", "title"):
        table.add_column(col)
    for d in docs:
        table.add_row(
            d.id, d.doc_type, str(d.version),
            "[red]yes[/]" if d.superseded else "no",
            ",".join(d.allowed_roles), d.title[:44],
        )
    console.print(table)

    qs = build_queries()
    qt = Table(title=f"Labelled queries — {len(qs)}")
    for col in ("id", "kind", "roles", "query", "relevant"):
        qt.add_column(col)
    for q in qs:
        qt.add_row(
            q.id, q.kind, ",".join(sorted(q.roles)) if q.roles else "—",
            q.text[:42], ", ".join(sorted(q.relevant_doc_ids)) or "[yellow]none[/]",
        )
    console.print(qt)


@app.command()
def query(
    text: Annotated[str, typer.Argument(help="Query text.")],
    k: Annotated[int, typer.Option("-k")] = 5,
    strategy: Annotated[str, typer.Option("--strategy", "-s")] = "hybrid",
    chunker: Annotated[str, typer.Option("--chunker", "-c")] = "recursive",
    roles: Annotated[str | None, typer.Option("--roles", help="Comma-separated.")] = None,
) -> None:
    """Query the synthetic corpus."""
    role_set = {r.strip() for r in roles.split(",")} if roles else None
    index = build_index(build_full_corpus(), chunker=chunker)

    retriever = {
        "dense": index.dense, "bm25": index.bm25, "hybrid": index.hybrid,
    }.get(strategy)
    if retriever is None:
        console.print(f"[red]Unknown strategy[/] {strategy!r}")
        raise typer.Exit(1)

    results = retriever().retrieve(text, k=k, roles=role_set)
    if not results:
        console.print(
            "[yellow]No results.[/] If a role filter is set, this may be correct — "
            "restricted content is withheld rather than summarised."
        )
        return

    table = Table(title=f"{strategy} · k={k} · roles={role_set or 'unrestricted'}")
    for col in ("#", "score", "doc", "v", "found by", "text"):
        table.add_column(col)
    for i, s in enumerate(results, 1):
        table.add_row(
            str(i), f"{s.score:.4f}", s.chunk.doc_id, str(s.chunk.version),
            ",".join(s.source_ranks), s.text[:58].replace("\n", " "),
        )
    console.print(table)


@app.command()
def ablate(
    k: Annotated[int, typer.Option("-k")] = 5,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    skip_rerank: Annotated[bool, typer.Option("--skip-rerank")] = False,
) -> None:
    """Run the chunking x retrieval ablation."""
    strategies = ("dense", "bm25", "hybrid")
    if not skip_rerank:
        strategies = (*strategies, "hybrid+rerank")

    with console.status("running ablation...") as status:
        results = run_ablation(
            k=k, strategies=strategies,
            progress=lambda d, t: status.update(f"running ablation... {d}/{t}"),
        )

    table = Table(title=f"Retrieval ablation (k={k})")
    for col in ("configuration", "recall", "prec", "MRR", "nDCG", "leak", "sup", "ms"):
        table.add_column(col)
    for r in results:
        table.add_row(
            r.config, f"{r.recall_at_k:.3f}", f"{r.precision_at_k:.3f}",
            f"{r.mrr:.3f}", f"{r.ndcg_at_k:.3f}",
            f"[red]{r.leaked_chunks}[/]" if r.leaked_chunks else "0",
            f"[yellow]{r.superseded_hits}[/]" if r.superseded_hits else "0",
            f"{r.mean_latency_ms:.0f}",
        )
    console.print(table)

    kinds = sorted({kind for r in results for kind in r.by_kind})
    kt = Table(title="recall@k by query kind")
    kt.add_column("configuration")
    for kind in kinds:
        kt.add_column(kind)
    for r in results:
        kt.add_row(r.config, *[
            f"{r.by_kind[kind]:.3f}" if kind in r.by_kind else "—" for kind in kinds
        ])
    console.print(kt)

    if out:
        write_ablation_markdown(results, out, k)
        console.print(f"[green]wrote[/] {out}")


if __name__ == "__main__":
    app()
