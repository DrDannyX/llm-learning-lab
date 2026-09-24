"""georag -- one CLI for the whole lab.

  georag doctor                 is everything running?
  georag ingest                 corpus -> graph + embeddings -> Neo4j
  georag ask "question"         three answers, side by side
  georag inspect "question"     what each system RETRIEVED (the context, not the answer)
  georag bench build|run|report the quantitative comparison
  georag mcp                    serve the systems over MCP
  georag agent                  a Strands agent that uses that MCP server
  georag web                    the same comparison in a browser
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config as config_mod
from . import db, paths

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
bench_app = typer.Typer(no_args_is_help=True, help="Build, run and report the benchmark.")
app.add_typer(bench_app, name="bench")
console = Console()

CfgOpt = typer.Option(None, "--config", "-c", help="config YAML (default configs/default.yaml)")
SYSTEM_CHOICES = ["rag", "kg", "graphrag"]


def _cfg(path: Path | None):
    paths.ensure()
    return config_mod.load(path)


@app.command()
def doctor(config: Path = CfgOpt) -> None:
    """Check Neo4j, LM Studio, the corpus and the loaded graph."""
    import httpx

    cfg = _cfg(config)
    ok = True

    def line(good: bool, what: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        console.print(f"  {'[green]✓' if good else '[red]✗'}[/] {what} [dim]{detail}[/]")

    console.print("[bold]corpus")
    line(paths.PASSAGES.exists(), "geo-sft passages", str(paths.PASSAGES))
    line(paths.VOCAB.exists(), "geo-sft vocab", str(paths.VOCAB))

    console.print("[bold]LM Studio")
    try:
        base = cfg.llm.base_url.rsplit("/v1", 1)[0]
        models = httpx.get(f"{base}/api/v0/models", timeout=5).json()["data"]
        byid = {m["id"]: m for m in models}
        for mid in {cfg.llm.chat_model, cfg.llm.judge_model, cfg.llm.embed_model}:
            state = byid.get(mid, {}).get("state", "not available")
            line(mid in byid, f"model {mid}", state)
            if mid in byid and state != "loaded":
                # JIT loading with auto-evict swaps the chat and embedding models
                # on EVERY question: no error, just a benchmark that takes hours
                console.print(f"    [yellow]![/] not loaded: pin it so JIT does not evict the "
                              f"other model -- [dim]lms load {mid}[/]")
        ctx = byid.get(cfg.llm.chat_model, {}).get("loaded_context_length")
        if ctx and ctx < 16384:
            # a warning, not a failure: ask/bench fit in 8k, a multi-turn agent does not
            console.print(f"  [yellow]![/] chat context {ctx} tokens [dim]fine for ask/bench; "
                          "reload with >=16k for `georag agent`: "
                          f"lms load {cfg.llm.chat_model} --context-length 32768[/]")
        elif ctx:
            line(True, f"chat context {ctx} tokens")
        t = time.perf_counter()
        from .llm import Embedder
        dim = len(Embedder(cfg.llm).query("test"))
        line(dim == cfg.llm.embed_dim, f"embeddings {dim}-d", f"{time.perf_counter() - t:.1f}s")
    except Exception as e:  # noqa: BLE001
        line(False, "LM Studio reachable", f"{cfg.llm.base_url}: {e}")

    console.print("[bold]Neo4j")
    try:
        line(True, "connected", db.ping(cfg.neo4j))
        n = db.read(cfg.neo4j, "MATCH (p:Passage) RETURN count(p) AS n")[0]["n"]
        line(n > 0, f"{n:,} passages loaded", "" if n else "run `make ingest`")
        idx = db.read(cfg.neo4j, "SHOW INDEXES YIELD name, state WHERE name = $n RETURN state",
                      n=db.VECTOR_INDEX)
        line(bool(idx) and idx[0]["state"] == "ONLINE", "vector index",
             idx[0]["state"] if idx else "missing")
    except Exception as e:  # noqa: BLE001
        line(False, "Neo4j reachable", f"{cfg.neo4j.uri}: {e} -- run `make up`")

    raise typer.Exit(0 if ok else 1)


@app.command()
def ingest(config: Path = CfgOpt,
           reset: bool = typer.Option(True, help="wipe the database first")) -> None:
    """Corpus -> knowledge graph + passage embeddings -> Neo4j."""
    from .ingest import extract, load

    cfg = _cfg(config)
    t = time.perf_counter()
    g = extract.build(paths.PASSAGES, paths.VOCAB, cfg.ingest.limit_units, cfg.bench.seed)
    g.save(paths.GRAPH_JSON)
    console.print(f"[bold]extracted[/] -> {paths.GRAPH_JSON}")
    console.print_json(json.dumps(g.stats))
    if reset:
        load.reset(cfg)
    load.load_graph(cfg, g)
    vecs = load.embed_passages(cfg, g)
    load.load_embeddings(cfg, g, vecs)
    console.print("[bold]loaded into Neo4j[/]")
    console.print_json(json.dumps(load.counts(cfg)))
    console.print(f"done in {time.perf_counter() - t:.0f}s. Browse it at http://localhost:7474")


@app.command()
def ask(question: str,
        system: list[str] = typer.Option(SYSTEM_CHOICES, "--system", "-s"),
        show_context: bool = typer.Option(False, "--context", help="also print retrieved context"),
        config: Path = CfgOpt) -> None:
    """Ask all three systems the same question and show the answers side by side."""
    from .retrievers.pipeline import LABELS, answer

    cfg = _cfg(config)
    panels = []
    for s in system:
        with console.status(f"{LABELS[s]} ..."):
            a = answer(cfg, s, question)
        r = a.retrieval
        foot = f"{a.seconds:.1f}s total · {r.seconds:.1f}s retrieval · {len(r.context):,} chars context"
        if s == "kg" and r.debug.get("attempts"):
            foot += f"\n\n[dim]{r.debug['attempts'][-1]['cypher']}[/]"
        panels.append(Panel(f"{a.answer}\n\n[dim]{foot}[/]", title=LABELS[s], expand=True))
        if show_context:
            console.print(Panel(r.context, title=f"{LABELS[s]} context", border_style="dim"))
    console.print(Panel(question, title="question", border_style="cyan"))
    grid = Table.grid(expand=True, padding=(0, 1))
    for _ in panels:
        grid.add_column(ratio=1)
    grid.add_row(*panels)
    console.print(grid)


@app.command()
def inspect(question: str, config: Path = CfgOpt) -> None:
    """Show exactly what each system retrieves for a question -- the context the answer
    model will see, plus linked entities, Cypher attempts and routed passages.

    The single most useful screen in this lab: most wrong answers are decided here,
    before any answer is generated."""
    from .retrievers.pipeline import LABELS, SYSTEMS

    cfg = _cfg(config)
    for name, retrieve in SYSTEMS.items():
        with console.status(f"retrieving with {LABELS[name]} ..."):
            r = retrieve(cfg, question)
        console.rule(f"[bold]{LABELS[name]}[/]  {r.seconds:.1f}s")
        if r.debug:
            console.print_json(json.dumps(r.debug, default=str))
        console.print(Panel(r.context or "(empty)", title="context sent to the answer model",
                            border_style="dim"))


@bench_app.command("build")
def bench_build(config: Path = CfgOpt) -> None:
    """Generate the question set from the extracted graph (and the LLM, for descriptive)."""
    from .bench.build import build

    cfg = _cfg(config)
    out = paths.BENCH / f"{cfg.bench.name}.jsonl"
    qs = build(cfg, paths.GRAPH_JSON, out)
    t = Table("category", "n", "example")
    for c in dict.fromkeys(q["category"] for q in qs):
        ex = [q for q in qs if q["category"] == c]
        t.add_row(c, str(len(ex)), ex[0]["question"])
    console.print(t)
    console.print(f"wrote {len(qs)} questions -> {out}")


@bench_app.command("run")
def bench_run(config: Path = CfgOpt,
              system: list[str] = typer.Option(SYSTEM_CHOICES, "--system", "-s"),
              out: Path = typer.Option(None, help="run dir; reuse one to resume"),
              limit: int = typer.Option(0, help="only the first N questions per category")) -> None:
    """Answer every question with every system, score, and write runs/<name>/report.md."""
    from .bench.run import run

    cfg = _cfg(config)
    qpath = paths.BENCH / f"{cfg.bench.name}.jsonl"
    qs = [json.loads(line) for line in qpath.open()]
    if limit:
        kept: dict[str, int] = {}
        subset = []
        for q in qs:
            kept[q["category"]] = kept.get(q["category"], 0) + 1
            if kept[q["category"]] <= limit:
                subset.append(q)
        qs = subset
    out = out or paths.RUNS / f"{cfg.bench.name}-{datetime.now():%Y%m%d-%H%M}"
    run(cfg, qs, out, system)
    console.print((out / "report.md").read_text())


@bench_app.command("report")
def bench_report(run_dir: Path) -> None:
    """Recompute and print the report for an existing run directory."""
    from .bench.run import report

    report(run_dir)
    console.print((run_dir / "report.md").read_text())


@app.command()
def mcp(http: bool = typer.Option(False, help="streamable HTTP on :8765 instead of stdio")) -> None:
    """Serve the three systems as MCP tools."""
    import sys

    from . import mcp_server

    if http:
        sys.argv.append("--http")
    mcp_server.main()


@app.command()
def web(host: str = "127.0.0.1", port: int = 8000) -> None:
    """A local web page: one question, three answers side by side."""
    import uvicorn

    console.print(f"geo-graphrag web UI on [bold]http://{host}:{port}[/]  (ctrl-c to stop)")
    uvicorn.run("georag.web.app:app", host=host, port=port, log_level="warning")


@app.command()
def agent(question: str = typer.Argument(None, help="ask once and exit; omit for a REPL"),
          config: Path = CfgOpt) -> None:
    """A Strands agent (local LLM) that interrogates the three systems over MCP."""
    from .agent import session

    cfg = _cfg(config)
    with session(cfg, str(config) if config else None) as a:
        if question:
            a(question)
            console.print()
            return
        console.print("[bold]geo-graphrag agent[/] -- ask about US geologic units; empty line to quit")
        while q := console.input("\n[cyan]you>[/] ").strip():
            a(q)
            console.print()


if __name__ == "__main__":
    app()
