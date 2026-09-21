"""geocpt -- continued pre-training, one stage per command.

    corpus    gather, filter, dedup, mix -> train/val text
    pack      tokenise and pack into fixed-length blocks
    train     the CPT loop (full fine-tune by default)
    eval      domain vs general perplexity: did it learn, what did it forget?
    probe     cloze probes: does it actually know more geoscience?
    transfer  the payoff: SFT on top of the CPT'd model, vs SFT alone
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import paths
from .config import Config

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
console = Console()
ConfigOpt = typer.Option(None, "--config", "-c", help="Path to a YAML config.")


def _cfg(path: Optional[str]) -> Config:
    cfg = Config.load(path)
    paths.ensure()
    return cfg


@app.command()
def doctor() -> None:
    """Check environment, disk, the SFT project next door, and stage data."""
    import platform

    t = Table(title="environment")
    t.add_column("check"); t.add_column("value")
    t.add_row("platform", f"{platform.machine()} / {platform.platform()}")
    try:
        import mlx.core as mx
        gb = mx.device_info().get("max_recommended_working_set_size", 0) / 1e9
        t.add_row("mlx", f"{mx.__version__} ({mx.default_device()}, {gb:.1f} GB working set)")
    except Exception as e:
        t.add_row("mlx", f"[red]{e}[/red]")
    for mod in ("mlx_lm", "transformers", "datasketch", "geosft"):
        try:
            m = __import__(mod)
            t.add_row(mod, getattr(m, "__version__", "ok"))
        except Exception as e:
            t.add_row(mod, f"[red]{e}[/red]")

    sft_ok = (paths.SFT_PROJECT / "data" / "processed" / "train.meta.jsonl").exists()
    t.add_row("geo-sft task data",
              "present" if sft_ok else "[yellow]missing — run `geosft build` next door[/yellow]")
    total, _, free = shutil.disk_usage(paths.ROOT)
    t.add_row("disk free", f"{free/1e9:.1f} GB")
    if free < 80e9:
        t.add_row("[yellow]warning[/yellow]",
                  "[yellow]CPT checkpoints are FULL models (~3.4 GB each)[/yellow]")
    for stage in ("tapt", "dapt", "dapt_tapt"):
        c = paths.CORPUS / stage / "train.jsonl"
        pk = paths.PACKED / stage / "train.npy"
        t.add_row(f"stage {stage}",
                  ("corpus " + ("✓" if c.exists() else "·")
                   + "  packed " + ("✓" if pk.exists() else "·")))
    console.print(t)


@app.command()
def corpus(config: Optional[str] = ConfigOpt,
           force: bool = typer.Option(False, help="Rebuild ignoring cache.")) -> None:
    """Gather, quality-filter, deduplicate and mix the raw text corpus."""
    cfg = _cfg(config)
    from .corpus import build
    build.run(cfg.corpus, force=force)


@app.command()
def pack(config: Optional[str] = ConfigOpt) -> None:
    """Tokenise and pack documents into fixed-length training blocks."""
    cfg = _cfg(config)
    from mlx_lm.utils import load as load_model

    from .corpus.build import load_split
    from .pack import pack_documents, pack_stats_table, save

    _, tok = load_model(cfg.train.base_model)
    out_dir = paths.PACKED / cfg.corpus.stage_dir
    for split in ("train", "val", "val_general"):
        try:
            texts = load_split(cfg.corpus.stage_dir, split)
        except FileNotFoundError:
            console.print(f"[dim]no {split} split — skipping[/dim]")
            continue
        if not texts:
            continue
        arr, stats = pack_documents(texts, tok, cfg.pack.block_size,
                                    cfg.pack.add_eos_between_docs,
                                    cfg.pack.no_cross_document)
        console.print(pack_stats_table(stats))
        save(arr, stats, out_dir, split)


@app.command()
def train(config: Optional[str] = ConfigOpt,
          name: Optional[str] = typer.Option(None, help="Run name.")) -> None:
    """Run continued pre-training."""
    cfg = _cfg(config)
    from .train.cpt import run
    s = run(cfg, run_name=name)
    console.print(f"[green]done[/green] -> {s['checkpoint']}")


@app.command(name="eval")
def eval_cmd(config: Optional[str] = ConfigOpt,
             checkpoint: str = typer.Option(..., help="CPT checkpoint directory.")) -> None:
    """Domain vs general perplexity: did it learn, and what did it forget?"""
    cfg = _cfg(config)
    from .eval.perplexity import compare, save
    rep = compare(cfg.train.base_model, checkpoint, cfg.corpus.stage_dir,
                  cfg.train.batch_size, cfg.eval.max_blocks)
    save(rep, Path(checkpoint).parent / "ppl_report.json")


@app.command()
def probe(config: Optional[str] = ConfigOpt,
          checkpoint: str = typer.Option(..., help="CPT checkpoint directory.")) -> None:
    """Cloze probes: does the model recover geoscience terms better?"""
    cfg = _cfg(config)
    from .eval.probe import compare
    rep = compare(cfg.train.base_model, checkpoint, cfg.corpus.stage_dir, cfg.eval.probe_n)
    (Path(checkpoint).parent / "probe_report.json").write_text(json.dumps(rep, indent=2))


@app.command()
def transfer(config: Optional[str] = ConfigOpt,
             checkpoint: str = typer.Option(..., help="CPT checkpoint to SFT on top of."),
             name: Optional[str] = typer.Option(None, help="SFT run name."),
             sft_config: Optional[str] = typer.Option(None, help="geo-sft config YAML.")) -> None:
    """The payoff experiment: does CPT then SFT beat SFT alone?

    Quantises the CPT'd model to 4-bit MLX, then runs the SFT lab's own
    trainer and evaluator on it — same data, same hyperparameters, same
    metrics. The only variable is the starting model, so the comparison
    against geo-sft's published macro F1 is apples to apples.
    """
    cfg = _cfg(config)
    from geosft.config import Config as SftConfig
    from geosft.eval.evaluate import run as sft_eval
    from geosft.train.mlx_train import quantize_for_mlx
    from geosft.train.mlx_train import run as sft_train

    scfg = SftConfig.load(sft_config or str(paths.SFT_PROJECT / "configs" / "default.yaml"))
    q = quantize_for_mlx(checkpoint, paths.MODELS / f"{Path(checkpoint).parent.name}-4bit")
    console.print(f"[bold]SFT on top of CPT model[/bold]: {q}")
    summary = sft_train(scfg, model_path=str(q),
                        run_name=name or f"sft-after-{Path(checkpoint).parent.name}")
    sft_eval(scfg, adapter_path=summary["adapter_path"], model_path=str(q))
    console.print("[dim]Compare macro F1 against geo-sft's SFT-only baseline "
                  "(runs/geo-sft-v1/eval_report.json).[/dim]")


@app.command()
def runs() -> None:
    """Tabulate CPT runs: learning and forgetting side by side."""
    t = Table(title="cpt runs")
    for c in ("run", "stage", "iters", "domain ppl", "general ppl", "forgetting", "min"):
        t.add_column(c, justify="left" if c in ("run", "stage") else "right")
    found = False
    for d in sorted(paths.RUNS.glob("*")):
        p = d / "summary.json"
        if not p.exists():
            continue
        found = True
        s = json.loads(p.read_text())
        t.add_row(d.name, str(s.get("stage", "-")), str(s.get("iters_done", "-")),
                  f"{s.get('domain_ppl_before', 0):.2f}→{s.get('domain_ppl_after', 0):.2f}",
                  f"{s.get('general_ppl_before', 0):.2f}→{s.get('general_ppl_after', 0):.2f}",
                  f"{s.get('forgetting_pct', 0):+.1f}%",
                  f"{s.get('wall_minutes', 0):.1f}")
    console.print(t if found else "[dim]no runs yet[/dim]")


@app.command(name="all")
def run_all(config: Optional[str] = ConfigOpt) -> None:
    """corpus -> pack -> train -> eval -> probe."""
    cfg = _cfg(config)
    from .corpus import build as corpus_build
    from .eval.perplexity import compare, save
    from .eval.probe import compare as probe_compare
    from .train.cpt import run as train_run

    console.rule("[bold]1/5 corpus"); corpus_build.run(cfg.corpus)
    console.rule("[bold]2/5 pack"); pack(config)
    console.rule("[bold]3/5 train"); s = train_run(cfg)
    console.rule("[bold]4/5 perplexity")
    rep = compare(cfg.train.base_model, s["checkpoint"], cfg.corpus.stage_dir,
                  cfg.train.batch_size, cfg.eval.max_blocks)
    save(rep, Path(s["checkpoint"]).parent / "ppl_report.json")
    console.rule("[bold]5/5 probes")
    probe_compare(cfg.train.base_model, s["checkpoint"], cfg.corpus.stage_dir, cfg.eval.probe_n)
    console.print("\n[bold green]done[/bold green]  next: `geocpt transfer "
                  f"--checkpoint {s['checkpoint']}`")


if __name__ == "__main__":
    app()
