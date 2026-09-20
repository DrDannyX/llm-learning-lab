"""geosft -- the whole workflow, one stage per command.

Run `geosft all` for the full pipeline, or run the stages one at a time while
you are learning. Every stage writes its outputs to disk and is re-runnable,
so you can iterate on one stage without repeating the ones before it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from . import paths
from .config import Config

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
console = Console()

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to a YAML config file.")


def _cfg(path: Optional[str]) -> Config:
    cfg = Config.load(path)
    paths.ensure()
    return cfg


@app.command()
def doctor() -> None:
    """Check the environment before you burn an hour on a failed run."""
    import platform

    table = Table(title="environment")
    table.add_column("check"); table.add_column("value")

    table.add_row("platform", f"{platform.machine()} / {platform.platform()}")
    try:
        import mlx.core as mx
        dev = mx.default_device()
        table.add_row("mlx", f"{mx.__version__} (device {dev})")
        info = mx.device_info()
        gb = info.get("max_recommended_working_set_size", 0) / 1e9
        table.add_row("mlx working set", f"{gb:.1f} GB")
    except Exception as e:
        table.add_row("mlx", f"[red]unavailable: {e}[/red]")
    try:
        import torch
        table.add_row("torch", f"{torch.__version__} (mps={torch.backends.mps.is_available()})")
    except Exception as e:
        table.add_row("torch", f"[red]{e}[/red]")
    for mod in ("mlx_lm", "transformers", "peft", "trl", "datasets", "tokenizers"):
        try:
            m = __import__(mod)
            table.add_row(mod, getattr(m, "__version__", "?"))
        except Exception as e:
            table.add_row(mod, f"[red]{e}[/red]")

    total, used, free = shutil.disk_usage(paths.ROOT)
    table.add_row("disk free", f"{free/1e9:.1f} GB of {total/1e9:.1f} GB")
    if free < 60e9:
        table.add_row("[yellow]warning[/yellow]",
                      "[yellow]<60GB free; base weights + quantised copies may not fit[/yellow]")
    table.add_row("HF cache", str(paths.HF_HOME))

    for name, p in (("passages", paths.INTERIM / "passages.jsonl"),
                    ("train.jsonl", paths.PROCESSED / "train.jsonl"),
                    ("gold.jsonl", paths.GOLD / "gold.jsonl")):
        table.add_row(name, "present" if p.exists() else "[dim]missing[/dim]")
    console.print(table)


@app.command()
def fetch(config: Optional[str] = ConfigOpt,
          force: bool = typer.Option(False, help="Ignore the cache and refetch.")) -> None:
    """Download the Geolex corpus and the Macrostrat gazetteers."""
    cfg = _cfg(config)
    from .data import fetch as fetch_mod
    from .data import vocab as vocab_mod

    vocab_mod.build(force=force)
    fetch_mod.run(cfg.data.max_units, cfg.data.min_passage_chars,
                  cfg.data.max_passage_chars, force=force)


@app.command()
def build(config: Optional[str] = ConfigOpt) -> None:
    """Label passages, split by unit, and write train/valid/test/gold."""
    cfg = _cfg(config)
    from .data import build as build_mod
    build_mod.run(cfg.data)


@app.command()
def tokenizer(config: Optional[str] = ConfigOpt) -> None:
    """Train the geoscience tokenizer and report fertility vs the base."""
    cfg = _cfg(config)
    from .tokenizer import analyze, train as tok_train
    out = tok_train.train(cfg.tokenizer)
    analyze.report(out, cfg.train.hf_base_model)


@app.command()
def extend(config: Optional[str] = ConfigOpt,
           quantize: bool = typer.Option(True, help="Also convert to 4-bit MLX.")) -> None:
    """Graft domain tokens onto the base model (embedding surgery)."""
    cfg = _cfg(config)
    from .tokenizer import extend as ext
    out = ext.extend(cfg.train.hf_base_model, cfg.tokenizer)
    ext.verify(out, cfg.train.hf_base_model)
    if quantize:
        from .train.mlx_train import quantize_for_mlx
        quantize_for_mlx(out, paths.MODELS / f"{out.name}-mlx-4bit")


@app.command()
def train(config: Optional[str] = ConfigOpt,
          backend: Optional[str] = typer.Option(None, help="mlx | hf"),
          model: Optional[str] = typer.Option(None, help="Override the base model path."),
          name: Optional[str] = typer.Option(None, help="Run name."),
          data_dir: Optional[str] = typer.Option(None, help="Alternate dataset directory."),
          train_embeddings: bool = typer.Option(
              False, help="HF only: also train grafted embedding rows.")) -> None:
    """Fine-tune. MLX by default; --backend hf for the PEFT/TRL path."""
    cfg = _cfg(config)
    backend = backend or cfg.train.backend
    if backend == "mlx":
        from .train.mlx_train import run as run_train
        summary = run_train(cfg, model_path=model, run_name=name, data_dir=data_dir)
    else:
        from .train.hf_train import run as run_train
        summary = run_train(cfg, model_path=model, run_name=name,
                            train_embeddings=train_embeddings)
    console.print(f"[green]done[/green] -> {summary.get('adapter_path')}")


@app.command(name="eval")
def eval_cmd(config: Optional[str] = ConfigOpt,
             adapter: Optional[str] = typer.Option(None, help="Run dir with adapters."),
             model: Optional[str] = typer.Option(None, help="Base model path."),
             gold: bool = typer.Option(False, help="Score against the gold set."),
             n: Optional[int] = typer.Option(None, help="Number of examples.")) -> None:
    """Score base vs fine-tuned on held-out data."""
    cfg = _cfg(config)
    if n:
        cfg.eval.n_examples = n
    from .eval.evaluate import run as run_eval
    run_eval(cfg, adapter_path=adapter, model_path=model, use_gold=gold)


@app.command()
def review(n: int = typer.Option(20, help="How many gold rows to show."),
           start: int = typer.Option(0)) -> None:
    """Print gold rows for hand-checking.

    Correct the JSON in data/gold/gold.jsonl and set "reviewed": true. Until
    you do, `geosft eval --gold` is scoring rules against rules.
    """
    src = paths.GOLD / "gold.jsonl"
    rows = [json.loads(l) for l in src.open()]
    done = sum(1 for r in rows if r.get("reviewed"))
    console.print(f"[bold]{done}/{len(rows)}[/bold] gold rows reviewed\n")
    for i, row in enumerate(rows[start:start + n], start=start):
        console.rule(f"[{i}] {row.get('unit_name')}  {'✓' if row.get('reviewed') else ''}")
        console.print(row["passage"])
        console.print_json(json.dumps(row["target"]))


@app.command()
def inspect(index: int = typer.Option(0, help="Which training row to inspect."),
            config: Optional[str] = ConfigOpt,
            model: Optional[str] = typer.Option(None, help="Base model path.")) -> None:
    """Show exactly what the model is trained on: prompt vs loss region.

    This is the single most useful thing to look at before starting a run, and
    the check that catches the whole family of silent prompt/masking bugs.
    You should see:

      * the PROMPT (grey) carrying no loss -- system, passage, and the
        assistant turn marker,
      * the LOSS REGION (green) containing the JSON answer AND the EOS token
        (without EOS the model never learns to stop),
      * and the prompt matching what inference will send, byte for byte.
    """
    cfg = _cfg(config)
    from mlx_lm import load

    from .schema import build_messages
    from .train.mlx_train import PromptCompletionDataset

    rows = [json.loads(l) for l in (paths.PROCESSED / "train.jsonl").open()]
    row = rows[index]
    _, tok = load(model or cfg.train.base_model)

    tokens, offset = PromptCompletionDataset([row], tok).process(row)
    prompt_txt = tok.decode(tokens[:offset])
    answer_txt = tok.decode(tokens[offset:])

    # escape(): the schema hint contains "[string]", which rich would parse as
    # console markup and silently delete -- making the prompt look malformed.
    console.print(Panel(escape(prompt_txt),
                        title=f"PROMPT — masked, no loss ({offset} tokens)",
                        border_style="bright_black"))
    console.print(Panel(escape(answer_txt),
                        title=f"LOSS REGION — trained ({len(tokens) - offset} tokens)",
                        border_style="green"))

    passage = json.loads((paths.PROCESSED / "train.meta.jsonl").read_text()
                         .splitlines()[index])["passage"]
    infer_prompt = tok.apply_chat_template(
        build_messages(passage), tokenize=False, add_generation_prompt=True
    )
    match = infer_prompt == prompt_txt
    pct = 100 * offset / max(len(tokens), 1)

    t = Table(show_header=False, box=None)
    t.add_row("total tokens", str(len(tokens)))
    t.add_row("masked (prompt)", f"{offset}  ({pct:.0f}%)")
    t.add_row("trained (answer)", f"{len(tokens) - offset}  ({100 - pct:.0f}%)")
    t.add_row("ends with EOS",
              "[green]yes[/green]" if tokens[-1] == tok.eos_token_id
              else "[red]NO — model will not learn to stop[/red]")
    t.add_row("train prompt == inference prompt",
              "[green]yes[/green]" if match else "[red]NO — silent failure ahead[/red]")
    console.print(t)
    if not match:
        console.print("[red]Prompt mismatch. See docs/LEARNING.md section 4.[/red]")


@app.command()
def subsample(n: int = typer.Argument(..., help="Training rows to keep."),
              out: str = typer.Option(None, help="Output directory."),
              seed: int = typer.Option(17)) -> None:
    """Write a smaller copy of the dataset, for the data-size experiment.

    valid/test are copied unchanged so every point on the curve is scored
    against the same held-out data -- otherwise you are measuring two things
    at once.
    """
    import random

    out_dir = Path(out or paths.DATA / f"processed-{n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = (paths.PROCESSED / "train.jsonl").read_text().splitlines()
    random.Random(seed).shuffle(rows)
    (out_dir / "train.jsonl").write_text("\n".join(rows[:n]))
    for split in ("valid", "test"):
        shutil.copy(paths.PROCESSED / f"{split}.jsonl", out_dir / f"{split}.jsonl")
    console.print(f"{min(n, len(rows))} train rows (of {len(rows)}) -> {out_dir}")
    console.print(f"train with: geosft train --data-dir {out_dir} --name size-{n}")


@app.command()
def runs() -> None:
    """Tabulate every run: the comparison table IS the deliverable."""
    table = Table(title="runs")
    for col in ("run", "iters", "best val", "test loss", "macro F1", "parse",
                "min", "alerts"):
        table.add_column(col, justify="right" if col != "run" else "left")

    found = False
    for d in sorted(paths.RUNS.glob("*")):
        summary_p, eval_p = d / "summary.json", d / "eval_report.json"
        if not summary_p.exists():
            continue
        found = True
        s = json.loads(summary_p.read_text())
        f1 = parse = "-"
        if eval_p.exists():
            res = json.loads(eval_p.read_text()).get("results", {})
            tuned = res.get("tuned") or res.get("base")
            if tuned:
                f1 = f"{tuned['macro_f1']:.3f}"
                parse = f"{tuned['gates']['parse_rate']:.3f}"

        def fmt(v, spec=".4f"):
            return format(v, spec) if isinstance(v, (int, float)) else "-"

        table.add_row(d.name, str(s.get("iters_done", "-")),
                      fmt(s.get("best_val_loss")), fmt(s.get("test_loss")),
                      f1, parse, fmt(s.get("wall_minutes"), ".1f"),
                      str(len(s.get("alerts", []))))
    if not found:
        console.print("[dim]no runs yet[/dim]")
        return
    console.print(table)


@app.command(name="all")
def run_all(config: Optional[str] = ConfigOpt,
            skip_extend: bool = typer.Option(
                False, help="Skip vocabulary extension (train on the stock base).")) -> None:
    """The full workflow: fetch, build, tokenizer, extend, train, eval."""
    cfg = _cfg(config)
    from .data import build as build_mod, fetch as fetch_mod, vocab as vocab_mod
    from .eval.evaluate import run as run_eval
    from .tokenizer import analyze, train as tok_train
    from .train.mlx_train import run as run_train

    console.rule("[bold]1/6 corpus")
    vocab_mod.build()
    fetch_mod.run(cfg.data.max_units, cfg.data.min_passage_chars, cfg.data.max_passage_chars)
    console.rule("[bold]2/6 dataset")
    build_mod.run(cfg.data)
    console.rule("[bold]3/6 tokenizer")
    tok_dir = tok_train.train(cfg.tokenizer)
    analyze.report(tok_dir, cfg.train.hf_base_model)

    model_path = cfg.train.base_model
    if not skip_extend:
        console.rule("[bold]4/6 vocabulary extension")
        from .tokenizer import extend as ext
        from .train.mlx_train import quantize_for_mlx
        ext_dir = ext.extend(cfg.train.hf_base_model, cfg.tokenizer)
        ext.verify(ext_dir, cfg.train.hf_base_model)
        model_path = str(quantize_for_mlx(ext_dir, paths.MODELS / f"{ext_dir.name}-mlx-4bit"))
    else:
        console.rule("[bold]4/6 vocabulary extension [dim]skipped")

    console.rule("[bold]5/6 training")
    summary = run_train(cfg, model_path=model_path)
    console.rule("[bold]6/6 evaluation")
    run_eval(cfg, adapter_path=summary["adapter_path"], model_path=model_path)
    console.print("\n[bold green]pipeline complete[/bold green]")
    console.print("Next: hand-check data/gold/gold.jsonl, then `geosft eval --gold`.")


if __name__ == "__main__":
    app()
