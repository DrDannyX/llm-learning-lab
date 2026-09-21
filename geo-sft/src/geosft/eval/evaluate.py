"""Generate predictions and score them, base vs fine-tuned.

The headline number of this project is not the tuned model's F1. It is the
*difference* between the tuned model and the same base model given the same
prompt. Always run both -- a fine-tune that scores 0.71 means nothing until
you know the base scored 0.42 (a real win) or 0.70 (you burned an afternoon).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn
from rich.table import Table

from .. import paths
from ..config import Config
from ..schema import build_messages
from .metrics import Scorer

console = Console()


def load_eval_rows(split: str, n: int, use_gold: bool = False) -> list[dict]:
    src = (paths.GOLD / "gold.jsonl") if use_gold else (paths.PROCESSED / f"{split}.meta.jsonl")
    if not src.exists():
        raise FileNotFoundError(f"{src} missing -- run `geosft build` first")
    rows = [json.loads(l) for l in src.open()]
    if use_gold:
        reviewed = [r for r in rows if r.get("reviewed")]
        if reviewed:
            console.print(f"[green]using {len(reviewed)} hand-reviewed gold rows[/green]")
            rows = reviewed
        else:
            console.print(
                "[yellow]gold.jsonl has no rows marked reviewed=true. Scoring against "
                "RULE labels -- treat these numbers as a smoke test, not as truth.[/yellow]"
            )
    return rows[:n]


def generate_mlx(model_path: str, adapter_path: str | None, rows: list[dict],
                 max_new_tokens: int, temperature: float, label: str) -> list[str]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    sampler = make_sampler(temp=temperature)
    outs: list[str] = []
    with Progress(*Progress.get_default_columns(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task(f"generate [{label}]", total=len(rows))
        for row in rows:
            msgs = build_messages(row["passage"])
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            text = generate(model, tokenizer, prompt=prompt,
                            max_tokens=max_new_tokens, sampler=sampler, verbose=False)
            outs.append(text)
            prog.advance(task)
    return outs


def score(rows: list[dict], outputs: list[str]) -> tuple[dict, list[dict]]:
    scorer = Scorer()
    preds = []
    for row, out in zip(rows, outputs):
        parsed = scorer.add(out, row["target"])
        preds.append({
            "unit_name": row.get("unit_name"),
            "passage": row["passage"][:400],
            "gold": row["target"],
            "raw_output": out,
            "parsed": parsed,
            "source_url": row.get("source_url"),
        })
    return scorer.report(), preds


def run(cfg: Config, adapter_path: str | Path | None, model_path: str | None = None,
        use_gold: bool = False, split: str = "test", out_dir: Path | None = None) -> dict:
    paths.ensure()
    # An adapter is only meaningful on the base it was trained against -- and
    # on the vocabulary-extended path the base is NOT cfg.train.base_model.
    # Pairing an adapter with the wrong base produces plausible-looking
    # garbage rather than an error, so recover it from the run's own metadata.
    if model_path is None and adapter_path:
        meta = Path(adapter_path) / "summary.json"
        if meta.exists():
            recorded = json.loads(meta.read_text()).get("model_path")
            if recorded:
                model_path = recorded
                console.print(f"[dim]base model from run metadata: {recorded}[/dim]")
    model_path = model_path or cfg.train.base_model
    rows = load_eval_rows(split, cfg.eval.n_examples, use_gold=use_gold)
    out_dir = Path(out_dir or (Path(adapter_path) if adapter_path else paths.RUNS / "base-only"))
    out_dir.mkdir(parents=True, exist_ok=True)
    # Name outputs by eval set. Gold and test runs previously both wrote
    # eval_report.json, so scoring against gold silently DESTROYED the test
    # results for that run -- and the two are not comparable, so conflating
    # them is worse than losing one.
    # NOTE: this must be defined BEFORE the first prediction file is written.
    # Defining it lower down (next to `report`) raised UnboundLocalError and
    # took out five completed training runs' evaluations.
    tag = "gold" if use_gold else split
    console.print(f"evaluating on [cyan]{len(rows)}[/cyan] examples ({tag})")

    results: dict[str, dict] = {}
    t0 = time.perf_counter()

    if cfg.eval.compare_base:
        base_out = generate_mlx(model_path, None, rows, cfg.eval.max_new_tokens,
                                cfg.eval.temperature, "base")
        results["base"], base_preds = score(rows, base_out)
        (out_dir / f"predictions_base{'' if tag == 'test' else '_' + tag}.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in base_preds))

    if adapter_path:
        tuned_out = generate_mlx(model_path, str(adapter_path), rows,
                                 cfg.eval.max_new_tokens, cfg.eval.temperature, "tuned")
        results["tuned"], tuned_preds = score(rows, tuned_out)
        (out_dir / f"predictions_tuned{'' if tag == 'test' else '_' + tag}.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in tuned_preds))

    report = {
        "model": model_path,
        "adapter": str(adapter_path) if adapter_path else None,
        "eval_set": "gold" if use_gold else split,
        "n": len(rows),
        "minutes": round((time.perf_counter() - t0) / 60, 2),
        "results": results,
    }
    name = "eval_report.json" if tag == "test" else f"eval_report_{tag}.json"
    (out_dir / name).write_text(json.dumps(report, indent=2))
    _print_table(results)
    console.print(f"-> {out_dir / name}")
    return report


def _print_table(results: dict) -> None:
    if not results:
        return
    cols = list(results)
    table = Table(title="extraction quality")
    table.add_column("metric", style="bold")
    for c in cols:
        table.add_column(c, justify="right")
    if len(cols) == 2:
        table.add_column("delta", justify="right")

    def row(label: str, getter) -> None:
        vals = []
        for c in cols:
            try:
                vals.append(float(getter(results[c])))
            except (KeyError, TypeError):
                vals.append(float("nan"))
        cells = [f"{v:.3f}" for v in vals]
        if len(vals) == 2:
            d = vals[1] - vals[0]
            colour = "green" if d > 0.005 else "red" if d < -0.005 else "dim"
            cells.append(f"[{colour}]{d:+.3f}[/{colour}]")
        table.add_row(label, *cells)

    row("parse rate", lambda r: r["gates"]["parse_rate"])
    row("strict JSON rate", lambda r: r["gates"]["strict_json_rate"])
    row("schema valid rate", lambda r: r["gates"]["schema_valid_rate"])
    row("unit_name acc", lambda r: r["scalar_fields"]["unit_name_accuracy"])
    row("rank acc", lambda r: r["scalar_fields"]["rank_accuracy"])
    row("thickness acc", lambda r: r["scalar_fields"]["thickness_accuracy"])
    for f in ("lithologies", "chronostrat", "minerals", "states", "relations"):
        row(f"{f} F1", lambda r, f=f: r["set_fields"][f]["f1"])
    row("MACRO F1", lambda r: r["macro_f1"])
    console.print(table)
