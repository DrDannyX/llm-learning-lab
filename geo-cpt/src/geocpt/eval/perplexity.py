"""The two numbers that define a CPT run.

    domain perplexity   did it LEARN?
    general perplexity  what did it FORGET?

Reporting only the first is how people convince themselves a CPT run worked
while quietly degrading the model at everything else. Catastrophic forgetting
is the signature failure of continued pretraining in the way that overfitting
is the signature failure of SFT, and it is invisible unless you deliberately
hold out off-domain text and measure it.

Perplexity = exp(mean next-token cross-entropy): roughly "how many equally
likely options is the model choosing between at each token". Lower is better.
It is comparable ONLY between models sharing a tokenizer and evaluated on
identical text -- which is why this module always scores both models on the
same packed blocks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def perplexity(model, blocks: np.ndarray, batch_size: int = 2,
               max_batches: int = 200) -> float:
    from ..train.cpt import evaluate
    return float(np.exp(evaluate(model, blocks, batch_size, max_batches)))


def compare(base_model: str, adapted_path: str | Path, stage: str,
            batch_size: int = 2, max_batches: int = 200) -> dict:
    """Score the original and the CPT'd model on identical held-out blocks."""
    from mlx_lm.utils import load

    from .. import paths
    from ..pack import load as load_packed

    stage_dir = paths.PACKED / stage
    domain = load_packed(stage_dir / "val.npy")
    gen_p = stage_dir / "val_general.npy"
    general = load_packed(gen_p) if gen_p.exists() else np.zeros((0, 1), np.int32)

    results: dict[str, dict] = {}
    for label, path in (("base", base_model), ("adapted", str(adapted_path))):
        console.print(f"scoring [cyan]{label}[/cyan]: {path}")
        model, _ = load(path)
        model.eval()
        results[label] = {
            "domain_ppl": perplexity(model, domain, batch_size, max_batches),
            "general_ppl": perplexity(model, general, batch_size, max_batches),
        }
        del model

    b, a = results["base"], results["adapted"]
    report = {
        "stage": stage,
        "base_model": base_model,
        "adapted": str(adapted_path),
        "domain_blocks": int(domain.shape[0]),
        "general_blocks": int(general.shape[0]),
        "results": results,
        "domain_improvement_pct": 100 * (b["domain_ppl"] - a["domain_ppl"]) / b["domain_ppl"],
        "forgetting_pct": 100 * (a["general_ppl"] - b["general_ppl"]) / b["general_ppl"],
    }

    t = Table(title="continued pre-training: learn vs forget")
    t.add_column("corpus"); t.add_column("base", justify="right")
    t.add_column("adapted", justify="right"); t.add_column("change", justify="right")
    d = report["domain_improvement_pct"]
    f = report["forgetting_pct"]
    t.add_row("geoscience (domain)", f"{b['domain_ppl']:.2f}", f"{a['domain_ppl']:.2f}",
              f"[{'green' if d > 0 else 'red'}]{-d:+.1f}%[/]")
    t.add_row("general English", f"{b['general_ppl']:.2f}", f"{a['general_ppl']:.2f}",
              f"[{'red' if f > 1 else 'green'}]{f:+.1f}%[/]")
    console.print(t)
    console.print(
        "[dim]domain should FALL (learning); general should stay flat. A large "
        "general rise is catastrophic forgetting — lower the LR or raise "
        "replay_fraction.[/dim]")
    return report


def save(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    console.print(f"-> {out}")
