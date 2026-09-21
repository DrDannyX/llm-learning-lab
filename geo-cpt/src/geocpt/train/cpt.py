"""The continued-pre-training loop.

WHY THIS IS HAND-WRITTEN AND NOT mlx_lm's TRAINER
-------------------------------------------------
The SFT lab next door uses mlx-lm's trainer, which is excellent but hides the
objective behind a dataset abstraction. For learning CPT the objective is the
whole point, so here it is in four lines and nothing else:

    logits = model(block[:, :-1])
    loss   = cross_entropy(logits, block[:, 1:]).mean()

**No mask.** Every token contributes. That single difference from SFT --
where ~72% of tokens were masked out -- is what makes this pretraining rather
than instruction tuning.

FOUR THINGS THAT DIFFER FROM THE SFT RUN
----------------------------------------
1. **Learning rate is 10x lower** (1e-5 vs 1e-4). The model already sits in a
   good minimum. CPT nudges it; too large a step is the fastest route to
   catastrophic forgetting.
2. **Gradient clipping matters.** In LoRA the adapter is small and updates are
   naturally bounded. In full fine-tuning a single bad batch can wreck every
   weight in the model, so we clip the global grad norm.
3. **Two validation sets, always.** Domain perplexity says whether it learned.
   General perplexity says what it FORGOT. A run that improves the first while
   wrecking the second is a failed run, and you cannot see that with one number.
4. **Checkpoints are full models** (~3.4 GB each at 1.7B), not 58 MB adapters.
   We prune to `keep_last_n` automatically or the disk disappears.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from rich.console import Console

from .. import paths
from ..config import Config

console = Console()


def build_schedule(lr: float, total_opt_steps: int, warmup_opt_steps: int,
                   kind: str, min_fraction: float):
    """Warmup then cosine decay, expressed in OPTIMIZER steps.

    The SFT project next door lost a whole run to this: MLX indexes schedules
    by optimizer steps, and with gradient accumulation there is one optimizer
    step every `grad_accumulation_steps` iterations. Config here is therefore
    converted explicitly, and the resolved numbers are printed before training.
    """
    warmup_opt_steps = min(max(warmup_opt_steps, 0), max(total_opt_steps - 1, 0))
    if kind == "constant":
        if not warmup_opt_steps:
            return lr
        return optim.join_schedules(
            [optim.linear_schedule(0.0, lr, warmup_opt_steps),
             optim.constant_schedule(lr)], [warmup_opt_steps + 1])
    decay = optim.cosine_decay(lr, max(total_opt_steps - warmup_opt_steps, 1),
                               lr * min_fraction)
    if not warmup_opt_steps:
        return decay
    return optim.join_schedules(
        [optim.linear_schedule(0.0, lr, warmup_opt_steps), decay],
        [warmup_opt_steps + 1])


def loss_on_blocks(model, blocks: mx.array) -> tuple[mx.array, int]:
    """Next-token cross-entropy over an entire block. No masking."""
    inputs, targets = blocks[:, :-1], blocks[:, 1:]
    logits = model(inputs).astype(mx.float32)
    losses = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    )
    return losses.mean(), int(targets.size)


def evaluate(model, blocks: np.ndarray, batch_size: int, max_batches: int) -> float:
    """Mean next-token loss over held-out blocks. exp(loss) is perplexity."""
    if blocks.shape[0] == 0:
        return float("nan")
    model.eval()
    total, n = 0.0, 0
    for i in range(0, min(blocks.shape[0], max_batches * batch_size), batch_size):
        batch = mx.array(blocks[i:i + batch_size])
        if batch.shape[0] == 0:
            break
        l, _ = loss_on_blocks(model, batch)
        total += float(l.item())
        n += 1
    model.train()
    return total / max(n, 1)


def _prune_checkpoints(run_dir: Path, keep: int) -> None:
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in ckpts[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
        console.print(f"[dim]pruned {old.name} (checkpoints are full models)[/dim]")


def save_checkpoint(model, tokenizer, run_dir: Path, step: int, keep_last_n: int,
                    src_repo: str, config: dict) -> Path:
    """Write a full, reloadable model checkpoint.

    THREE TRAPS, TWO OF WHICH BIT THIS PROJECT
    ------------------------------------------
    1. `mlx_lm.utils` has `save`, `save_model` and `save_config`. There is no
       `save_weights`. Guessing that name meant the first checkpoint raised
       ImportError at iteration 300 -- 48 minutes in -- and took the run with
       it. **Exercise your checkpoint path before any long run.**
    2. The convenient `save()` wrapper re-resolves the SOURCE repo to copy
       auxiliary files, so it calls `snapshot_download` and fails on an
       incomplete cache with `IncompleteSnapshotError: missing .gitattributes,
       LICENSE, README.md`. Checkpointing must never depend on the network:
       the one thing you need at hour three of a run is the ability to save.
       So we call the primitives directly.
    3. `save()`/`save_model()` accept `donate_model`, which frees the model's
       buffers to save memory -- destroying the model you are still training.
       Must be False here.
    """
    from mlx_lm.utils import save_config, save_model

    out = run_dir / f"checkpoint-{step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    save_model(out, model, donate_model=False)
    save_config(dict(config), config_path=out / "config.json")
    try:
        tokenizer.save_pretrained(out)
    except Exception as exc:  # a tokenizer hiccup must not lose the weights
        console.print(f"[yellow]tokenizer not saved: {exc}[/yellow]")
    _prune_checkpoints(run_dir, keep_last_n)
    return out


def run(cfg: Config, run_name: str | None = None) -> dict:
    from mlx_lm.utils import load

    from geosft.monitor.tracker import RunTracker  # reused from the SFT lab

    from ..pack import load as load_packed

    paths.ensure()
    tcfg = cfg.train
    run_dir = paths.RUNS / (run_name or f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(run_dir / "config.yaml")
    mx.random.seed(tcfg.seed)

    stage_dir = paths.PACKED / cfg.corpus.stage_dir
    train_blocks = load_packed(stage_dir / "train.npy")
    domain_val = load_packed(stage_dir / "val.npy")
    general_val_p = stage_dir / "val_general.npy"
    general_val = load_packed(general_val_p) if general_val_p.exists() else np.zeros((0, 1), np.int32)

    console.print(f"[bold]loading[/bold] {tcfg.base_model} (unquantised — full FT needs real weights)")
    model, tokenizer, model_config = load(tcfg.base_model, return_config=True)

    if tcfg.fine_tune_type == "lora":
        from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
        model.freeze()
        n_layers = len(model.layers) if tcfg.lora_layers == -1 else tcfg.lora_layers
        linear_to_lora_layers(model, n_layers,
                              {"rank": tcfg.lora_rank, "scale": 2.0, "dropout": 0.0})
        print_trainable_parameters(model)
    else:
        n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
        console.print(f"[bold]full fine-tune[/bold]: all {n_params/1e9:.3f}B parameters trainable "
                      f"(~{n_params*2/1e9:.1f} GB weights + grads + optimizer state)")

    if tcfg.grad_checkpoint:
        # Recompute activations in the backward pass instead of storing them.
        # mlx-lm patches the layer CLASS via layer[0], so this covers every
        # block. Costs ~30% time, saves a large slice of activation memory --
        # the difference between batch 2 and batch 4 fitting.
        from mlx_lm.tuner.trainer import grad_checkpoint as _enable_ckpt
        _enable_ckpt(model.layers[0])
        console.print("[dim]gradient checkpointing enabled[/dim]")

    accum = max(tcfg.grad_accumulation_steps, 1)
    total_opt = max(tcfg.iters // accum, 1)
    warmup_opt = max(tcfg.warmup_iters // accum, 1) if tcfg.warmup_iters else 0
    sched = build_schedule(tcfg.learning_rate, total_opt, warmup_opt,
                           tcfg.lr_schedule, tcfg.min_lr_fraction)
    opt = optim.AdamW(learning_rate=sched, weight_decay=tcfg.weight_decay)

    console.print(
        f"blocks: train={train_blocks.shape[0]:,} domain_val={domain_val.shape[0]:,} "
        f"general_val={general_val.shape[0]:,} | block={train_blocks.shape[1]}")
    console.print(
        f"schedule: {tcfg.iters} iters = [bold]{total_opt} optimizer steps[/bold] "
        f"(accum {accum}), warmup {warmup_opt} steps, peak lr {tcfg.learning_rate:.1e}")

    loss_and_grad = nn.value_and_grad(model, loss_on_blocks)
    tracker = RunTracker(run_dir=run_dir, total_iters=tcfg.iters)
    forgetting: list[tuple[int, float, float]] = []
    rng = np.random.default_rng(tcfg.seed)
    base_general = base_domain = None
    accumulated = None
    tokens_seen = 0
    t0 = time.perf_counter()

    try:
        # Baseline BEFORE any training: without it you cannot compute forgetting.
        base_domain = evaluate(model, domain_val, tcfg.batch_size, cfg.eval.max_blocks)
        base_general = evaluate(model, general_val, tcfg.batch_size, cfg.eval.max_blocks)
        console.print(f"[bold]baseline[/bold] domain ppl {np.exp(base_domain):.2f} | "
                      f"general ppl {np.exp(base_general):.2f}")
        tracker.log_val({"iteration": 0, "val_loss": base_domain})
        forgetting.append((0, base_domain, base_general))

        # Restart the clock AFTER the baseline evaluation. Otherwise the eval's
        # wall time is charged to training and tokens/sec is understated --
        # measured 120 tok/s reported against 266 tok/s actual on the first run.
        t0 = time.perf_counter()
        model.train()
        for it in range(1, tcfg.iters + 1):
            idx = rng.integers(0, train_blocks.shape[0], size=tcfg.batch_size)
            batch = mx.array(train_blocks[idx])
            (lval, ntok), grads = loss_and_grad(model, batch)

            # MEMORY NOTE: this accumulator is a FULL extra copy of the
            # gradient tree (~3.4 GB at 1.7B), held for the whole accumulation
            # window. In LoRA the gradient tree is tiny so nobody notices; in
            # full fine-tuning it is the difference between fitting and not.
            # Measured peak on an M4 Pro, batch 2 + accumulation + gradient
            # checkpointing: 41.8 GB, against a 40.2 GB recommended working
            # set. Use batch_size 1 if you want headroom.
            accumulated = grads if accumulated is None else tree_map(
                lambda a, b: a + b, accumulated, grads)

            if it % accum == 0:
                accumulated = tree_map(lambda g: g / accum, accumulated)
                if tcfg.max_grad_norm and tcfg.max_grad_norm > 0:
                    accumulated, _ = optim.clip_grad_norm(accumulated, tcfg.max_grad_norm)
                opt.update(model, accumulated)
                accumulated = None
            mx.eval(model.parameters(), opt.state, lval)
            tokens_seen += ntok

            if it % tcfg.steps_per_report == 0:
                tracker.log_train({
                    "iteration": it, "train_loss": float(lval.item()),
                    "learning_rate": float(opt.learning_rate.item()),
                    "trained_tokens": tokens_seen,
                    "peak_memory": mx.get_peak_memory() / 1e9,
                    "tokens_per_second": tokens_seen / max(time.perf_counter() - t0, 1e-9),
                })

            if it % tcfg.steps_per_eval == 0 or it == tcfg.iters:
                d = evaluate(model, domain_val, tcfg.batch_size, cfg.eval.max_blocks)
                g = evaluate(model, general_val, tcfg.batch_size, cfg.eval.max_blocks)
                forgetting.append((it, d, g))
                tracker.log_val({"iteration": it, "val_loss": d})
                drift = np.exp(g) - np.exp(base_general)
                console.print(
                    f"  iter {it}: domain ppl [green]{np.exp(d):.2f}[/green] "
                    f"(from {np.exp(base_domain):.2f}) | general ppl "
                    f"{np.exp(g):.2f} ([{'red' if drift > 0.5 else 'green'}]{drift:+.2f}[/])")

            if tcfg.save_every and it % tcfg.save_every == 0:
                save_checkpoint(model, tokenizer, run_dir, it, tcfg.keep_last_n,
                                tcfg.base_model, model_config)
    finally:
        summary = tracker.close()

    final = save_checkpoint(model, tokenizer, run_dir, tcfg.iters,
                            tcfg.keep_last_n, tcfg.base_model, model_config)
    d_end, g_end = forgetting[-1][1], forgetting[-1][2]
    summary.update({
        "stage": cfg.corpus.stage_dir,
        "base_model": tcfg.base_model,
        "fine_tune_type": tcfg.fine_tune_type,
        "checkpoint": str(final),
        "domain_ppl_before": float(np.exp(base_domain)),
        "domain_ppl_after": float(np.exp(d_end)),
        "domain_ppl_improvement_pct": float(100 * (np.exp(base_domain) - np.exp(d_end))
                                            / np.exp(base_domain)),
        "general_ppl_before": float(np.exp(base_general)),
        "general_ppl_after": float(np.exp(g_end)),
        "forgetting_pct": float(100 * (np.exp(g_end) - np.exp(base_general))
                                / np.exp(base_general)),
        "tokens_seen": tokens_seen,
        "curve": [{"iter": i, "domain_ppl": float(np.exp(a)),
                   "general_ppl": float(np.exp(b))} for i, a, b in forgetting],
    })
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    console.print(
        f"\n[bold]domain[/bold] {summary['domain_ppl_before']:.2f} -> "
        f"{summary['domain_ppl_after']:.2f} "
        f"([green]{summary['domain_ppl_improvement_pct']:+.1f}%[/green])   "
        f"[bold]general[/bold] {summary['general_ppl_before']:.2f} -> "
        f"{summary['general_ppl_after']:.2f} "
        f"([red]{summary['forgetting_pct']:+.1f}%[/red] = forgetting)")
    return summary
