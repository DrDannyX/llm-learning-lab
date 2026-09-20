"""QLoRA fine-tuning with MLX -- the primary path on Apple Silicon.

WHY QLORA, AND WHAT THE "Q" ACTUALLY MEANS HERE
-----------------------------------------------
The base model is loaded already quantised to 4 bits. Those weights stay
frozen and quantised for the whole run; only the small LoRA matrices are
trained, in bf16/fp32. Memory therefore scales with the adapter, not the
model, which is what makes a 4B-8B fine-tune comfortable in 48 GB of unified
memory -- and why a 4-bit base is not a compromise here but the point.

On an M4 Pro, MLX is typically 2-3x faster than PyTorch/MPS for this workload
because it targets unified memory directly and avoids host<->device copies.

KNOBS THAT MATTER, ROUGHLY IN ORDER
-----------------------------------
mask_prompt   Loss on the JSON answer only. Without it most of your gradient
              is spent teaching the model to reproduce the input passage,
              which is not the task. Single biggest quality lever here.
num_layers    How many transformer blocks (from the top) get adapters. 16 is
              a good default; -1 adapts everything and costs more memory.
rank          Adapter capacity. 8-16 is plenty for a format/extraction task;
              raise it only if train loss plateaus high.
learning_rate 1e-4 is a sane LoRA default. 1e-3 will often diverge; 1e-5
              will look like nothing is happening.
batch_size x grad_accumulation_steps is your EFFECTIVE batch size. Prefer
              raising accumulation over batch size -- it costs time, not RAM.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from rich.console import Console

from .. import paths
from ..config import Config
from ..monitor.tracker import MLXCallback, RunTracker

console = Console()


def quantize_for_mlx(hf_dir: str | Path, out_dir: Path, bits: int = 4,
                     group_size: int = 64) -> Path:
    """Convert (and quantise) a local HF model into MLX format.

    Needed when you train on the vocabulary-extended model: the stock 4-bit
    MLX community weights have the original vocabulary, so the extended model
    has to be quantised yourself.
    """
    from mlx_lm.convert import convert

    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        console.print(f"[dim]quantised model exists -> {out_dir}[/dim]")
        return out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    console.print(f"quantising {hf_dir} to {bits}-bit MLX -> {out_dir}")
    convert(str(hf_dir), str(out_dir), quantize=True, q_bits=bits, q_group_size=group_size)
    return out_dir


class PromptCompletionDataset:
    """Build each training sequence as prompt + answer, exactly as inference sees it.

    THE TEMPLATE-PREFIX TRAP (this repo hit it for real)
    ----------------------------------------------------
    Qwen3's chat template renders a FULL conversation as

        <|im_start|>assistant\n<think>\n\n</think>\n\n{"unit_name": ...}<|im_end|>

    but `add_generation_prompt=True` -- what you actually have at inference --
    stops at `<|im_start|>assistant\n`. MLX's ChatDataset tokenises the full
    form and derives the mask offset from the generation-prompt form, so the
    injected `<think>\n\n</think>\n\n` lands INSIDE the trained region.
    The model then learns, perfectly correctly, to emit that scaffolding
    before every answer. Measured on this project's first full run: every
    single generation was prefixed with an empty think block, so
    `json.loads(output)` failed on 100% of them (strict JSON rate 1.000 on the
    base model, 0.000 after fine-tuning) while macro F1 went UP. Nothing
    errors. The loss curve looks perfect.

    Masking those tokens is NOT enough: masking removes them from the loss but
    leaves them in the input, so the model would be trained to produce JSON
    *conditioned on* a think block that will not be there at inference. The
    only clean fix is to stop using the full-conversation rendering and
    concatenate the two halves ourselves:

        tokens = apply_chat_template(messages[:-1], add_generation_prompt=True)
               + encode(answer) + [eos]
        offset = len(prompt part)

    That is byte-identical to what eval/evaluate.py feeds the model, which is
    the prompt-symmetry property the whole pipeline depends on.
    """

    def __init__(self, data, tokenizer, chat_key: str = "messages"):
        self._data = data
        self._tok = tokenizer
        self._chat_key = chat_key
        eos = getattr(tokenizer, "eos_token_id", None)
        self._eos = [eos] if eos is not None else []

    def process(self, d):
        messages = d[self._chat_key]
        prompt_ids = self._tok.apply_chat_template(
            messages[:-1], add_generation_prompt=True, return_dict=False
        )
        answer_ids = self._tok.encode(messages[-1]["content"], add_special_tokens=False)
        return list(prompt_ids) + list(answer_ids) + self._eos, len(prompt_ids)

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def _schedule(cfg, iters: int):
    """Warmup then cosine decay to 10% of peak.

    THE GRADIENT-ACCUMULATION TRAP
    ------------------------------
    MLX indexes an optimizer schedule by **optimizer steps**, and with
    gradient accumulation there is only one optimizer step every
    `grad_accumulation_steps` iterations. Passing iteration counts straight
    into the schedule therefore stretches it by that factor: with
    grad_accumulation_steps=8, a `warmup_steps=30` config warms up over 240
    iterations, and a 600-iteration run spends 40% of its life at an
    effectively zero learning rate while the loss sits flat. It looks exactly
    like "LoRA doesn't work on my data".

    Config values here are expressed in ITERATIONS, matching `iters`. This
    function converts them to optimizer steps.

    Warmup matters more for LoRA than people expect: the B matrix starts at
    zero, so the first updates are large and badly scaled.
    """
    accum = max(cfg.grad_accumulation_steps, 1)
    total_steps = max(iters // accum, 1)
    warmup_steps = min(max(cfg.warmup_steps // accum, 1) if cfg.warmup_steps else 0,
                       max(total_steps - 1, 0))

    if cfg.lr_schedule == "constant":
        if warmup_steps:
            return optim.join_schedules(
                [optim.linear_schedule(0.0, cfg.learning_rate, warmup_steps),
                 optim.constant_schedule(cfg.learning_rate)],
                [warmup_steps + 1],
            )
        return cfg.learning_rate

    decay = optim.cosine_decay(cfg.learning_rate, max(total_steps - warmup_steps, 1),
                               cfg.learning_rate * 0.1)
    if not warmup_steps:
        return decay
    return optim.join_schedules(
        [optim.linear_schedule(0.0, cfg.learning_rate, warmup_steps), decay],
        [warmup_steps + 1],
    )


def run(cfg: Config, model_path: str | None = None, run_name: str | None = None,
        data_dir: str | Path | None = None) -> dict:
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
    from mlx_lm.utils import load

    paths.ensure()
    tcfg = cfg.train
    model_path = model_path or tcfg.base_model
    # data_dir override lets you train on a subsampled copy of the dataset
    # without touching data/processed (see docs/EXPERIMENTS.md #3).
    data_dir = Path(data_dir) if data_dir else paths.PROCESSED
    run_dir = paths.RUNS / (run_name or f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(run_dir / "config.yaml")

    mx.random.seed(tcfg.seed)
    console.print(f"[bold]loading[/bold] {model_path}")
    model, tokenizer = load(model_path)

    # ---- freeze the base, attach adapters --------------------------------
    model.freeze()
    lora_cfg = {
        "rank": tcfg.lora_rank,
        "scale": tcfg.lora_alpha / tcfg.lora_rank,  # standard alpha/r scaling
        "dropout": tcfg.lora_dropout,
    }
    n_layers = len(model.layers) if tcfg.num_layers == -1 else tcfg.num_layers
    if n_layers > len(model.layers):
        raise ValueError(f"num_layers={n_layers} exceeds model depth {len(model.layers)}")
    linear_to_lora_layers(model, n_layers, lora_cfg, use_dora=(tcfg.fine_tune_type == "dora"))
    print_trainable_parameters(model)

    # ---- data -------------------------------------------------------------
    args = SimpleNamespace(
        data=str(data_dir), train=True, test=True, hf_dataset=None,
        mask_prompt=tcfg.mask_prompt, chat_feature="messages",
        prompt_feature="prompt", completion_feature="completion", text_feature="text",
    )
    train_set, val_set, test_set = load_dataset(args, tokenizer)
    if tcfg.mask_prompt and tcfg.align_loss_to_answer:
        # Re-wrap the raw rows; we do our own tokenisation (see the docstring).
        train_set, val_set, test_set = (
            PromptCompletionDataset(d, tokenizer) if len(d) else d
            for d in (train_set, val_set, test_set)
        )
    console.print(
        f"data: train={len(train_set)} valid={len(val_set)} test={len(test_set)} "
        f"| mask_prompt={tcfg.mask_prompt} | effective batch="
        f"{tcfg.batch_size * tcfg.grad_accumulation_steps}"
    )

    # ---- optimiser --------------------------------------------------------
    opt = optim.AdamW(learning_rate=_schedule(tcfg, tcfg.iters))
    n_opt_steps = max(tcfg.iters // max(tcfg.grad_accumulation_steps, 1), 1)
    console.print(
        f"schedule: {tcfg.iters} iters = [bold]{n_opt_steps} optimizer steps[/bold] "
        f"(accum {tcfg.grad_accumulation_steps}), warmup "
        f"{max(tcfg.warmup_steps // max(tcfg.grad_accumulation_steps, 1), 1)} steps, "
        f"peak lr {tcfg.learning_rate:.1e}"
    )

    targs = TrainingArgs(
        batch_size=tcfg.batch_size,
        iters=tcfg.iters,
        val_batches=tcfg.val_batches,
        steps_per_report=tcfg.steps_per_report,
        steps_per_eval=tcfg.steps_per_eval,
        steps_per_save=tcfg.save_every,
        max_seq_length=tcfg.max_seq_length,
        adapter_file=str(run_dir / "adapters.safetensors"),
        grad_checkpoint=tcfg.grad_checkpoint,
        grad_accumulation_steps=tcfg.grad_accumulation_steps,
    )
    (run_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": tcfg.fine_tune_type,
        "num_layers": n_layers,
        "lora_parameters": lora_cfg,
        "model": model_path,
    }, indent=2))

    tracker = RunTracker(run_dir=run_dir, total_iters=tcfg.iters)
    try:
        model.train()
        # CacheDataset memoises apply_chat_template per example; without it
        # the trainer's length-sorting indexes raw dicts and dies with
        # KeyError: 0.
        train(
            model=model, optimizer=opt,
            train_dataset=CacheDataset(train_set), val_dataset=CacheDataset(val_set),
            args=targs, training_callback=MLXCallback(tracker),
        )
    finally:
        summary = tracker.close()

    # ---- held-out loss ----------------------------------------------------
    if len(test_set):
        model.eval()
        test_loss = evaluate(
            model=model, dataset=CacheDataset(test_set), batch_size=tcfg.batch_size,
            num_batches=-1, max_seq_length=tcfg.max_seq_length,
        )
        summary["test_loss"] = float(test_loss)
        summary["test_perplexity"] = float(mx.exp(mx.array(test_loss)).item())
        console.print(
            f"[bold]test loss[/bold] {summary['test_loss']:.4f} "
            f"(ppl {summary['test_perplexity']:.2f})"
        )

    summary["adapter_path"] = str(run_dir)
    summary["model_path"] = model_path
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
