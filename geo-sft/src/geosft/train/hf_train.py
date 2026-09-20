"""LoRA fine-tuning with HuggingFace PEFT + TRL -- the portable path.

READ THIS FIRST: THIS IS NOT QLoRA ON A MAC
-------------------------------------------
True QLoRA needs bitsandbytes 4-bit (NF4) kernels, and bitsandbytes has no
working MPS backend. On Apple Silicon this path therefore runs **bf16 LoRA on
a full-precision base**: more memory, slower, but numerically straightforward.
The MLX path is where you get real 4-bit QLoRA on this machine. The same code
here becomes genuine QLoRA unchanged on an NVIDIA box -- set
`load_in_4bit=True` in the commented block below.

So why keep it? Two reasons that are worth the disk space:

1. **Transferable skill.** PEFT/TRL is what you will meet everywhere else.
   Seeing the same dataset, schema and metrics run through both stacks makes
   the concepts separable from the framework.
2. **Trainable new embeddings.** This is the only path that can actually train
   the grafted vocabulary rows, via PEFT `modules_to_save`. MLX LoRA adapts
   attention and MLP projections only, so extended tokens keep their
   mean-initialised embeddings there. If you want to know whether vocabulary
   extension *earns its place*, you have to measure it here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console

from .. import paths
from ..config import Config
from ..monitor.tracker import RunTracker

console = Console()


def _bf16_ok(device: str) -> bool:
    import torch
    if device == "cpu":
        return False
    if device == "mps":
        try:
            torch.zeros(2, dtype=torch.bfloat16, device="mps") + 1
            return True
        except Exception:
            return False
    return torch.cuda.is_bf16_supported()


def _load_split(split: str) -> list[dict]:
    """TRL prompt-completion format.

    Deliberately NOT `assistant_only_loss`: that requires `{% generation %}`
    markers in the chat template, which many templates (Qwen3 included) do not
    carry, and it fails loudly mid-run. A prompt/completion dataset gets
    completion-only loss from TRL unconditionally.
    """
    path = paths.PROCESSED / f"{split}.jsonl"
    rows = []
    for line in path.open():
        msgs = json.loads(line)["messages"]
        rows.append({"prompt": msgs[:-1], "completion": [msgs[-1]]})
    return rows


class _HFCallback:
    """Bridges transformers' TrainerCallback onto our RunTracker."""

    def __init__(self, tracker: RunTracker):
        self.tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        it = int(state.global_step)
        if "loss" in logs:
            self.tracker.log_train({
                "iteration": it, "train_loss": logs["loss"],
                "learning_rate": logs.get("learning_rate", 0.0),
            })
        if "eval_loss" in logs:
            self.tracker.log_val({"iteration": it, "val_loss": logs["eval_loss"]})


def run(cfg: Config, model_path: str | None = None, run_name: str | None = None,
        train_embeddings: bool = False) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    paths.ensure()
    tcfg = cfg.train
    model_path = model_path or tcfg.hf_base_model
    run_dir = paths.RUNS / (run_name or f"{cfg.name}-hf-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(run_dir / "config.yaml")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"[bold]loading[/bold] {model_path} on {device} (bf16 LoRA, not 4-bit)")

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    # --- on CUDA, swap the two lines above for real QLoRA: -----------------
    # from transformers import BitsAndBytesConfig
    # from peft import prepare_model_for_kbit_training
    # qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    #                         bnb_4bit_compute_dtype=torch.bfloat16,
    #                         bnb_4bit_use_double_quant=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=qc)
    # model = prepare_model_for_kbit_training(model)
    model.to(device)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # Gradient checkpointing + a frozen base is the classic "element 0 of
    # tensors does not require grad" crash: the checkpointed block sees only
    # frozen inputs and drops out of the autograd graph. Forcing the input
    # embeddings to emit grad-requiring activations reconnects it.
    if tcfg.grad_checkpoint:
        model.enable_input_require_grads()

    # Resize only when the tokenizer genuinely outgrew the matrix -- the same
    # "never shrink" rule as in tokenizer/extend.py.
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))

    peft_cfg = LoraConfig(
        r=tcfg.lora_rank,
        lora_alpha=int(tcfg.lora_alpha),
        lora_dropout=tcfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # The knob MLX cannot offer: make the grafted embedding rows trainable.
        # Costs a full extra copy of the embedding matrix in the checkpoint.
        modules_to_save=["embed_tokens", "lm_head"] if train_embeddings else None,
    )

    train_ds = Dataset.from_list(_load_split("train"))
    val_ds = Dataset.from_list(_load_split("valid"))

    sft_cfg = SFTConfig(
        output_dir=str(run_dir),
        max_length=tcfg.max_seq_length,
        per_device_train_batch_size=tcfg.batch_size,
        gradient_accumulation_steps=tcfg.grad_accumulation_steps,
        max_steps=tcfg.iters,
        learning_rate=tcfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=tcfg.warmup_steps,
        logging_steps=tcfg.steps_per_report,
        eval_strategy="steps",
        eval_steps=tcfg.steps_per_eval,
        save_steps=tcfg.save_every,
        save_total_limit=2,
        # MPS bf16 support varies by torch build; falling back to fp32 is slow
        # but correct, whereas an unsupported bf16 flag fails mid-run.
        bf16=_bf16_ok(device),
        fp16=False,
        gradient_checkpointing=tcfg.grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,     # the mask_prompt equivalent
        packing=False,
        report_to=[],
        seed=tcfg.seed,
        dataloader_pin_memory=False,   # pinning is a no-op and warns on MPS
    )

    tracker = RunTracker(run_dir=run_dir, total_iters=tcfg.iters)
    bridge = _HFCallback(tracker)

    class _Cb(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):  # noqa: ANN001
            bridge.on_log(args, state, control, logs=logs, **kw)

    trainer = SFTTrainer(
        model=model, args=sft_cfg,
        train_dataset=train_ds, eval_dataset=val_ds,
        peft_config=peft_cfg, callbacks=[_Cb()],
    )
    trainer.model.print_trainable_parameters()
    try:
        trainer.train()
    finally:
        summary = tracker.close()

    trainer.save_model(str(run_dir / "adapter"))
    tok.save_pretrained(run_dir / "adapter")
    summary.update({"adapter_path": str(run_dir / "adapter"), "model_path": model_path,
                    "backend": "hf", "train_embeddings": train_embeddings})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
