"""Typed run configuration, loaded from YAML.

One config object drives fetch -> label -> tokenizer -> train -> eval, so a run
is reproducible from a single file that gets copied into its run directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class DataCfg(BaseModel):
    max_units: int = 4000          # Geolex units to download (16,684 available)
    min_passage_chars: int = 200
    max_passage_chars: int = 4000
    min_filled_fields: int = 2     # drop near-empty targets: they teach silence
    val_frac: float = 0.10
    test_frac: float = 0.10
    gold_n: int = 150              # size of the hand-checkable gold slice
    seed: int = 17
    #: Split on the *unit concept*, not the passage. Two passages about the
    #: Austin Chalk in train and test would leak the answer.
    group_split: bool = True


class TokenizerCfg(BaseModel):
    vocab_size: int = 8000
    model_type: Literal["bpe", "unigram"] = "bpe"
    min_frequency: int = 2
    #: How many domain tokens to graft onto the base model's vocabulary.
    #: 0 disables vocabulary extension (train with the stock tokenizer).
    extend_top_k: int = 512
    #: A candidate must be at least this many base-tokenizer tokens long to be
    #: worth adding. Adding a token the base already encodes as 1 token is pure
    #: embedding bloat.
    min_base_tokens: int = 3


class TrainCfg(BaseModel):
    base_model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    hf_base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    backend: Literal["mlx", "hf"] = "mlx"
    fine_tune_type: Literal["lora", "dora", "full"] = "lora"
    num_layers: int = 16           # -1 = all layers
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    batch_size: int = 4
    grad_accumulation_steps: int = 4
    #: iters counts MICRO-batches, so examples seen = iters * batch_size.
    iters: int = 900
    learning_rate: float = 1e-4
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    #: In ITERATIONS. Converted to optimizer steps internally (see the
    #: gradient-accumulation trap in train/mlx_train.py).
    warmup_steps: int = 60
    max_seq_length: int = 2048
    steps_per_report: int = 10
    steps_per_eval: int = 50
    save_every: int = 100
    val_batches: int = 20
    grad_checkpoint: bool = True
    mask_prompt: bool = True       # loss on the JSON only, not on the passage
    #: Walk the loss offset past template-injected scaffolding (Qwen3 renders
    #: an empty <think></think> before assistant content in full conversations
    #: but not in the generation prompt, so it lands inside the loss region
    #: and the model learns to emit it). See AnswerAlignedDataset.
    align_loss_to_answer: bool = True
    seed: int = 17


class EvalCfg(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.0
    n_examples: int = 200
    compare_base: bool = True      # always score the untuned base too


class Config(BaseModel):
    name: str = "geo-sft-v1"
    data: DataCfg = Field(default_factory=DataCfg)
    tokenizer: TokenizerCfg = Field(default_factory=TokenizerCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(raw)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
