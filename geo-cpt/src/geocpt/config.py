"""Typed configuration for a CPT run."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

Stage = Literal["tapt", "dapt", "dapt_tapt"]


class CorpusCfg(BaseModel):
    #: Which corpus this run trains on.
    #:   dapt  = broad geoscience domain text (GA publication abstracts, all of ASUD)
    #:   tapt  = the unlabelled text of the DOWNSTREAM TASK (geo-sft's passages)
    #: They are the same technique with different data -- see docs/LEARNING.md.
    stage: Stage = "tapt"

    #: eCat record types to take abstracts from. "document" = GA publications;
    #: "dataset" adds ~10k dataset descriptions, which are drier and repetitive.
    ecat_resource_types: list[str] = Field(default_factory=lambda: ["document"])
    ecat_max_docs: int | None = None       # None = every record (~21k)
    #: Include ASUD's superseded ("not current") units. Their notes are real
    #: geology prose; for CPT the name being obsolete does not matter.
    asud_not_current: bool = True

    min_doc_chars: int = 300
    max_doc_chars: int = 50_000

    #: Apply heuristic quality filtering. Default ON for DAPT, and you should
    #: turn it OFF for pure TAPT: the task corpus IS the target distribution by
    #: definition, so filtering it makes the pretraining distribution differ
    #: from the distribution you will be evaluated on -- the opposite of what
    #: TAPT is for. Measured here: the filter rejects 21.6% of geo-sft's
    #: passages as "no_sentences"/"too_short", and those are perfectly valid
    #: lexicon fragments the downstream task must handle.
    quality_filter: bool = True

    #: Near-duplicate removal. CPT is far more sensitive to duplication than
    #: SFT: repeated text gets memorised verbatim and wastes the whole budget.
    dedup_threshold: float = 0.8           # Jaccard similarity via MinHash/LSH
    dedup_num_perm: int = 128
    dedup_shingle: int = 5                 # word n-gram size

    #: Fraction of the final corpus that is GENERAL text, replayed to fight
    #: catastrophic forgetting. 0.0 disables replay (and makes forgetting the
    #: headline result of your first run).
    replay_fraction: float = 0.15
    replay_dataset: str = "Salesforce/wikitext"
    replay_config: str = "wikitext-2-raw-v1"

    #: THE FORGETTING PROBE MUST NOT COME FROM THE REPLAY CORPUS.
    #: The first TAPT run measured "forgetting" on held-out wikitext while
    #: replaying wikitext into training. Different documents, same
    #: distribution -- so general perplexity IMPROVED 40% and the metric
    #: reported -40% "forgetting". It was measuring "did we also learn
    #: wikitext", not "did we keep our general ability".
    #: This corpus is held out from training entirely and shares no
    #: distribution with the replay data.
    forgetting_dataset: str = "fancyzhx/ag_news"
    forgetting_config: str | None = None
    forgetting_split: str = "test"

    val_docs: int = 400
    seed: int = 17

    #: Suffix appended to the stage directory, so runs that tokenise
    #: DIFFERENTLY do not overwrite each other's packed blocks. Required when
    #: comparing a stock base against a vocabulary-extended one: same corpus,
    #: different tokenizer, therefore different blocks.
    variant: str = ""

    @property
    def stage_dir(self) -> str:
        """Directory name for this stage, including any variant suffix."""
        return f"{self.stage}-{self.variant}" if self.variant else self.stage


class PackCfg(BaseModel):
    #: CPT packs documents end-to-end into fixed-length blocks rather than
    #: padding each document. Padding a 40-token abstract to 2048 wastes 98%
    #: of the compute; packing wastes ~0%.
    block_size: int = 1024
    #: Insert EOS between documents so the model still learns boundaries.
    add_eos_between_docs: bool = True
    #: If True, never let a block span two documents (costs some tokens to
    #: padding, removes cross-document attention contamination entirely).
    no_cross_document: bool = False


class TrainCfg(BaseModel):
    base_model: str = "Qwen/Qwen3-1.7B"     # unquantised: full FT needs real weights
    #: full  = every weight trains. The right choice for injecting knowledge.
    #: lora  = cheap, but LoRA learns less AND forgets less -- good for SFT,
    #:         weak for CPT. Offered so you can measure the difference.
    fine_tune_type: Literal["full", "lora"] = "full"
    lora_rank: int = 64                      # higher than SFT if you use it
    lora_layers: int = -1

    #: An order of magnitude BELOW the SFT learning rate. The model is already
    #: at a good minimum; CPT nudges it, it does not reshape it. Too high here
    #: is the fastest route to catastrophic forgetting.
    learning_rate: float = 1e-5
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    warmup_iters: int = 100
    min_lr_fraction: float = 0.1

    #: Measured on an M4 Pro, Qwen3-1.7B full FT, block_size 1024.
    #: A bare forward+backward+step, no accumulation, no checkpointing:
    #:     batch 1 -> 17.2 GB peak, 2.74 s/step
    #:     batch 2 -> 24.1 GB peak, 5.23 s/step
    #: In the real loop, peak memory is set by the OPTIMIZER STEP, not the
    #: batch. Measured at batch 1, accum 16: 12.3 GB before the first
    #: optimizer step (iteration 16), 49.0 GB immediately after, as AdamW
    #: allocates m and v (~13.6 GB) on top of params + grads + a full extra
    #: gradient tree for accumulation (~3.4 GB each).
    #: Dropping batch 2 -> 1 barely moved peak (48.7 -> 49.0 GB) but cut
    #: steady-state pressure enough to stop swapping: 9.1 -> 3.4 s/iter.
    #: A 4B model roughly doubles all of this, which is why this lab defaults
    #: to 1.7B where the SFT lab used 4B.
    batch_size: int = 2
    grad_accumulation_steps: int = 8
    iters: int = 2000
    max_grad_norm: float = 1.0               # clipping matters more in full FT
    weight_decay: float = 0.01

    steps_per_report: int = 10
    steps_per_eval: int = 100
    save_every: int = 500                    # each save is a FULL model (~3.4 GB)
    keep_last_n: int = 2                     # prune old checkpoints automatically
    grad_checkpoint: bool = True
    seed: int = 17


class EvalCfg(BaseModel):
    #: The two numbers that define a CPT run. Domain perplexity should fall.
    #: General perplexity should NOT rise much -- that is forgetting, and it is
    #: the whole reason replay exists.
    domain_val_docs: int = 300
    general_val_docs: int = 300
    max_blocks: int = 200
    #: Cloze probes: mask a domain term and see if the model recovers it.
    probe_n: int = 200


class Config(BaseModel):
    """A CPT run. `corpus.stage_dir` names the on-disk directory."""

    name: str = "geo-cpt-tapt"
    corpus: CorpusCfg = Field(default_factory=CorpusCfg)
    pack: PackCfg = Field(default_factory=PackCfg)
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
