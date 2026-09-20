"""Every path in the project, in one place."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"              # untouched API responses
INTERIM = DATA / "interim"      # parsed passages + gazetteers
PROCESSED = DATA / "processed"  # train/valid/test.jsonl, ready for a trainer
GOLD = DATA / "gold"            # hand-checked evaluation set

ARTIFACTS = ROOT / "artifacts"
TOKENIZERS = ARTIFACTS / "tokenizers"
MODELS = ARTIFACTS / "models"   # quantised bases, merged/extended models
RUNS = ROOT / "runs"            # one directory per training run

#: Keep multi-GB model weights inside the project so they are easy to find and
#: delete. Override with GEOSFT_HF_HOME if your project disk is tight.
HF_HOME = Path(os.environ.get("GEOSFT_HF_HOME", ARTIFACTS / "hf-cache"))

ALL = [RAW, INTERIM, PROCESSED, GOLD, TOKENIZERS, MODELS, RUNS, HF_HOME]


def ensure() -> None:
    for p in ALL:
        p.mkdir(parents=True, exist_ok=True)
    # transformers/huggingface_hub read these at import time
    os.environ.setdefault("HF_HOME", str(HF_HOME))
