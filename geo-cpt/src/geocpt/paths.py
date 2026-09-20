"""Every path in the project, in one place.

Note the sizes here are a different order of magnitude from the SFT project
next door. CPT checkpoints are FULL MODEL COPIES (~3.4 GB each for a 1.7B
model in bf16), not 58 MB adapters. Budget disk accordingly and prune often.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
SFT_PROJECT = REPO / "geo-sft"          # the SFT lab we reuse corpus + scorer from

DATA = ROOT / "data"
RAW = DATA / "raw"                      # untouched API responses
INTERIM = DATA / "interim"              # per-source cleaned documents
CORPUS = DATA / "corpus"                # deduped, filtered, mixed, shard-ready
PACKED = DATA / "packed"                # fixed-length token blocks

ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"           # adapted model checkpoints (GBs each)
RUNS = ROOT / "runs"

HF_HOME = Path(os.environ.get("GEOCPT_HF_HOME", SFT_PROJECT / "artifacts" / "hf-cache"))

ALL = [RAW, INTERIM, CORPUS, PACKED, MODELS, RUNS]


def ensure() -> None:
    for p in ALL:
        p.mkdir(parents=True, exist_ok=True)
    # share the SFT project's model cache -- these are multi-GB downloads and
    # there is no reason to hold two copies of the same base model
    HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
