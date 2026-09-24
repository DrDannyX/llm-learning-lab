"""Every path in the project, in one place."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent

#: The corpus is not re-downloaded: this lab reads what geo-sft already built.
SFT_INTERIM = REPO / "geo-sft" / "data" / "interim"
PASSAGES = SFT_INTERIM / "passages.jsonl"
VOCAB = SFT_INTERIM / "vocab.json"

DATA = ROOT / "data"
INTERIM = DATA / "interim"           # extracted graph (JSON) + embedding cache
GRAPH_JSON = INTERIM / "graph.json"
EMBED_CACHE = INTERIM / "embeddings.npy"
EMBED_IDS = INTERIM / "embeddings.ids.json"

BENCH = DATA / "bench"               # generated question sets
RUNS = ROOT / "runs"                 # one directory per benchmark run
CONFIGS = ROOT / "configs"


def ensure() -> None:
    for p in (INTERIM, BENCH, RUNS):
        p.mkdir(parents=True, exist_ok=True)
