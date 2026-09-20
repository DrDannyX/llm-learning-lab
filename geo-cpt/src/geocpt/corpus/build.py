"""Assemble the training corpus: fetch -> clean -> filter -> dedup -> mix -> split.

The order is deliberate and each step is reported, because a corpus pipeline
that silently drops 60% of your data is worse than none at all.

    fetch     per-source raw documents
    filter    heuristic quality rejection (counted by reason)
    dedup     exact then near-duplicate removal (MinHash/LSH)
    mix       blend in general text as REPLAY against forgetting
    split     hold out validation BEFORE shuffling sources together

The replay step is the one people skip and then wonder why their model got
worse at everything else. See docs/LEARNING.md on catastrophic forgetting.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .. import paths
from ..config import CorpusCfg
from . import dedup as dedup_mod
from . import quality, sources

console = Console()


def _write(path: Path, docs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def gather(cfg: CorpusCfg) -> list[dict]:
    """Collect raw documents according to the configured stage."""
    docs: list[dict] = []
    if cfg.stage in ("dapt", "dapt_tapt"):
        docs += sources.fetch_usgs(cfg.usgs_queries, cfg.usgs_max_per_query)
        docs += sources.fetch_geolex_full(cfg.geolex_max_units)
    if cfg.stage in ("tapt", "dapt_tapt"):
        docs += sources.load_task_text()
    if not docs:
        raise RuntimeError(f"no documents gathered for stage={cfg.stage}")
    return docs


def run(cfg: CorpusCfg, force: bool = False) -> dict:
    paths.ensure()
    out_dir = paths.CORPUS / cfg.stage
    train_p, val_p = out_dir / "train.jsonl", out_dir / "val.jsonl"
    gen_val_p = out_dir / "val_general.jsonl"
    if train_p.exists() and not force:
        console.print(f"[dim]corpus cached -> {out_dir}[/dim]")
        return json.loads((out_dir / "corpus_card.json").read_text())

    raw = gather(cfg)
    console.print(f"gathered [cyan]{len(raw)}[/cyan] raw documents")

    if cfg.quality_filter:
        kept, qstats = quality.filter_docs(raw, cfg.min_doc_chars, cfg.max_doc_chars)
    else:
        # See CorpusCfg.quality_filter: for TAPT the task text is the target
        # distribution, so filtering it would move pretraining AWAY from it.
        kept, qstats = list(raw), {"kept": len(raw), "in": len(raw),
                                   "rejects": {}, "skipped": True}
        console.print("[dim]quality filter skipped (task text is the target "
                      "distribution by definition)[/dim]")
    kept, dstats = dedup_mod.deduplicate(kept, cfg.dedup_threshold,
                                         cfg.dedup_num_perm, cfg.dedup_shingle)

    rng = random.Random(cfg.seed)
    rng.shuffle(kept)

    # Hold out DOMAIN validation before any mixing, so domain perplexity is
    # measured on domain text alone.
    n_val = min(cfg.val_docs, max(len(kept) // 10, 1))
    domain_val, domain_train = kept[:n_val], kept[n_val:]

    # Replay: general text mixed into TRAIN, plus a separate general VAL set
    # that is never trained on -- that held-out set is how forgetting is
    # measured, and it must stay clean.
    replay_train: list[dict] = []
    if cfg.replay_fraction > 0:
        n_replay = int(len(domain_train) * cfg.replay_fraction / (1 - cfg.replay_fraction))
        replay_train = sources.load_replay(cfg.replay_dataset, cfg.replay_config, n_replay)

    # The forgetting probe comes from a DIFFERENT corpus than the replay data
    # and is never trained on. Drawing it from the replay pool -- even as
    # disjoint documents -- measures whether we learned the replay
    # distribution, not whether we kept general ability. See config.py.
    general_val = sources.load_general(cfg.forgetting_dataset, cfg.forgetting_config,
                                       cfg.forgetting_split, cfg.val_docs,
                                       tag="forgetting")

    train = domain_train + replay_train
    rng.shuffle(train)

    _write(train_p, train)
    _write(val_p, domain_val)
    _write(gen_val_p, general_val)

    card = {
        "stage": cfg.stage,
        "raw_documents": len(raw),
        "quality": qstats,
        "dedup": dstats.as_dict(),
        "domain_train": len(domain_train),
        "replay_train": len(replay_train),
        "replay_fraction_actual": round(len(replay_train) / max(len(train), 1), 3),
        "train_total": len(train),
        "domain_val": len(domain_val),
        "general_val": len(general_val),
        "replay_corpus": cfg.replay_dataset,
        "forgetting_corpus": cfg.forgetting_dataset,
        "train_chars": sum(len(d["text"]) for d in train),
        "approx_train_tokens": sum(len(d["text"]) for d in train) // 4,
    }
    (out_dir / "corpus_card.json").write_text(json.dumps(card, indent=2))

    t = Table(title=f"corpus [{cfg.stage}]", show_header=False)
    for k in ("raw_documents", "domain_train", "replay_train",
              "replay_fraction_actual", "train_total", "domain_val",
              "general_val", "approx_train_tokens"):
        t.add_row(k, f"{card[k]:,}" if isinstance(card[k], int) else str(card[k]))
    console.print(t)
    console.print(f"-> {out_dir}")
    return card


def load_split(stage: str, split: str) -> list[str]:
    p = paths.CORPUS / stage / f"{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run `geocpt corpus` first")
    return [json.loads(l)["text"] for l in p.open()]
