"""Turn labelled passages into train / valid / test / gold splits.

The two decisions in this file that most affect whether your eval numbers mean
anything:

1. **Group-wise splitting.** Geolex carries several reference summaries per
   unit, and they overlap heavily -- seven passages all describing the Aarde
   Shale Member. Splitting at random puts near-duplicates on both sides of the
   wall and inflates test scores badly. We split on `unit_id`, so every
   passage about a unit lands in exactly one split.

2. **A gold slice held out of everything.** The rule labels have a ceiling.
   `data/gold/gold.jsonl` is drawn from test and is meant to be corrected by
   hand before you trust any headline number.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .. import paths
from ..config import DataCfg
from ..schema import build_messages
from . import label as label_mod
from . import vocab as vocab_mod

console = Console()


def _norm_for_dedupe(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(cfg: DataCfg, force: bool = False) -> dict:
    paths.ensure()
    src = paths.INTERIM / "passages.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing -- run `geosft fetch` first")

    records = [json.loads(line) for line in src.open()]
    labeller = label_mod.Labeller.from_vocab(vocab_mod.load())

    # ---- label, filter, dedupe -------------------------------------------
    seen: set[str] = set()
    kept: list[dict] = []
    dropped_empty = dropped_dupe = 0
    for rec in records:
        digest = hashlib.sha1(_norm_for_dedupe(rec["passage"]).encode()).hexdigest()
        if digest in seen:
            dropped_dupe += 1
            continue
        seen.add(digest)
        target = labeller.label(rec)
        if label_mod.filled_fields(target) < cfg.min_filled_fields:
            dropped_empty += 1
            continue
        kept.append({
            "unit_id": rec["unit_id"],
            "unit_name": rec.get("unit_name"),
            "passage": rec["passage"],
            "target": json.loads(target.to_json()),
            "messages": build_messages(rec["passage"], target),
            "source_url": rec.get("source_url"),
            "ref_year": rec.get("ref_year"),
        })

    # ---- group-wise split -------------------------------------------------
    by_unit: dict[int, list[dict]] = defaultdict(list)
    for row in kept:
        by_unit[row["unit_id"]].append(row)
    unit_ids = sorted(by_unit)
    rng = random.Random(cfg.seed)
    rng.shuffle(unit_ids)

    n_units = len(unit_ids)
    n_test = int(n_units * cfg.test_frac)
    n_val = int(n_units * cfg.val_frac)
    test_u = set(unit_ids[:n_test])
    val_u = set(unit_ids[n_test:n_test + n_val])

    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for uid in unit_ids:
        key = "test" if uid in test_u else "valid" if uid in val_u else "train"
        splits[key].extend(by_unit[uid])
    for rows in splits.values():
        rng.shuffle(rows)

    # ---- write trainer-facing files --------------------------------------
    # mlx_lm and trl both accept {"messages": [...]} chat rows, so one file
    # feeds both backends and the two paths cannot diverge on data.
    for name, rows in splits.items():
        _write_jsonl(paths.PROCESSED / f"{name}.jsonl", [{"messages": r["messages"]} for r in rows])
        _write_jsonl(paths.PROCESSED / f"{name}.meta.jsonl", [
            {k: v for k, v in r.items() if k != "messages"} for r in rows
        ])

    # ---- gold slice, drawn from test and never trained on ----------------
    gold_rows = splits["test"][: cfg.gold_n]
    _write_jsonl(paths.GOLD / "gold.jsonl", [
        {**{k: v for k, v in r.items() if k != "messages"},
         "reviewed": False, "reviewer_notes": ""}
        for r in gold_rows
    ])

    stats = {
        "passages_in": len(records),
        "dropped_duplicate": dropped_dupe,
        "dropped_low_signal": dropped_empty,
        "kept": len(kept),
        "units": n_units,
        "train": len(splits["train"]),
        "valid": len(splits["valid"]),
        "test": len(splits["test"]),
        "gold": len(gold_rows),
        "audit": label_mod.agreement_report(records[:2000], labeller),
    }
    (paths.PROCESSED / "dataset_card.json").write_text(json.dumps(stats, indent=2))

    table = Table(title="dataset", show_header=False)
    for k in ("passages_in", "dropped_duplicate", "dropped_low_signal", "kept",
              "units", "train", "valid", "test", "gold"):
        table.add_row(k, str(stats[k]))
    console.print(table)
    console.print("[bold]labeller audit[/bold]", stats["audit"])
    return stats
