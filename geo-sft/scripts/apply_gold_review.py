"""Apply hand-review corrections to the gold set.

Corrections live in data/gold/corrections.json as {index: {field: value}}.
Only the named fields change; everything else is kept from the rule label.
Each reviewed row records WHO reviewed it, because that materially changes
how much the resulting numbers are worth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GOLD = Path("data/gold/gold.jsonl")
CORR = Path("data/gold/corrections.json")


def main(reviewer: str) -> None:
    rows = [json.loads(l) for l in GOLD.open()]
    corrections = json.loads(CORR.read_text()) if CORR.exists() else {}

    changed = 0
    for idx_s, patch in corrections.items():
        i = int(idx_s)
        row = rows[i]
        notes = patch.pop("_note", "")
        flag = patch.pop("_needs_expert", False)
        before = json.dumps(row["target"], sort_keys=True)
        row["target"].update(patch)
        row["reviewed"] = True
        row["reviewed_by"] = reviewer
        row["reviewer_notes"] = notes
        if flag:
            row["needs_expert_review"] = True
        if json.dumps(row["target"], sort_keys=True) != before:
            changed += 1

    with GOLD.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_rev = sum(1 for r in rows if r.get("reviewed"))
    n_exp = sum(1 for r in rows if r.get("needs_expert_review"))
    print(f"reviewed {n_rev}/{len(rows)} rows; {changed} had label corrections; "
          f"{n_exp} flagged for expert review")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "unknown")
