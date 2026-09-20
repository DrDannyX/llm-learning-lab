"""Heuristic quality filtering.

Pretraining corpora are only as good as their worst documents, and unlike SFT
there is no label to tell you a document is junk. These filters are the
classic cheap heuristics (Gopher/C4 lineage), adapted to scientific prose:

* **too short** -- a two-line abstract stub teaches nothing
* **low alphabetic ratio** -- tables, coordinate dumps, reference lists
* **low unique-word ratio** -- boilerplate and repeated headers
* **excessive digits** -- assay tables and data appendices masquerading as text
* **no sentence structure** -- keyword lists, index fragments

Each rejection is COUNTED AND REPORTED. A filter that silently eats 40% of
your corpus is worse than no filter, because you will never know. Always read
the rejection histogram and sample some rejects before trusting it.
"""
from __future__ import annotations

import re
from collections import Counter

from rich.console import Console
from rich.table import Table

console = Console()
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']*")


def reject_reason(text: str, min_chars: int, max_chars: int) -> str | None:
    """Return a reason string if the document should be dropped, else None."""
    n = len(text)
    if n < min_chars:
        return "too_short"
    if n > max_chars:
        return "too_long"

    alpha = sum(c.isalpha() or c.isspace() for c in text) / n
    if alpha < 0.70:
        return "low_alpha_ratio"

    digits = sum(c.isdigit() for c in text) / n
    if digits > 0.20:
        return "digit_heavy"

    words = _WORD.findall(text)
    if len(words) < 50:
        return "too_few_words"

    if len(set(w.lower() for w in words)) / len(words) < 0.25:
        return "repetitive"

    if text.count(".") < 3:
        return "no_sentences"

    return None


def filter_docs(docs: list[dict], min_chars: int, max_chars: int) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    rejects: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for d in docs:
        why = reject_reason(d["text"], min_chars, max_chars)
        if why:
            rejects[why] += 1
            samples.setdefault(why, d["text"][:160])
        else:
            kept.append(d)

    t = Table(title="quality filter rejections")
    t.add_column("reason"); t.add_column("n", justify="right"); t.add_column("example")
    for reason, n in rejects.most_common():
        t.add_row(reason, str(n), samples.get(reason, "")[:70] + "...")
    if rejects:
        console.print(t)
    console.print(f"[green]quality[/green] kept {len(kept)}/{len(docs)} "
                  f"({100*len(kept)/max(len(docs),1):.1f}%)")
    return kept, {"kept": len(kept), "in": len(docs), "rejects": dict(rejects)}
