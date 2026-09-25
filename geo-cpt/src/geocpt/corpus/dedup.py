"""Near-duplicate removal with MinHash + LSH.

WHY THIS MATTERS FAR MORE FOR CPT THAN FOR SFT
----------------------------------------------
In SFT you show the model a few thousand labelled pairs and grade it on an
answer. A duplicate is a mild nuisance.

In CPT the model is predicting *every token* of raw text, repeatedly, with no
labels to anchor it. Duplicated text is therefore memorised close to verbatim,
and every duplicate token is budget you spent teaching the model to recite
something it already recites. Published pretraining work consistently finds
deduplication to be one of the highest-leverage data interventions available.

This corpus is unusually duplicate-heavy:

* ASUD files a note per reference per unit, and later notes often repeat
  earlier ones nearly word for word ("Geological Province: Sydney Basin.").
* eCat abstracts recur across editions and derived products of one report.

EXACT vs NEAR DUPLICATES
------------------------
Hashing the whole string only catches byte-identical text. Two abstracts
differing by a single year or author are different strings but the same
content. So we compare *sets of word shingles* by Jaccard similarity:

    J(A, B) = |A n B| / |A u B|

Computing that for every pair is O(n^2) -- 100k documents is 5 billion
comparisons. MinHash reduces each document to a small signature whose
collision probability equals the Jaccard similarity, and LSH buckets
signatures so only plausible pairs are ever compared. That turns O(n^2) into
roughly O(n).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH
from rich.console import Console
from rich.progress import Progress

console = Console()
_WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, k: int = 5) -> set[str]:
    """Overlapping word k-grams. Order-sensitive, so reshuffled text differs."""
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def signature(text: str, num_perm: int, k: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text, k):
        m.update(sh.encode("utf8"))
    return m


@dataclass
class DedupStats:
    n_in: int = 0
    exact_removed: int = 0
    near_removed: int = 0
    n_out: int = 0

    def as_dict(self) -> dict:
        return {"in": self.n_in, "exact_duplicates_removed": self.exact_removed,
                "near_duplicates_removed": self.near_removed, "out": self.n_out,
                "removed_pct": round(100 * (self.n_in - self.n_out) / max(self.n_in, 1), 1)}


def deduplicate(docs: list[dict], threshold: float = 0.8, num_perm: int = 128,
                k: int = 5) -> tuple[list[dict], DedupStats]:
    """Drop exact then near duplicates. First occurrence wins.

    `threshold` is Jaccard similarity: 0.8 means "80% of word 5-grams shared".
    Lower is more aggressive. Below ~0.6 you start removing genuinely distinct
    documents that merely share boilerplate.
    """
    stats = DedupStats(n_in=len(docs))

    # pass 1: exact, on normalised text -- cheap, catches the obvious
    seen: set[int] = set()
    staged: list[dict] = []
    for d in docs:
        h = hash(" ".join(_WORD.findall(d["text"].lower())))
        if h in seen:
            stats.exact_removed += 1
            continue
        seen.add(h)
        staged.append(d)

    # pass 2: near, via LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict] = []
    with Progress(*Progress.get_default_columns(), console=console) as prog:
        task = prog.add_task("dedup (minhash/lsh)", total=len(staged))
        for i, d in enumerate(staged):
            sig = signature(d["text"], num_perm, k)
            if lsh.query(sig):                 # a near-neighbour already kept
                stats.near_removed += 1
            else:
                lsh.insert(f"d{i}", sig)
                kept.append(d)
            prog.advance(task)

    stats.n_out = len(kept)
    console.print(f"[green]dedup[/green] {stats.as_dict()}")
    return kept, stats
