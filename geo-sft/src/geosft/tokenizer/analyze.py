"""Does the base tokenizer actually struggle with geoscience text?

Fertility (tokens per whitespace word) is the metric. Lower is better: it
means more of the model's context and compute is spent on meaning rather than
on re-assembling words from fragments.

The comparison that matters is not "geo tokenizer beats base tokenizer on geo
text" -- that is guaranteed and tells you nothing. It is:

  * how badly the BASE tokenizer fragments domain terms specifically, and
  * how much the domain tokenizer LOSES on general English.

The second number is the one people forget, and it is why replacing a
tokenizer wholesale is a bad trade.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from .. import paths
from ..data import vocab as vocab_mod

console = Console()

WORDS = re.compile(r"\S+")

#: A small general-English control set, so we can see what a domain tokenizer
#: costs on ordinary text.
GENERAL_TEXT = [
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
    "She argued that the committee should reconsider its position before the vote.",
    "Installing the package requires a recent version of the runtime and a build tool.",
    "He bought milk, eggs, and a loaf of bread on the way home from work.",
    "The economic outlook remains uncertain despite recent improvements in employment.",
]


def fertility(encode: Callable[[str], list], texts: list[str]) -> float:
    n_tok = sum(len(encode(t)) for t in texts)
    n_word = sum(len(WORDS.findall(t)) for t in texts)
    return n_tok / max(n_word, 1)


def report(geo_tokenizer_dir: Path, base_model: str, n_texts: int = 400) -> dict:
    from transformers import AutoTokenizer

    from .train import load as load_geo

    meta = paths.PROCESSED / "test.meta.jsonl"
    texts = [json.loads(l)["passage"] for l in meta.open()][:n_texts]

    geo = load_geo(geo_tokenizer_dir)
    base = AutoTokenizer.from_pretrained(base_model)

    geo_enc = lambda t: geo.encode(t).ids            # noqa: E731
    base_enc = lambda t: base.encode(t, add_special_tokens=False)  # noqa: E731

    res = {
        "base_model": base_model,
        "geo_vocab": geo.get_vocab_size(),
        "base_vocab": base.vocab_size,
        "fertility_geo_text": {
            "base": round(fertility(base_enc, texts), 3),
            "geo": round(fertility(geo_enc, texts), 3),
        },
        "fertility_general_text": {
            "base": round(fertility(base_enc, GENERAL_TEXT), 3),
            "geo": round(fertility(geo_enc, GENERAL_TEXT), 3),
        },
    }

    # Which domain terms does the base tokenizer shred the worst?
    v = vocab_mod.load()
    terms = (
        [t for t in v["lithologies"]]
        + [c["name"] for c in v["chronostrat"]]
        + [m for m in v["minerals"][:1500]]
    )
    # Raw fragmentation alone surfaces junk: "Clino-ferro-ferri-fluoro-
    # holmquistite" costs 16 tokens and appears in the corpus zero times.
    # Weighting by how often a term actually occurs is what makes this list
    # actionable -- and it is the same ranking extend.py uses.
    from collections import Counter
    train_meta = paths.PROCESSED / "train.meta.jsonl"
    freq: Counter[str] = Counter()
    if train_meta.exists():
        for line in train_meta.open():
            freq.update(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}",
                                                      json.loads(line)["passage"]))

    scored = []
    for term in terms:
        n_base = len(base_enc(" " + term))
        if n_base >= 3:
            scored.append((n_base, term, freq.get(term.lower(), 0)))
    scored.sort(reverse=True)
    res["n_terms_ge3_tokens"] = len(scored)
    res["n_terms_checked"] = len(terms)
    res["worst_fragmented_raw"] = [
        {"term": t, "base_tokens": n, "corpus_freq": f} for n, t, f in scored[:15]
    ]
    by_savings = sorted(scored, key=lambda x: -(x[2] * (x[0] - 1)))
    res["worst_fragmented_weighted"] = [
        {"term": t, "base_tokens": n, "corpus_freq": f, "tokens_wasted": f * (n - 1)}
        for n, t, f in by_savings[:25] if f > 0
    ]

    table = Table(title="tokenizer fertility (tokens per word, lower is better)")
    table.add_column("corpus"); table.add_column("base"); table.add_column("geo")
    table.add_row("geoscience passages",
                  str(res["fertility_geo_text"]["base"]), str(res["fertility_geo_text"]["geo"]))
    table.add_row("general English",
                  str(res["fertility_general_text"]["base"]), str(res["fertility_general_text"]["geo"]))
    console.print(table)
    console.print(
        f"[bold]{len(scored)}/{len(terms)}[/bold] domain terms cost the base tokenizer "
        f"3+ tokens."
    )
    top = res["worst_fragmented_weighted"][:8]
    if top:
        console.print(
            "Worst offenders that actually occur in the corpus: "
            + ", ".join(f"{d['term']}({d['base_tokens']}t x{d['corpus_freq']})" for d in top)
        )
        console.print(
            f"[dim]Ranked by tokens wasted = frequency x (pieces - 1). Raw "
            f"fragmentation alone surfaces unused 16-token mineral names.[/dim]"
        )

    out = paths.ARTIFACTS / "tokenizer_report.json"
    out.write_text(json.dumps(res, indent=2))
    console.print(f"-> {out}")
    return res
