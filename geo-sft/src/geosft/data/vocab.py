"""Domain gazetteers, downloaded from Macrostrat (CC-BY 4.0).

These four vocabularies are load-bearing for the whole project:

* they are the **closed label space** the extraction schema normalises to,
* they seed the **rule-based labeller**,
* they are the candidate pool for the **geoscience tokenizer extension**.
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from .. import paths
from .http import CachedClient

console = Console()
API = "https://macrostrat.org/api"

#: Chronostratigraphic ranks we care about. Zones/chrons/subchrons are far too
#: fine-grained to appear in lexicon prose and would only add false positives.
CHRONO_RANKS = {"eon", "era", "period", "epoch", "age"}

#: Macrostrat carries planetary timescales too. "Amazonian" and "Noachian" are
#: Martian periods and would be nonsense labels on a USGS lexicon corpus, so
#: they must not enter the gazetteer. Matched against the timescale *name*.
NON_TERRESTRIAL = ("martian", "lunar", "mars", "moon")


def _unwrap(payload: dict) -> list[dict]:
    return payload["success"]["data"]


def build(force: bool = False) -> dict[str, list]:
    """Download and normalise all gazetteers into data/interim/vocab.json."""
    paths.ensure()
    out = paths.INTERIM / "vocab.json"
    if out.exists() and not force:
        console.print(f"[dim]vocab cached -> {out}[/dim]")
        return json.loads(out.read_text())

    client = CachedClient(paths.RAW / "http-cache")
    try:
        liths = _unwrap(client.get_json(f"{API}/defs/lithologies", {"all": ""}))
        intervals = _unwrap(client.get_json(f"{API}/defs/intervals", {"all": ""}))
        minerals = _unwrap(client.get_json(f"{API}/defs/minerals", {"all": ""}))
        strats = _unwrap(
            client.get_json(f"{API}/defs/strat_names", {"all": "", "response": "short"})
        )
    finally:
        client.close()

    lith_names = sorted({d["name"].lower() for d in liths if d.get("name")})

    chrono = []
    for d in intervals:
        if d.get("int_type") not in CHRONO_RANKS:
            continue
        scales = " ".join(
            str((ts or {}).get("name") or "") for ts in (d.get("timescales") or [])
        ).lower()
        if any(bad in scales for bad in NON_TERRESTRIAL):
            continue
        chrono.append({"name": d["name"], "rank": d["int_type"],
                       "t_age": d.get("t_age"), "b_age": d.get("b_age")})
    chrono.sort(key=lambda d: d["name"])

    # Short mineral names ("ice", "gold") collide with ordinary English and with
    # lithology terms; require 5+ chars so the gazetteer stays high-precision.
    mineral_names = sorted(
        {d["mineral"] for d in minerals if d.get("mineral") and len(d["mineral"]) >= 5}
    )

    strat_names = sorted({
        d["strat_name"] for d in strats if d.get("strat_name") and len(d["strat_name"]) >= 4
    })

    vocab = {
        "lithologies": lith_names,
        "chronostrat": chrono,
        "minerals": mineral_names,
        "strat_names": strat_names,
        "_license": "Macrostrat API, CC-BY 4.0",
    }
    out.write_text(json.dumps(vocab))
    console.print(
        f"[green]vocab[/green] liths={len(lith_names)} chrono={len(chrono)} "
        f"minerals={len(mineral_names)} strat_names={len(strat_names)} -> {out}"
    )
    return vocab


def load() -> dict[str, list]:
    p = paths.INTERIM / "vocab.json"
    if not p.exists():
        return build()
    return json.loads(p.read_text())
