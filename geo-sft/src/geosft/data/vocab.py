"""Domain gazetteers: rock, time and mineral terms from Macrostrat (CC-BY 4.0),
unit names from ASUD (CC BY 4.0).

These vocabularies are load-bearing for the whole project:

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
#: Martian periods and would be nonsense labels on an Australian lexicon corpus,
#: so they must not enter the gazetteer. Matched against the timescale *name*.
NON_TERRESTRIAL = ("martian", "lunar", "mars", "moon")

#: Intervals ASUD uses that Macrostrat lacks: the informal Precambrian, and the
#: ICS's still-unnamed Cambrian series and stages (common in Australian
#: Cambrian stratigraphy, which helped define them).
EXTRA_INTERVALS = [
    {"name": "Precambrian", "rank": "supereon", "t_age": 538.8, "b_age": 4600.0},
    {"name": "Cambrian Series 2", "rank": "epoch", "t_age": 506.5, "b_age": 521.0},
    {"name": "Cambrian Stage 2", "rank": "age", "t_age": 521.0, "b_age": 529.0},
    {"name": "Cambrian Stage 3", "rank": "age", "t_age": 514.5, "b_age": 521.0},
    {"name": "Cambrian Stage 4", "rank": "age", "t_age": 506.5, "b_age": 514.5},
]


def chrono_aliases(names: list[str]) -> dict[str, str]:
    """Surface spellings Australian prose uses -> the canonical ICS name.

    Australian (and older ICS) usage writes Palaeozoic and Archaean, and names
    chronostratigraphic series ("Lower Devonian") where Macrostrat has the
    geochronologic epoch ("Early Devonian"). Canonicalising all of them to one
    name is what keeps the label space closed.
    """
    known = set(names)
    out: dict[str, str] = {}
    for n in names:
        brit = n.replace("Paleo", "Palaeo").replace("paleo", "palaeo").replace("rchean", "rchaean")
        if brit != n:
            out[brit.lower()] = n
        for series, epoch in (("Lower", "Early"), ("Upper", "Late")):
            if n.startswith(epoch + " "):
                out[(series + n[len(epoch):]).lower()] = n
    return {k: v for k, v in out.items() if k not in {x.lower() for x in known}}


def _unwrap(payload: dict) -> list[dict]:
    return payload["success"]["data"]


def build(force: bool = False) -> dict[str, list]:
    """Download and normalise all gazetteers into data/interim/vocab.json.

    Needs the ASUD snapshot (`asud.fetch()`) for the unit names.
    """
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
    have = {c["name"] for c in chrono}
    chrono += [c for c in EXTRA_INTERVALS if c["name"] not in have]
    chrono.sort(key=lambda d: d["name"])

    # Short mineral names ("ice", "gold") collide with ordinary English and with
    # lithology terms; require 5+ chars so the gazetteer stays high-precision.
    mineral_names = sorted(
        {d["mineral"] for d in minerals if d.get("mineral") and len(d["mineral"]) >= 5}
    )

    # Unit names come from ASUD itself: the relation rules only accept a
    # captured name whose first word starts a real Australian unit name.
    from . import asud
    strat_names = sorted({u.name for u in asud.load_units().values() if len(u.name) >= 4})

    vocab = {
        "lithologies": lith_names,
        "chronostrat": chrono,
        "chronostrat_aliases": chrono_aliases([c["name"] for c in chrono]),
        "minerals": mineral_names,
        "strat_names": strat_names,
        "_license": "Macrostrat API, CC-BY 4.0; unit names from ASUD, CC BY 4.0",
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
