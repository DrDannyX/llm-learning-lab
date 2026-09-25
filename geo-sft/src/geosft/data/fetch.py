"""Corpus acquisition: Geoscience Australia's stratigraphic lexicon (ASUD).

Why this source, for a fine-tuning lab:

* It is **open** (CC BY 4.0, attribution required), and it is the national
  authority on Australian stratigraphic names, maintained by Geoscience
  Australia.
* The passages are **real geologist prose**: definition cards written by the
  units' authors, and thousands of short notes on how each published reference
  uses a unit. The notes are abbreviated, telegraphic and full of house style
  ("Conformably overlain by ...", "qtz-rich", "Sample GSWA 178851 yielded ...")
  -- exactly the kind of text a general-purpose instruct model handles badly.
* Each passage arrives with **curated metadata** (rank, ages, states,
  thickness and, unusually, the unit's stratigraphic relations) that we use to
  audit weak supervision instead of hallucinating labels with a teacher model.

Two kinds of passage come out of ASUD (see asud.py for the tables):

  definition  a unit's definition card: lithology, relationships and
              boundaries, thickness, extent, age reasons, name source ...
  article     one reference's comment on the unit ("Briefly described, p45:
              Conformably overlies Minnie Point Formation ...")

Every passage is rendered as a lexicon ENTRY -- "<Unit Name>. <text>" --
because that is how ASUD presents it: the note is filed under the unit's name,
and without the heading "Overlain by Cygnet Coal Measures" has no subject.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console

from .. import paths
from . import asud

console = Console()

#: Definition-card sections that are administration, not geology.
ADMIN_SECTIONS = {
    "Proposer", "Defn author", "Defn approved by", "Reserved? Yes/No",
    "Proposed publication", "Defn Reference", "References", "Name first published by",
    "Reference", "Approved by", "Date approved", "Reserved by", "Reserved date",
}

#: ASUD relation wording -> schema relation kind. Used for AUDIT ONLY.
CURATED_KIND = {
    "overlies": "overlies", "underlies": "underlies", "is equivalent to": "equivalent_to",
    "grades into": "grades_into", "intrudes": "intrudes", "is intruded by": "intruded_by",
    "is interbedded with": "intertongues_with", "intermingles with": "intertongues_with",
}

#: Article comments carry page-location boilerplate that says nothing about the
#: rock: "Location in text includes p325 Fig.3, p328.", "See also p66 Fig.1.",
#: "See 100k_geologyp_lut.csv." Stripping them raises signal-to-noise without
#: removing any fact we ask the model to extract.
_SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z])")
_BOILER = re.compile(r"^(Location in text|See also (pp?\.?\s?\d|Fig|Table|Plate)|Page\s)|\.csv\b", re.I)
_WS = re.compile(r"\s+")


def clean_passage(text: str) -> str:
    sents = [s for s in _SENT.split(_WS.sub(" ", text or "").strip()) if not _BOILER.search(s)]
    return " ".join(sents).strip()


def _curated(unit: asud.Unit, related: dict[int, list[asud.Related]],
             units: dict[int, asud.Unit]) -> dict:
    """The unit's curated facts, attached to every passage as audit anchors.

    NEVER shown to the model and never written into a label (see label.py).
    """
    rels = []
    for r in related.get(unit.stratno, []):
        kind = CURATED_KIND.get(r.relation)
        other = units.get(r.dst)
        if kind and other:
            if kind == "overlies" and "unconformity" in r.contact:
                kind = "unconformable_on"
            rels.append({"kind": kind, "unit": other.name})
    return {
        "asud_rank": unit.rank,
        "age_names": [a for a in (unit.base_age, unit.top_age) if a],
        "states": unit.states,
        "curated_relations": rels,
        "thickness_min_m": unit.thickness_min_m,
        "thickness_max_m": unit.thickness_max_m,
        "provinces": unit.provinces,
        "source_url": unit.url,
    }


def definition_passages(units: dict[int, asud.Unit], max_chars: int) -> list[dict]:
    """One passage per definition card, split on section boundaries if long."""
    sections: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for r in asud.read_table("definition"):
        sn = asud._int(r.get("Stratno"))
        cat, body = r.get("Category", ""), clean_passage(r.get("Contents", ""))
        if sn in units and body and cat not in ADMIN_SECTIONS:
            sections[sn].append((cat, body))

    out: list[dict] = []
    for sn, secs in sections.items():
        head = f"{units[sn].name}. "
        chunk = head
        for cat, body in secs:
            piece = f"{cat}: {body} "
            if len(head) + len(piece) > max_chars:
                continue  # one oversized section: drop it rather than cut mid-sentence
            if len(chunk) + len(piece) > max_chars:
                out.append({"unit_id": sn, "kind": "definition", "passage": chunk.strip()})
                chunk = head
            chunk += piece
        if chunk != head:
            out.append({"unit_id": sn, "kind": "definition", "passage": chunk.strip()})
    return out


def article_passages(units: dict[int, asud.Unit]) -> list[dict]:
    """One passage per reference comment.

    A comment filed under several units ("Of the Parmeener Supergroup.") is not
    about any one of them, and would put identical text on both sides of the
    group split -- so any comment body seen under two different units is
    dropped entirely.
    """
    refs = asud.load_references()
    rows = []
    for r in asud.read_table("articles"):
        sn = asud._int(r.get("Stratno"))
        body = clean_passage(r.get("Reference Comments", ""))
        if sn in units and body:
            rows.append((sn, body, r))
    owners: dict[str, set[int]] = defaultdict(set)
    for sn, body, _ in rows:
        owners[body.lower()].add(sn)

    out, seen = [], set()
    for sn, body, r in rows:
        key = (sn, body.lower())
        if len(owners[body.lower()]) > 1 or key in seen:
            continue
        seen.add(key)
        ref = refs.get(r.get("Reference Id", ""), {})
        out.append({
            "unit_id": sn, "kind": "article",
            "passage": f"{units[sn].name}. {body}",
            "usage": r.get("Usage") or None,
            "ref_age_names": [a for a in (r.get("Maximum Age Name"), r.get("Minimum Age Name")) if a],
            "ref_year": asud._int(ref.get("Year")),
            "ref_publication": ref.get("Title") or None,
        })
    return out


def run(max_units: int | None, min_chars: int, max_chars: int,
        max_per_unit: int | None = None, seed: int = 17, force: bool = False) -> Path:
    paths.ensure()
    out = paths.INTERIM / "passages.jsonl"
    if out.exists() and not force:
        n = sum(1 for _ in out.open())
        console.print(f"[dim]passages cached ({n}) -> {out}[/dim]")
        return out

    asud.fetch(force=force)
    units = asud.load_units()
    related: dict[int, list[asud.Related]] = defaultdict(list)
    for r in asud.load_related():
        related[r.src].append(r)

    raw = definition_passages(units, max_chars) + article_passages(units)
    raw = [p for p in raw if min_chars <= len(p["passage"]) <= max_chars]

    by_unit: dict[int, list[dict]] = defaultdict(list)
    for p in raw:
        by_unit[p["unit_id"]].append(p)
    rng = random.Random(seed)
    unit_ids = sorted(by_unit)
    if max_units and len(unit_ids) > max_units:
        unit_ids = sorted(rng.sample(unit_ids, max_units))

    passages: list[dict] = []
    for sn in unit_ids:
        ps = by_unit[sn]
        if max_per_unit and len(ps) > max_per_unit:
            # A few famous units carry 100+ reference notes. Capping them keeps
            # one unit from dominating the loss; definitions are kept first.
            defs = [p for p in ps if p["kind"] == "definition"]
            arts = [p for p in ps if p["kind"] != "definition"]
            rng.shuffle(arts)
            ps = (defs + arts)[:max_per_unit]
        u = units[sn]
        for p in ps:
            age_names = p.pop("ref_age_names", [])
            cur = _curated(u, related, units)
            cur["age_names"] = cur["age_names"] + age_names
            passages.append({"unit_id": sn, "unit_name": u.name, **p, **cur})

    with out.open("w") as fh:
        for p in passages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    kinds = Counter(p["kind"] for p in passages)
    console.print(
        f"[green]passages[/green] {len(passages)} from {len({p['unit_id'] for p in passages})} "
        f"units ({dict(kinds)}) -> {out}"
    )
    return out
