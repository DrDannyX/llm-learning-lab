"""Entity linking: find the units, intervals, states and lithologies a
question names, and map them to graph keys.

Both graph-aware systems depend on this step, and it is where they most often
fail. Text-to-Cypher with a 12B model cannot be trusted to guess that "the
Eagle Ford Shale" is stored as `{name: 'Eagle Ford'}`, and it certainly cannot
choose between the two units called Ada. So the names are resolved here,
deterministically, and handed to the model as exact keys.

Matching is longest-match over word n-grams, and a unit match must start with
a capital letter in the question -- 289 units have names like "Big", "Box" or
"Alum" that are also ordinary words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from geosft.data.states import STATES

from .. import db
from ..config import Neo4jConfig
from ..ingest.extract import _RANK_OR_LITH, core_name

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
MAX_NGRAM = 6
_NAME_TAIL = _RANK_OR_LITH | {"serie"}


@dataclass
class Linked:
    units: list[dict] = field(default_factory=list)
    intervals: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)      # postal codes
    lithologies: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = []
        for u in self.units:
            where = ", ".join(u["states"]) or "?"
            lines.append(f"- Unit {u['full_name']!r} ({u['rank']}; states {where}) -> key '{u['key']}'")
        if self.intervals:
            lines.append(f"- Intervals: {', '.join(repr(i) for i in self.intervals)}")
        if self.states:
            lines.append(f"- States (codes): {', '.join(repr(s) for s in self.states)}")
        if self.lithologies:
            lines.append(f"- Lithologies: {', '.join(repr(x) for x in self.lithologies)}")
        return "\n".join(lines) or "- (no known entities found in the question)"


class Linker:
    def __init__(self, cfg: Neo4jConfig):
        rows = db.read(cfg, """
            MATCH (u:Unit)
            RETURN u.key AS key, u.name AS name, u.full_name AS full_name, u.rank AS rank,
                   u.aliases AS aliases, u.geolex AS geolex,
                   [(u)-[:OCCURS_IN]->(s) | s.code] AS states
        """)
        self.units = {r["key"]: r for r in rows}
        self.surface: dict[str, set[str]] = {}
        for r in rows:
            names = {r["name"], r["full_name"], *(r["aliases"] or [])}
            for n in names:
                for form in {n.lower(), core_name(n)}:
                    if len(form) >= 3:
                        self.surface.setdefault(form, set()).add(r["key"])
        self.intervals = {r["n"].lower(): r["n"] for r in db.read(cfg, "MATCH (i:Interval) RETURN i.name AS n")}
        self.lithologies = {r["n"]: r["n"] for r in db.read(cfg, "MATCH (l:Lithology) RETURN l.name AS n")}
        self.states = {k: v for k, v in STATES.items() if k != "alaska peninsula"}

    def link(self, question: str) -> Linked:
        toks = [(m.group(0), m.start()) for m in _WORD.finditer(question)]
        low = [t[0].lower() for t in toks]
        out = Linked()
        unit_hits: list[set[str]] = []
        i = 0
        while i < len(toks):
            for span in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
                phrase = " ".join(low[i:i + span])
                if phrase in self.states:
                    out.states.append(self.states[phrase])
                elif phrase in self.intervals:
                    out.intervals.append(self.intervals[phrase])
                elif phrase in self.surface and toks[i][0][0].isupper():
                    unit_hits.append(self.surface[phrase])
                    # "Eagle Ford Shale": the trailing rank/lithology word is
                    # part of the name, not a lithology filter
                    while i + span < len(toks) and low[i + span].rstrip("s") in _NAME_TAIL:
                        span += 1
                elif span == 1 and (phrase in self.lithologies or phrase.rstrip("s") in self.lithologies):
                    out.lithologies.append(self.lithologies.get(phrase) or phrase.rstrip("s"))
                else:
                    continue
                i += span
                break
            else:
                i += 1
        # bare postal codes: "units in TX"
        codes = set(self.states.values())
        out.states += [t for t, _ in toks if t in codes and t not in out.states]

        for keys in unit_hits:
            cands = [self.units[k] for k in keys]
            # prefer curated Geolex units over placeholders, then homonyms that
            # share a state with the question
            if any(c["geolex"] for c in cands):
                cands = [c for c in cands if c["geolex"]]
            if out.states and len(cands) > 1:
                near = [c for c in cands if set(c["states"]) & set(out.states)]
                cands = near or cands
            out.units += [c for c in cands[:3] if c not in out.units]
        out.intervals = list(dict.fromkeys(out.intervals))
        out.states = list(dict.fromkeys(out.states))
        out.lithologies = list(dict.fromkeys(out.lithologies))
        return out


@lru_cache(maxsize=2)
def _linker(uri: str, user: str, password: str, database: str) -> Linker:
    return Linker(Neo4jConfig(uri=uri, user=user, password=password, database=database))


def linker(cfg: Neo4jConfig) -> Linker:
    return _linker(cfg.uri, cfg.user, cfg.password, cfg.database)
