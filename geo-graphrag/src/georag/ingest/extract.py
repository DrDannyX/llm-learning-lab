"""Corpus -> knowledge graph, as plain Python data (no database yet).

Keeping extraction separate from loading means every modelling decision here
is unit-testable and inspectable (`data/interim/graph.json`) before Neo4j is
involved. It is the KG equivalent of `geosft inspect`.

WHERE EACH FACT COMES FROM
-------------------------
A knowledge graph is only as good as its extraction, and this one mixes two
sources of very different quality. Every edge records which one it came from.

  source    | how                                        | trust
  ----------|--------------------------------------------|-----------------------
  geolex    | USGS curated metadata: usages, ages,       | high -- hand-curated
            | states, provinces, reference lithologies   |
  text      | geo-sft's rule labeller run over passages  | medium -- the SFT lab
            | (relations, lithologies, minerals,         | found 83% of rule-labelled
            | thickness)                                 | rows had at least one error

Ages come ONLY from curated metadata. The labeller's chronostrat tags every
interval mentioned anywhere in a passage, including the ages of neighbouring
units, and "which units are Cretaceous?" is exactly the query that noise ruins.

THE THREE MODELLING DECISIONS THAT MATTER
-----------------------------------------
1. **Entity resolution.** Prose says "Eagle Ford Clay"; Geolex calls the unit
   "Eagle Ford". Names are reduced to a core ("eagle ford") before matching,
   and 289 names in this corpus belong to more than one unit (there are two
   Adas), so ties are broken by shared states. Names that match nothing still
   become Unit nodes (``geolex: false``) so the relation is not lost.
2. **Canonical direction.** "A underlies B" is stored as B-[:OVERLIES]->A.
   One relationship type per fact means one way to query it -- which matters
   when a small model is writing the query.
3. **A computed timescale.** Intervals carry numeric ages, so each is linked
   to the intervals that contain it (Virgilian -[:WITHIN]-> Pennsylvanian
   -[:WITHIN]-> Carboniferous ...). That turns "Cretaceous units" into a graph
   traversal that also finds units tagged only "Cenomanian". No amount of
   vector search can do that.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from geosft.data.label import LITH_AS_RANK, RANK_WORDS, Labeller
from geosft.data.match import Gazetteer
from geosft.data.states import STATES

#: Macrostrat's timescale has no "Precambrian" -- a term Geolex uses constantly.
EXTRA_INTERVALS = [
    {"name": "Precambrian", "rank": "supereon", "t_age": 538.8, "b_age": 4600.0},
]
INTERVAL_RANKS = ["age", "epoch", "period", "era", "eon", "supereon"]

STATE_NAMES = {code: name.title() for name, code in STATES.items() if name != "alaska peninsula"}

#: Canonicalisation of the labeller's relation kinds: (edge type, reverse?, props)
RELATION_MAP = {
    "overlies": ("OVERLIES", False, {}),
    "underlies": ("OVERLIES", True, {}),
    "unconformable_on": ("OVERLIES", False, {"unconformable": True}),
    "grades_into": ("GRADES_INTO", False, {}),
    "intertongues_with": ("INTERTONGUES_WITH", False, {}),
    "equivalent_to": ("EQUIVALENT_TO", False, {}),
}

_RANK_OR_LITH = set(RANK_WORDS) | LITH_AS_RANK | {"subgroup", "supergroup"}
_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)")


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------

def clean_name(s: str) -> str:
    """Strip Geolex markup: '/Sacfox subgroup [informal]' -> 'Sacfox subgroup'."""
    s = _BRACKETS.sub("", s).replace("/", " ")
    return re.sub(r"\s+", " ", s).strip(" .,;:*")


def core_name(s: str) -> str:
    """The part of a unit name that identifies it, lowercased.

    'Eagle Ford Clay' -> 'eagle ford'; 'Howard Limestone' -> 'howard';
    'Kansas City Group' -> 'kansas city'. Trailing rank and lithology words are
    how prose varies a name, so they are exactly what must not be matched on.
    """
    words = clean_name(s).split()
    while len(words) > 1 and words[-1].lower().rstrip("s") in _RANK_OR_LITH | {"serie"}:
        words.pop()
    return " ".join(words).lower()


def rank_of(phrase: str) -> str:
    """Rank from a usage segment's last word. Lithology-as-rank means Formation."""
    words = clean_name(phrase).split()
    if not words:
        return "Unknown"
    last = words[-1].lower().rstrip("s")
    if last in RANK_WORDS:
        return RANK_WORDS[last]
    if last in LITH_AS_RANK:
        return "Formation"
    return "Unknown"


def parse_usage(usage: str, unit_name: str) -> list[str] | None:
    """'Aarde Shale Member of Howard Limestone of Wabaunsee Group' ->
    ['Aarde Shale Member', 'Howard Limestone', 'Wabaunsee Group'].

    Geolex usage lists also carry free-text notes ('Misspelled Critizer in
    early reports.'); a usage only counts if it starts with the unit's name.
    """
    # "X of Y" and "X in Y" both mean X is part of Y; requiring a capital (or
    # Geolex's "/informal" slash) after the connective keeps "Member of the
    # upper part" style prose from splitting
    segs = [clean_name(s) for s in re.split(r"\s+(?:of|in)\s+(?=[A-Z/])", usage)]
    segs = [s for s in segs if s]
    if not segs or not segs[0].lower().startswith(unit_name.lower()):
        return None
    return segs


# --------------------------------------------------------------------------
# graph container
# --------------------------------------------------------------------------

@dataclass
class Graph:
    units: dict[str, dict] = field(default_factory=dict)
    passages: list[dict] = field(default_factory=list)
    intervals: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def edge(self, type_: str, src: str, dst: str, **props) -> None:
        self.edges.append({"type": type_, "src": src, "dst": dst, "props": props})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> Graph:
        return cls(**json.loads(path.read_text()))


class Resolver:
    """Name -> unit key. Geolex units first, placeholders for the rest."""

    def __init__(self, graph: Graph):
        self.g = graph
        self.index: dict[str, list[str]] = defaultdict(list)
        for key, u in graph.units.items():
            self.index[core_name(u["name"])].append(key)
        self.outcomes: Counter = Counter()

    def resolve(self, name: str, context_states: list[str] | None = None,
                rank: str = "Unknown") -> str:
        core = core_name(name)
        cands = [k for k in self.index.get(core, []) if self.g.units[k]["geolex"]]
        if len(cands) == 1:
            self.outcomes["resolved"] += 1
            return cands[0]
        if len(cands) > 1:
            ctx = set(context_states or [])
            scored = sorted(cands, key=lambda k: -len(ctx & set(self.g.units[k]["states"])))
            best = scored[0]
            clear = len(ctx & set(self.g.units[best]["states"])) > len(
                ctx & set(self.g.units[scored[1]]["states"]))
            self.outcomes["ambiguous_by_state" if clear else "ambiguous_guess"] += 1
            return best
        # unknown to Geolex (or outside this corpus): keep it as a placeholder
        key = f"name:{core}"
        if key not in self.g.units:
            self.g.units[key] = {
                "key": key, "unit_id": None, "name": clean_name(name), "full_name": clean_name(name),
                "rank": rank, "geolex": False, "states": [], "aliases": [], "url": None,
                "age_text": [], "thickness_min_m": None, "thickness_max_m": None,
            }
            self.index[core].append(key)
            self.outcomes["placeholder_created"] += 1
        else:
            self.outcomes["placeholder_reused"] += 1
        return key


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------

def load_records(passages_path: Path, limit_units: int | None, seed: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in passages_path.open()]
    if limit_units:
        ids = sorted({r["unit_id"] for r in rows})
        keep = set(random.Random(seed).sample(ids, min(limit_units, len(ids))))
        rows = [r for r in rows if r["unit_id"] in keep]
    return rows


def build_intervals(vocab: dict) -> tuple[list[dict], list[tuple[str, str]]]:
    """Every interval, plus WITHIN edges to the smallest-rank containers.

    An interval links to all intervals of the next populated higher rank that
    contain it. 'All' matters: Miocene sits within both Neogene and the
    (informal, still widely used) Tertiary.
    """
    ivs = {c["name"]: c for c in vocab["chronostrat"]} | {c["name"]: c for c in EXTRA_INTERVALS}
    ivs = {n: c for n, c in ivs.items() if c["rank"] in INTERVAL_RANKS}
    eps = 0.01
    within: list[tuple[str, str]] = []
    for c in ivs.values():
        r = INTERVAL_RANKS.index(c["rank"])
        for higher in INTERVAL_RANKS[r + 1:]:
            parents = [p["name"] for p in ivs.values() if p["rank"] == higher
                       and p["t_age"] - eps <= c["t_age"] and c["b_age"] <= p["b_age"] + eps]
            if parents:
                within += [(c["name"], p) for p in parents]
                break
    return list(ivs.values()), within


def build(passages_path: Path, vocab_path: Path, limit_units: int | None = None,
          seed: int = 0) -> Graph:
    vocab = json.loads(vocab_path.read_text())
    labeller = Labeller.from_vocab(vocab)
    intervals, within = build_intervals(vocab)
    chrono = Gazetteer({i["name"].lower(): i["name"] for i in intervals})
    g = Graph(intervals=intervals)
    for child, parent in within:
        g.edge("WITHIN", f"interval:{child}", f"interval:{parent}")

    rows = load_records(passages_path, limit_units, seed)
    by_unit: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)

    # 1. Geolex units: identity and curated metadata ----------------------
    usages_of: dict[str, list[list[str]]] = {}
    for uid, recs in by_unit.items():
        r0 = recs[0]
        key = f"geolex:{uid}"
        chains = [c for u in r0["usages"] if (c := parse_usage(u, r0["unit_name"]))]
        usages_of[key] = chains
        firsts = list(dict.fromkeys(c[0] for c in chains))
        g.units[key] = {
            "key": key, "unit_id": uid, "name": r0["unit_name"],
            "full_name": firsts[0] if firsts else r0["unit_name"],
            "rank": rank_of(firsts[0]) if firsts else "Unknown",
            "geolex": True, "states": sorted(r0["states"]), "aliases": firsts,
            "url": r0["source_url"], "age_text": r0["age_description"],
            "thickness_min_m": None, "thickness_max_m": None,
        }
        for code in r0["states"]:
            g.edge("OCCURS_IN", key, f"state:{code}", source="geolex")
        for prov in r0["ref_province"]:
            g.edge("IN_PROVINCE", key, f"province:{prov}", source="geolex")
        for lith in r0["ref_lithology"]:
            g.edge("HAS_LITHOLOGY", key, f"lithology:{lith.lower()}", source="geolex")
        ages = {a for text in r0["age_description"] for a in chrono.values(text)}
        for a in sorted(ages):
            g.edge("HAS_AGE", key, f"interval:{a}", source="geolex")

    resolver = Resolver(g)

    # 2. hierarchy from usages: Member -> Formation -> Group ---------------
    for key, chains in usages_of.items():
        states = g.units[key]["states"]
        for chain in chains:
            prev = key
            for seg in chain[1:]:
                parent = resolver.resolve(seg, states, rank=rank_of(seg))
                if parent != prev:
                    g.edge("PART_OF", prev, parent, source="geolex", usage=" of ".join(chain))
                prev = parent

    # 3. passages, and what the rule labeller finds in them ---------------
    text_liths: dict[str, Counter] = defaultdict(Counter)
    text_mins: dict[str, Counter] = defaultdict(Counter)
    subject_confirmed = 0
    for uid, recs in by_unit.items():
        key = f"geolex:{uid}"
        u = g.units[key]
        for n, r in enumerate(recs):
            pid = f"{uid}:{n}"
            g.passages.append({
                "id": pid, "unit_key": key, "unit_name": u["full_name"], "text": r["passage"],
                "year": r.get("ref_year"), "publication": r.get("ref_publication"),
                "url": r["source_url"],
            })
            g.edge("DESCRIBES", f"passage:{pid}", key)
            lab = labeller.label(r)
            if lab.unit_name is None:
                # the labeller could not confirm this passage is ABOUT this
                # unit; its relations might belong to a neighbour
                continue
            subject_confirmed += 1
            for lith in lab.lithologies:
                text_liths[key][lith.lower()] += 1
            for m in lab.minerals:
                text_mins[key][m] += 1
            if lab.thickness:
                lo, hi = lab.thickness.min_m, lab.thickness.max_m
                if lo is not None:
                    u["thickness_min_m"] = min(lo, u["thickness_min_m"] or lo)
                if hi is not None:
                    u["thickness_max_m"] = max(hi, u["thickness_max_m"] or hi)
            for rel in lab.relations:
                other = resolver.resolve(rel.unit, u["states"], rank=rank_of(rel.unit))
                if other == key:
                    continue
                type_, reverse, props = RELATION_MAP[rel.kind]
                src, dst = (other, key) if reverse else (key, other)
                g.edge(type_, src, dst, source="text", passage=pid, **props)
                g.edge("MENTIONS", f"passage:{pid}", other)

    for key, c in text_liths.items():
        for lith in c:
            g.edge("HAS_LITHOLOGY", key, f"lithology:{lith}", source="text")
    for key, c in text_mins.items():
        for m in c:
            g.edge("HAS_MINERAL", key, f"mineral:{m}", source="text")

    g.edges = _merge_edges(g.edges)
    types = Counter(e["type"] for e in g.edges)
    g.stats = {
        "passages": len(g.passages),
        "geolex_units": sum(u["geolex"] for u in g.units.values()),
        "placeholder_units": sum(not u["geolex"] for u in g.units.values()),
        "intervals": len(g.intervals),
        "edges": dict(sorted(types.items())),
        "passages_subject_confirmed": subject_confirmed,
        "resolution": dict(resolver.outcomes),
    }
    return g


def _merge_edges(edges: list[dict]) -> list[dict]:
    """One edge per (type, src, dst). Evidence accumulates instead of duplicating:
    `sources` says who asserted it, `passages` says where the text said so."""
    merged: dict[tuple, dict] = {}
    for e in edges:
        k = (e["type"], e["src"], e["dst"])
        p = dict(e["props"])
        m = merged.setdefault(k, {"type": e["type"], "src": e["src"], "dst": e["dst"], "props": {}})
        mp = m["props"]
        if "source" in p:
            mp["sources"] = sorted(set(mp.get("sources", [])) | {p.pop("source")})
        if "passage" in p:
            mp["passages"] = sorted(set(mp.get("passages", [])) | {p.pop("passage")})
        if "usage" in p:
            mp.setdefault("usage", p.pop("usage"))
        if p.pop("unconformable", False):
            mp["unconformable"] = True
        mp.update(p)
    return list(merged.values())
