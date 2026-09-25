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
  asud      | Geoscience Australia's curated tables:     | high -- curated by the
            | rank, parent, ages, states, provinces,     | Australian Stratigraphy
            | lithology, thickness AND relations         | Commission
  text      | geo-sft's rule labeller run over passages  | medium -- the SFT lab's
            | (relations, lithologies, minerals)         | gold review corrected
            |                                            | 57% of rule-labelled rows

ASUD curates stratigraphic relations (overlies, intrudes, equivalent to ...),
which Geolex never did, so most OVERLIES edges are now curated and the text
relations ADD to them. The difference between the two is itself worth
querying: `sources` on an edge says who asserted it.

Ages come ONLY from curated metadata. The labeller's chronostrat tags every
interval mentioned anywhere in a passage, including the ages of neighbouring
units, and "which units are Cambrian?" is exactly the query that noise ruins.

THE THREE MODELLING DECISIONS THAT MATTER
-----------------------------------------
1. **Entity resolution.** Curated facts arrive keyed by ASUD's STRATNO, so
   they need none. Prose does: it says "Sugarbag Ck. Quartzite" or "Nirranda
   Gp". Names are reduced to a core ("sugarbag ck") before matching, homonyms
   are broken by shared states, and names that match nothing become
   placeholder Unit nodes (``asud: false``) so the relation is not lost.
2. **Canonical direction.** "A underlies B" is stored as B-[:OVERLIES]->A,
   and "A is intruded by B" as B-[:INTRUDES]->A. One relationship type per
   fact means one way to query it -- which matters when a small model is
   writing the query.
3. **A computed timescale.** Intervals carry numeric ages, so each is linked
   to the intervals that contain it (Statherian -[:WITHIN]-> Paleoproterozoic
   -[:WITHIN]-> Proterozoic ...). That turns "Paleoproterozoic units" into a
   graph traversal that also finds units tagged only "Statherian". No amount
   of vector search can do that.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from geosft.data import asud
from geosft.data.label import ABBREV_RANKS, LITH_AS_RANK, RANK_WORDS, Labeller, word_rank
from geosft.data.match import Gazetteer
from geosft.data.states import STATE_NAMES

INTERVAL_RANKS = ["age", "epoch", "period", "era", "eon", "supereon"]

#: Canonicalisation of the labeller's relation kinds: (edge type, reverse?, props)
RELATION_MAP = {
    "overlies": ("OVERLIES", False, {}),
    "underlies": ("OVERLIES", True, {}),
    "unconformable_on": ("OVERLIES", False, {"unconformable": True}),
    "grades_into": ("GRADES_INTO", False, {}),
    "intertongues_with": ("INTERTONGUES_WITH", False, {}),
    "equivalent_to": ("EQUIVALENT_TO", False, {}),
    "intrudes": ("INTRUDES", False, {}),
    "intruded_by": ("INTRUDES", True, {}),
}

#: ASUD's curated relation wording -> the same canonical edges.
CURATED_MAP = {
    "overlies": ("OVERLIES", False), "underlies": ("OVERLIES", True),
    "is equivalent to": ("EQUIVALENT_TO", False), "grades into": ("GRADES_INTO", False),
    "is interbedded with": ("INTERTONGUES_WITH", False),
    "intermingles with": ("INTERTONGUES_WITH", False),
    "intrudes": ("INTRUDES", False), "is intruded by": ("INTRUDES", True),
}

#: ASUD rank -> the schema rank, read together with the name ("X Suite" is
#: ASUD rank "Group, Suite" but a Suite by name).
def schema_rank(asud_rank: str, name: str) -> str:
    tail = word_rank(name.split()[-1])[1] if name.split() else "Unknown"
    return {
        "Formation, beds": "Formation", "Member, phase": "Member", "Subgroup": "Subgroup",
        "Bed": "Bed",
        "Group, Suite": "Suite" if tail == "Suite" else "Group",
        "Supergroup": "Supersuite" if tail == "Supersuite" else "Supergroup",
    }.get(asud_rank, "Unknown")


_RANK_OR_LITH = set(RANK_WORDS) | LITH_AS_RANK | {"subgroup", "supergroup"}
_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)")


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------

def clean_name(s: str) -> str:
    """Strip markup: 'Mount Walsh Granite (G4)' -> 'Mount Walsh Granite'."""
    s = _BRACKETS.sub("", s).replace("/", " ")
    return re.sub(r"\s+", " ", s).strip(" .,;:*?")


def core_name(s: str) -> str:
    """The part of a unit name that identifies it, lowercased.

    'Alsace Quartzite' -> 'alsace'; 'Bulgonunna Volcanics' -> 'bulgonunna';
    'Mount Isa Group' -> 'mount isa'. Trailing rank and lithology words are
    how prose varies a name, so they are exactly what must not be matched on.
    """
    words = clean_name(s).split()
    while len(words) > 1 and (word_rank(words[-1])[0] or words[-1].lower() in _RANK_OR_LITH
                              or words[-1].rstrip(".") in ABBREV_RANKS):
        words.pop()
    return " ".join(words).lower()


def rank_of(phrase: str) -> str:
    """Rank from a name's last word. Lithology-as-rank means Formation."""
    words = clean_name(phrase).split()
    return word_rank(words[-1])[1] if words else "Unknown"


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
    """Name -> unit key. ASUD units first, placeholders for the rest."""

    def __init__(self, graph: Graph):
        self.g = graph
        self.index: dict[str, list[str]] = defaultdict(list)
        for key, u in graph.units.items():
            self.index[core_name(u["name"])].append(key)
        self.outcomes: Counter = Counter()

    def resolve(self, name: str, context_states: list[str] | None = None,
                rank: str = "Unknown") -> str:
        core = core_name(name)
        cands = [k for k in self.index.get(core, []) if self.g.units[k]["asud"]]
        # "Murchison Granite" and "Murchison Volcanics" share a core: an exact
        # name, then a matching rank, narrows the field before states do
        exact = [k for k in cands if self.g.units[k]["name"].lower() == clean_name(name).lower()]
        same_rank = [k for k in cands if rank != "Unknown" and self.g.units[k]["rank"] == rank]
        cands = exact or (same_rank if len(same_rank) >= 1 else cands)
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
        # unknown to ASUD (or outside this graph): keep it as a placeholder
        key = f"name:{core}"
        if key not in self.g.units:
            self.g.units[key] = {
                "key": key, "unit_id": None, "name": clean_name(name), "full_name": clean_name(name),
                "rank": rank, "asud": False, "states": [], "aliases": [], "url": None,
                "age_text": [], "thickness_min_m": None, "thickness_max_m": None,
                "has_text": False,
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
    ivs = {c["name"]: c for c in vocab["chronostrat"]}
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


def _age_text(u: asud.Unit) -> list[str]:
    ages = [a for a in (u.base_age, u.top_age) if a]
    span = " to ".join(dict.fromkeys(ages))
    ma = (f" ({u.base_ma:g}-{u.top_ma:g} Ma)" if u.base_ma and u.top_ma is not None else "")
    return [span + ma] if span else []


def build(passages_path: Path, vocab_path: Path, limit_units: int | None = None,
          seed: int = 0) -> Graph:
    vocab = json.loads(vocab_path.read_text())
    labeller = Labeller.from_vocab(vocab)
    intervals, within = build_intervals(vocab)
    iv_names = {i["name"] for i in intervals}
    canon_age = {n.lower(): n for n in iv_names} | vocab.get("chronostrat_aliases", {})
    liths = Gazetteer({n: n for n in vocab["lithologies"]}, plurals=True)
    g = Graph(intervals=intervals)
    for child, parent in within:
        g.edge("WITHIN", f"interval:{child}", f"interval:{parent}")

    rows = load_records(passages_path, limit_units, seed)
    by_unit: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)

    # 1. ASUD units: identity and curated metadata --------------------------
    # The whole lexicon (~18k units), not just the units with passages: the
    # graph can answer "how many Cambrian units in Tasmania" for real. A smoke
    # run keeps the sampled units plus their curated neighbours.
    all_units = asud.load_units()
    related = asud.load_related()
    if limit_units:
        keep = set(by_unit)
        for r in related:
            if r.src in by_unit or r.dst in by_unit:
                keep |= {r.src, r.dst}
        keep |= {all_units[sn].parent_stratno for sn in list(keep)
                 if sn in all_units and all_units[sn].parent_stratno}
        units = {sn: u for sn, u in all_units.items() if sn in keep}
    else:
        units = all_units

    def key_of(sn: int) -> str:
        return f"asud:{sn}"

    for sn, u in units.items():
        key = key_of(sn)
        g.units[key] = {
            "key": key, "unit_id": sn, "name": u.name, "full_name": u.name,
            "rank": schema_rank(u.rank, u.name), "asud": True, "states": u.states,
            "aliases": [], "url": u.url, "age_text": _age_text(u),
            "thickness_min_m": u.thickness_min_m, "thickness_max_m": u.thickness_max_m,
            "has_text": sn in by_unit,
        }
        for code in u.states:
            g.edge("OCCURS_IN", key, f"state:{code}", source="asud")
        for prov in u.provinces:
            g.edge("IN_PROVINCE", key, f"province:{prov}", source="asud")
        # curated lithology: ASUD's one-line description, tagged with the
        # same rock vocabulary the text uses, so a question about "granite"
        # matches both
        for lith in liths.values(u.lithology_desc or ""):
            g.edge("HAS_LITHOLOGY", key, f"lithology:{lith.lower()}", source="asud")
        for a in (u.base_age, u.top_age):
            if a and (name := canon_age.get(a.lower())):
                g.edge("HAS_AGE", key, f"interval:{name}", source="asud")
        if u.parent_stratno in units and u.parent_stratno != sn:
            g.edge("PART_OF", key, key_of(u.parent_stratno), source="asud")

    for r in related:
        if r.src not in units or r.dst not in units or r.relation not in CURATED_MAP:
            continue
        type_, reverse = CURATED_MAP[r.relation]
        src, dst = (key_of(r.dst), key_of(r.src)) if reverse else (key_of(r.src), key_of(r.dst))
        props = {"contact": r.contact} if r.contact else {}
        if type_ == "OVERLIES" and "unconformity" in r.contact:
            props["unconformable"] = True
        g.edge(type_, src, dst, source="asud", **props)

    resolver = Resolver(g)

    # 2. passages, and what the rule labeller finds in them ---------------
    text_liths: dict[str, Counter] = defaultdict(Counter)
    text_mins: dict[str, Counter] = defaultdict(Counter)
    subject_confirmed = 0
    for uid, recs in by_unit.items():
        key = key_of(uid)
        if key not in g.units:
            continue
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
            if lab.thickness and u["thickness_max_m"] is None:
                lo, hi = lab.thickness.min_m, lab.thickness.max_m
                u["thickness_min_m"], u["thickness_max_m"] = lo, hi
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
        "asud_units": sum(u["asud"] for u in g.units.values()),
        "units_with_text": sum(u.get("has_text", False) for u in g.units.values()),
        "placeholder_units": sum(not u["asud"] for u in g.units.values()),
        "intervals": len(g.intervals),
        "edges": dict(sorted(types.items())),
        "overlies_by_source": dict(Counter("+".join(e["props"].get("sources", []))
                                           for e in g.edges if e["type"] == "OVERLIES")),
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
        if "contact" in p:
            mp.setdefault("contact", p.pop("contact"))
        mp.update(p)
    return list(merged.values())
