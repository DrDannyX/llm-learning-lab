"""Rule-based weak supervision: passage -> GeoExtraction.

THE CORE RULE OF THIS FILE
--------------------------
A target field may only contain facts that are **present in the passage
itself**. Geolex hands us tempting metadata (curated ages, state lists), but
training on facts the model cannot see is how you teach a model to hallucinate
confidently. The metadata is therefore used only to *audit* the labeller
(see `agreement_report`), never to write labels.

WHAT THIS BUYS YOU, AND WHAT IT COSTS
-------------------------------------
Rule labels are cheap, auditable and reproducible -- you can read every
decision this file makes. The cost is a ceiling: a model trained purely on
these labels learns to imitate the rules, so its score against the rules will
approach 100% without proving anything.

Two things keep the project honest:
  1. the rules are high-precision / low-recall by design, so the model has to
     *generalise* past the gazetteer to score well on unseen units, and
  2. a human-checked gold set (data/gold/) is the set that actually counts.
Always read the gold numbers, not the rule-matched numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schema import GeoExtraction, Relation, Thickness
from .match import Gazetteer
from .states import STATES

FT_TO_M = 0.3048

RANK_WORDS = {
    "supergroup": "Supergroup", "group": "Group", "subgroup": "Subgroup",
    "formation": "Formation", "member": "Member", "bed": "Bed",
    "tongue": "Tongue", "lentil": "Lentil",
}

#: Lithology words that historically stood in for a rank ("Austin chalk",
#: "Eagle Ford shales"). Treated as an informal Formation.
LITH_AS_RANK = {
    "shale", "limestone", "sandstone", "chalk", "marl", "clay", "sand",
    "dolomite", "quartzite", "conglomerate", "gravel", "schist", "gneiss",
    "granite", "basalt", "tuff", "slate", "beds", "series", "silt", "till",
}

# NOTE: these patterns are compiled WITHOUT re.IGNORECASE on purpose. The name
# group relies on [A-Z] to find proper nouns, and a global ignorecase flag
# silently turns that into "any letter" -- which makes the group swallow
# ordinary words ("Church member of Howard limestone"). Trigger words that do
# need case-insensitivity are wrapped in scoped (?i:...) groups instead.
_NAME_CORE = r"[A-Z][A-Za-z'\u2019\-]+(?:\s+[A-Z][A-Za-z'\u2019\-]+){0,3}"
_RANK_ALT = "|".join(sorted(RANK_WORDS | {w: w for w in LITH_AS_RANK}))
_TAIL = rf"(?:\s+(?i:{_RANK_ALT})s?)?"
NAME_RE = rf"({_NAME_CORE}{_TAIL})"
_THE = r"(?i:the\s+)?"

RELATION_PATTERNS: list[tuple[str, str]] = [
    ("unconformable_on", rf"(?i:unconformably\s+(?:overlies|rests\s+on))\s+{_THE}{NAME_RE}"),
    ("overlies",         rf"(?i:\boverlies)\s+{_THE}{NAME_RE}"),
    ("overlies",         rf"(?i:\brests\s+(?:conformably\s+)?(?:up)?on)\s+{_THE}{NAME_RE}"),
    ("overlies",         rf"(?i:\babove)\s+{_THE}{NAME_RE}"),
    ("underlies",        rf"(?i:\bunderlies)\s+{_THE}{NAME_RE}"),
    ("underlies",        rf"(?i:\bbelow)\s+{_THE}{NAME_RE}"),
    ("underlies",        rf"(?i:\boverlain\s+by)\s+{_THE}{NAME_RE}"),
    ("grades_into",      rf"(?i:\bgrades?\s+(?:laterally\s+|eastward\s+|westward\s+|northward\s+|southward\s+)?(?:in)?to)\s+{_THE}{NAME_RE}"),
    ("intertongues_with",rf"(?i:\binter(?:tongues|fingers)\s+with)\s+{_THE}{NAME_RE}"),
    ("equivalent_to",    rf"(?i:\b(?:equivalent\s+to|correlated\s+with|correlative\s+(?:with|of)))\s+{_THE}{NAME_RE}"),
]
_COMPILED_RELATIONS = [(k, re.compile(p)) for k, p in RELATION_PATTERNS]

_NUM = r"(\d{1,5}(?:[.,]\d+)?)"
_LEN_UNIT = r"(feet|foot|ft\.?|meters?|metres?|m\.?)\b"
_RANGE_RE = re.compile(rf"{_NUM}\s*(?:to|-|–|or)\s*{_NUM}\s*(?:\+\s*)?{_LEN_UNIT}", re.I)
_SINGLE_RE = re.compile(rf"(?:about|approximately|some|up\s+to|as\s+much\s+as|max(?:imum)?\s+of)?\s*{_NUM}\s*(?:\+\s*)?{_LEN_UNIT}", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.;])\s+")
_THICK_HINT = re.compile(r"thick|thickness", re.I)

_STOP_NAMES = {
    "The", "This", "That", "These", "Those", "It", "In", "At", "Age", "Pg",
    "Report", "Named", "Type", "Gulf", "North", "South", "East", "West",
    "Upper", "Lower", "Middle", "Late", "Early", "Basal", "Top", "Base",
}


def _to_metres(value: str, unit: str) -> float | None:
    try:
        v = float(value.replace(",", ""))
    except ValueError:
        return None
    u = unit.lower().rstrip(".")
    if u in {"feet", "foot", "ft"}:
        return round(v * FT_TO_M, 2)
    if u in {"m", "meter", "meters", "metre", "metres"}:
        return round(v, 2)
    return None


def parse_thickness(passage: str) -> Thickness | None:
    """Only trust a measurement in a sentence that actually talks about thickness.

    Lexicon prose is full of distances, elevations and page numbers; the
    'thick' keyword is what separates a thickness from a road log.
    """
    for sent in _SENT_SPLIT.split(passage):
        if not _THICK_HINT.search(sent):
            continue
        m = _RANGE_RE.search(sent)
        if m:
            lo, hi = _to_metres(m.group(1), m.group(3)), _to_metres(m.group(2), m.group(3))
            if lo is not None and hi is not None and lo <= hi:
                return Thickness(min_m=lo, max_m=hi)
        m = _SINGLE_RE.search(sent)
        if m:
            v = _to_metres(m.group(1), m.group(2))
            if v is not None and v > 0:
                return Thickness(min_m=v, max_m=v)
    return None


def _clean_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw).strip(" .,;:")
    name = re.sub(r"\s+(?:of|in|at|and|the)$", "", name, flags=re.I)
    # trailing lowercase rank word normalises to title case: "Taylor group" -> "Taylor Group"
    parts = name.split()
    if parts and parts[-1].lower().rstrip("s") in (RANK_WORDS | {w: w for w in LITH_AS_RANK}):
        parts[-1] = parts[-1].capitalize()
    return " ".join(parts)


def extract_relations(passage: str, strat_names: set[str], self_name: str | None) -> list[Relation]:
    rels: list[Relation] = []
    for kind, rx in _COMPILED_RELATIONS:
        for m in rx.finditer(passage):
            name = _clean_name(m.group(1))
            head = name.split()[0] if name else ""
            if head in _STOP_NAMES or len(head) < 4:
                continue
            # the head word must be a real stratigraphic name, else we are just
            # capturing capitalised English
            if head not in strat_names:
                continue
            if self_name and head.lower() == self_name.lower():
                continue
            rels.append(Relation(kind=kind, unit=name))
    return rels


def infer_rank(passage: str, unit_name: str | None) -> str:
    """Read the rank word that follows the unit name.

    Order matters: "Aarde shale member of Howard limestone" must resolve to
    Member, not to Formation via the lithology-as-rank fallback. So we collect
    the short window after the name and let an explicit rank word win.
    """
    if not unit_name:
        return "Unknown"
    rx = re.compile(rf"\b{re.escape(unit_name)}\b((?:\s+[A-Za-z'\-]+){{0,3}})", re.I)
    fallback = None
    for m in rx.finditer(passage):
        for tok in m.group(1).split():
            word = tok.lower().rstrip("s")
            if word in RANK_WORDS:
                return RANK_WORDS[word]
            if word in LITH_AS_RANK and fallback is None:
                fallback = "Formation"
    return fallback or "Unknown"


@dataclass
class Labeller:
    lithologies: Gazetteer
    chronostrat: Gazetteer
    minerals: Gazetteer
    states: Gazetteer
    strat_names: set[str] = field(default_factory=set)

    @classmethod
    def from_vocab(cls, vocab: dict) -> "Labeller":
        liths = Gazetteer({n: n for n in vocab["lithologies"]}, plurals=True)
        chrono = Gazetteer({c["name"].lower(): c["name"] for c in vocab["chronostrat"]})
        # minerals: 5+ chars already filtered upstream; plurals are rare in prose
        mins = Gazetteer({m.lower(): m for m in vocab["minerals"]})
        sts = Gazetteer(dict(STATES))
        heads = {n.split()[0] for n in vocab["strat_names"] if n}
        return cls(liths, chrono, mins, sts, heads)

    def label(self, rec: dict) -> GeoExtraction:
        passage = rec["passage"]
        unit_name = (rec.get("unit_name") or "").strip() or None
        # The unit name must be the SUBJECT of this passage, not a passing
        # mention. Geolex files each reference summary under a unit, but a
        # summary may really be about a neighbouring unit. Lexicon summaries
        # lead with their subject, so requiring an early mention is a cheap,
        # high-precision test.
        if unit_name:
            m = re.search(rf"\b{re.escape(unit_name)}\b", passage, re.I)
            if m is None or m.start() > 150:
                unit_name = None

        return GeoExtraction(
            unit_name=unit_name,
            rank=infer_rank(passage, unit_name),
            lithologies=self.lithologies.values(passage),
            chronostrat=self.chronostrat.values(passage),
            minerals=self.minerals.values(passage),
            thickness=parse_thickness(passage),
            relations=extract_relations(passage, self.strat_names, unit_name),
            states=self.states.values(passage),
        )


def filled_fields(g: GeoExtraction) -> int:
    """How many non-trivial fields the labeller actually populated."""
    return sum([
        bool(g.lithologies), bool(g.chronostrat), bool(g.minerals),
        bool(g.relations), g.thickness is not None, bool(g.states),
        g.rank != "Unknown",
    ])


# --------------------------------------------------------------------------
# Auditing the labeller
# --------------------------------------------------------------------------
# The labeller never *reads* Geolex's curated metadata, which leaves that
# metadata free to act as an independent check. If rule-derived chronostrat
# disagrees with the curated age description, or rule-derived states disagree
# with the curated state list, the rules are drifting. This is the cheapest
# quality signal in the project -- run it every time you touch the rules.

def _tokens(strings: list[str]) -> set[str]:
    return {w.lower() for s in strings for w in re.findall(r"[A-Za-z]+", s) if len(w) > 3}


def agreement_report(records: list[dict], labeller: "Labeller") -> dict:
    """Compare rule output against Geolex curated metadata, field by field."""
    chrono_hit = chrono_tot = 0
    state_tp = state_fp = state_fn = 0
    name_kept = 0
    field_counts: dict[str, int] = {}

    for rec in records:
        g = labeller.label(rec)
        for f in ("lithologies", "chronostrat", "minerals", "relations", "states"):
            if getattr(g, f):
                field_counts[f] = field_counts.get(f, 0) + 1
        if g.thickness:
            field_counts["thickness"] = field_counts.get("thickness", 0) + 1
        if g.rank != "Unknown":
            field_counts["rank"] = field_counts.get("rank", 0) + 1
        if g.unit_name:
            name_kept += 1

        # chronostrat: does any predicted interval appear in the curated age text?
        curated_age = _tokens(rec.get("age_description") or [])
        if curated_age and g.chronostrat:
            chrono_tot += 1
            if _tokens(g.chronostrat) & curated_age:
                chrono_hit += 1

        # states: overlap with the curated list. Expect LOW agreement and do
        # not "fix" it: we extract states named in this passage, while Geolex
        # curates every state the unit occurs in across all references. The
        # number is a drift tripwire, not an accuracy score.
        curated_states = set(rec.get("states") or [])
        pred_states = set(g.states)
        if curated_states or pred_states:
            state_tp += len(curated_states & pred_states)
            state_fp += len(pred_states - curated_states)
            state_fn += len(curated_states - pred_states)

    prec = state_tp / (state_tp + state_fp) if (state_tp + state_fp) else 0.0
    rec_ = state_tp / (state_tp + state_fn) if (state_tp + state_fn) else 0.0
    n = max(len(records), 1)
    return {
        "n_passages": len(records),
        "unit_name_kept_pct": round(100 * name_kept / n, 1),
        "chronostrat_consistency_pct": round(100 * chrono_hit / chrono_tot, 1) if chrono_tot else None,
        "states_precision": round(prec, 3),
        "states_recall": round(rec_, 3),
        "field_fill_pct": {k: round(100 * v / n, 1) for k, v in sorted(field_counts.items())},
    }
