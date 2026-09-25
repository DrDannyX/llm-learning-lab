"""Rule-based weak supervision: passage -> GeoExtraction.

THE CORE RULE OF THIS FILE
--------------------------
A target field may only contain facts that are **present in the passage
itself**. ASUD hands us tempting metadata (curated ages, states, relations,
thickness), but training on facts the model cannot see is how you teach a
model to hallucinate confidently. The metadata is therefore used only to
*audit* the labeller (see `agreement_report`), never to write labels.

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
from .states import NOT_STATES, STATES, abbreviations

FT_TO_M = 0.3048

RANK_WORDS = {
    "supergroup": "Supergroup", "supersuite": "Supersuite", "group": "Group",
    "suite": "Suite", "subgroup": "Subgroup", "formation": "Formation",
    "member": "Member", "phase": "Member", "bed": "Bed",
}

#: Rock names that stand in for a rank ("Arburee Rhyolite", "Mount Isa
#: Granodiorite", "Bulgonunna Volcanics"). Measured on ASUD: units named this
#: way are Formation rank 95-100% of the time, so they label as Formation.
#: "Complex" and "Sequence" are deliberately absent -- ASUD ranks them as Group
#: and Formation about equally, so the name says nothing reliable.
LITH_AS_RANK = {
    "shale", "limestone", "sandstone", "chalk", "marl", "clay", "sand",
    "dolomite", "quartzite", "conglomerate", "gravel", "schist", "gneiss",
    "granite", "basalt", "tuff", "slate", "beds", "series", "silt", "till",
    "granodiorite", "monzogranite", "microgranite", "leucogranite", "tonalite",
    "diorite", "monzodiorite", "monzonite", "syenite", "gabbro", "dolerite",
    "volcanics", "rhyolite", "dacite", "andesite", "ignimbrite", "porphyry",
    "metamorphics", "orthogneiss", "siltstone", "mudstone", "breccia", "chert",
    "pegmatite", "serpentinite", "lamproite", "measures", "coal", "pluton",
    "intrusion", "greywacke", "phyllite", "amphibolite", "migmatite", "ironstone",
    "arkose", "tillite", "diamictite", "calcarenite", "adamellite",
    "metagabbro", "metabasalt", "metadolerite", "metasediments", "hornfels",
    "pyroclastics",
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
_THE = r"(?i:(?:the|both)\s+)?"

_BY = r"(?:\s+\w+ly)?\s+by"   # "overlain by", "overlain conformably by"
_COLON = r"\s*:?"              # ASUD templates: "Overlies: Koolpin Formation."
RELATION_PATTERNS: list[tuple[str, str]] = [
    ("unconformable_on", rf"(?i:unconformabl[ye]\s+(?:overlies|overlie|rests\s+on|on))\s+{_THE}{NAME_RE}"),
    ("overlies",         rf"(?i:\boverlies|\boverlie){_COLON}\s+{_THE}{NAME_RE}"),
    ("overlies",         rf"(?i:\brests\s+(?:conformably\s+)?(?:up)?on)\s+{_THE}{NAME_RE}"),
    ("overlies",         rf"(?i:\b(?:dis)?conformable\s+on)\s+{_THE}{NAME_RE}"),
    ("overlies",         rf"(?i:\babove)\s+{_THE}{NAME_RE}"),
    ("overlies",         rf"(?i:\bunderlain{_BY})\s+{_THE}{NAME_RE}"),
    # "the underlying X" / "Underlying unit: X" put X below the subject; a bare
    # participle ("an interval overlying X") describes the SUBJECT, so it is
    # the opposite relation -- the article is what disambiguates them.
    ("overlies",         rf"(?i:\b(?:the\s+underlying|underlying\s+units?\s*:))\s+{NAME_RE}"),
    ("underlies",        rf"(?i:\b(?:the\s+overlying|overlying\s+units?\s*:))\s+{NAME_RE}"),
    ("overlies",         rf"(?<![Tt]he )(?i:\boverlying)\s+{_THE}{NAME_RE}"),
    ("underlies",        rf"(?<![Tt]he )(?i:\bunderlying)\s+{_THE}{NAME_RE}"),
    ("overlies",         rf"(?:^|[.;]\s*)Over\s+{NAME_RE}"),     # shorthand "Over X; under Y"
    ("underlies",        rf"(?:^|[.;]\s*)(?i:under)\s+{NAME_RE}"),
    ("underlies",        rf"(?i:\bunderlies|\bunderlie){_COLON}\s+{_THE}{NAME_RE}"),
    ("underlies",        rf"(?i:\bbelow)\s+{_THE}{NAME_RE}"),
    ("underlies",        rf"(?i:\boverlain{_BY})\s+{_THE}{NAME_RE}"),
    ("underlies",        rf"(?i:\btransgressed\s+by)\s+{_THE}{NAME_RE}"),
    ("grades_into",      rf"(?i:\bgrades?\s+(?:laterally\s+|eastward\s+|westward\s+|northward\s+|southward\s+)?(?:in)?to)\s+{_THE}{NAME_RE}"),
    ("intertongues_with",rf"(?i:\b(?:inter(?:tongues?|fingers?|fingering|digitates?)|interbedded)\s+(?:with|within))\s+{_THE}{NAME_RE}"),
    ("equivalent_to",    rf"(?i:\b(?:equivalent\s+(?:to|of)|correlated\s+(?:with|to)|correlates\s+with|correlative\s+(?:with|of)))\s+{_THE}{NAME_RE}"),
    ("intruded_by",      rf"(?i:\bintruded\s+by)\s+{_THE}{NAME_RE}"),
    ("intrudes",         rf"(?i:\bintrudes?|\bintruding)\s+{_THE}{NAME_RE}"),
]
_COMPILED_RELATIONS = [(k, re.compile(p)) for k, p in RELATION_PATTERNS]

_NUM = r"(\d{1,5}(?:[.,]\d+)?)"
_LEN_UNIT = r"(feet|foot|ft\.?|meters?|metres?|km|m\.?)\b"
_RANGE_RE = re.compile(rf"{_NUM}\s*(?:to|-|–|or)\s*{_NUM}\s*(?:\+\s*)?{_LEN_UNIT}", re.I)
_SINGLE_RE = re.compile(rf"(?:about|approximately|some|up\s+to|as\s+much\s+as|max(?:imum)?\s+of)?\s*{_NUM}\s*(?:\+\s*)?{_LEN_UNIT}", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.;])\s+")
_THICK_HINT = re.compile(r"thick|thickness", re.I)

_STOP_NAMES = {
    "The", "This", "That", "These", "Those", "It", "In", "At", "Age", "Pg",
    "Report", "Named", "Type", "Gulf", "North", "South", "East", "West",
    "Upper", "Lower", "Middle", "Late", "Early", "Basal", "Top", "Base",
}


#: One to four capitalised words, the last being the candidate rank/rock word.
_PROPER_NAME = re.compile(r"\b((?:[A-Z][A-Za-z'\u2019\-]+\s+){1,4})([A-Z][A-Za-z']+)\b")
#: ASUD's house abbreviations for ranks and rock names in unit names.
ABBREV_RANKS = {"Fm", "Fms", "Gp", "Gps", "Sgp", "Supgp", "Mbr", "Sst", "Sandst", "Lst", "L'st",
                "Sltst", "Volcs", "Congl", "Sh", "Bds", "Gr", "Dol", "Qtzite"}


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
    if u == "km":
        return round(v * 1000, 2)
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
    if parts and word_rank(parts[-1])[0]:
        parts[-1] = parts[-1].capitalize()
    return " ".join(parts)


#: Words that open a captured name but are not part of it:
#: "overlies Palaeoproterozoic Murphy Metamorphics" -> "Murphy Metamorphics".
_NAME_LEAD = re.compile(
    r"^(?:(?:Early|Middle|Late|Lower|Upper|Basal|Uppermost|Lowermost)\s+|"
    r"(?i:(?:pal(?:a)?eo|meso|neo|eo)?(?:proterozoic|archa?ean|zoic))\s+|"
    r"(?:Cambrian|Ordovician|Silurian|Devonian|Carboniferous|Permian|Triassic|Jurassic|"
    r"Cretaceous|Tertiary|Paleogene|Palaeogene|Neogene|Quaternary|Precambrian)\s+)+")


def _core(name: str) -> str:
    """'Breakfast Sandstone' -> 'breakfast': the name without its rank/lithology tail."""
    words = name.split()
    while len(words) > 1 and word_rank(words[-1])[0]:
        words.pop()
    return " ".join(words).lower()


def extract_relations(passage: str, strat_names: set[str], self_name: str | None) -> list[Relation]:
    rels: list[Relation] = []
    for kind, rx in _COMPILED_RELATIONS:
        for m in rx.finditer(passage):
            name = _NAME_LEAD.sub("", _clean_name(m.group(1)))
            head = name.split()[0] if name else ""
            if head in _STOP_NAMES or len(head) < 3:
                continue
            # the head word must be a real stratigraphic name, else we are just
            # capturing capitalised English
            if head not in strat_names:
                continue
            # a unit is never related to itself -- but "Murchison Granite" is
            # not the same unit as "Murchison Volcanics", so only the full name
            # or the bare core ("Breakfast") counts as self
            if self_name and name.lower() in {self_name.lower(), _core(self_name)}:
                continue
            rels.append(Relation(kind=kind, unit=name))
    # "unconformably overlies X" also fires the plain "overlies" rule
    unconf = {r.unit.lower() for r in rels if r.kind == "unconformable_on"}
    return [r for r in rels if not (r.kind == "overlies" and r.unit.lower() in unconf)]


def infer_rank(passage: str, unit_name: str | None) -> str:
    """Read the rank word that follows the unit name.

    Order matters: "Aarde shale member of Howard limestone" must resolve to
    Member, not to Formation via the lithology-as-rank fallback. So we collect
    the short window after the name and let an explicit rank word win.
    """
    if not unit_name:
        return "Unknown"
    rx = re.compile(rf"\b{re.escape(unit_name)}\b((?:\s+[A-Za-z'\-]+){{0,3}})", re.I)
    # ASUD names usually carry their own rank word ("Tumblagooda Sandstone",
    # "Glyde Hill Volcanic Complex"), so the name's last word is read first;
    # Geolex-style bare names ("Austin chalk") fall through to the window.
    tail = unit_name.split()[-1] if len(unit_name.split()) > 1 else ""
    fallback = None
    for m in rx.finditer(passage):
        for tok in [tail, *m.group(1).split()]:
            kind, rank = word_rank(tok)
            if kind == "rank":
                return rank
            if kind == "lith" and fallback is None:
                fallback = rank
    return fallback or "Unknown"


def word_rank(tok: str) -> tuple[str | None, str]:
    """('rank', R) for a rank word, ('lith', 'Formation') for a lithology-as-rank
    word, (None, 'Unknown') otherwise.

    The exact word is tried before the de-pluralised one: "Beds" is ASUD's
    Formation-rank informal unit, not a Bed, and "Volcanics" is not "volcanic".
    """
    w = tok.lower()
    for form in (w, w.rstrip("s")):
        if form in RANK_WORDS:
            return "rank", RANK_WORDS[form]
        if form in LITH_AS_RANK:
            return "lith", "Formation"
    return None, "Unknown"


@dataclass
class Labeller:
    lithologies: Gazetteer
    chronostrat: Gazetteer
    minerals: Gazetteer
    states: Gazetteer
    strat_names: set[str] = field(default_factory=set)
    unit_names: Gazetteer | None = None

    @classmethod
    def from_vocab(cls, vocab: dict) -> "Labeller":
        liths = Gazetteer({n: n for n in vocab["lithologies"]}, plurals=True)
        chrono = Gazetteer({c["name"].lower(): c["name"] for c in vocab["chronostrat"]}
                           | vocab.get("chronostrat_aliases", {}))
        # minerals: 5+ chars already filtered upstream; plurals are rare in prose
        mins = Gazetteer({m.lower(): m for m in vocab["minerals"]})
        sts = Gazetteer(dict(STATES) | NOT_STATES)
        heads = {n.split()[0] for n in vocab["strat_names"] if n}
        names = Gazetteer({n.lower(): n for n in vocab["strat_names"] if len(n.split()) > 1},
                          max_ngram=6)
        return cls(liths, chrono, mins, sts, heads, names)

    def mask_names(self, passage: str) -> str:
        """Blank out proper unit names before tagging rock, mineral, age and place terms.

        "Tumblagooda Sandstone" names a unit; it does not report sandstone.
        "Alaska Bench Limestone" does not put limestone -- or Alaska -- in the
        passage's facts. The gold review of the Geolex version found this the
        single most common rule error, and ASUD passages lead with a name, so
        without the mask nearly every label would carry it. Only capitalised
        multi-word names are masked: "coal measures" in lowercase is prose.
        """
        out = list(passage)
        spans = [(h.start, h.end) for h in (self.unit_names.find(passage) if self.unit_names else [])
                 if h.surface[:1].isupper()]
        # Names ASUD no longer lists (superseded, misspelt, abbreviated) still
        # look like names: capitalised words ending in a capitalised rank or
        # rock word -- "Carcoar Granite", "Nirranda Gp", "Newer Volcs".
        for m in _PROPER_NAME.finditer(passage):
            if word_rank(m.group(2))[0] or m.group(2) in ABBREV_RANKS:
                spans.append((m.start(), m.end()))
        for a, b in spans:
            out[a:b] = " " * (b - a)
        return "".join(out)

    @staticmethod
    def _common(gaz: Gazetteer, text: str) -> list[str]:
        """Gazetteer values, ignoring capitalised hits in mid-sentence.

        ASUD writes rock and mineral names in lowercase, so "Silver" in
        "Silver Spur Subprovince" is a place, while "Biotite granite: I-type"
        opens a sentence and is a mineral.
        """
        keep = set()
        for h in gaz.find(text):
            before = text[:h.start].rstrip()
            after = text[h.end:].lstrip()[:1]
            if h.surface[:1].isupper() and (
                    (before and before[-1] not in ".:;") or after.isupper()):
                continue  # mid-sentence capital, or the start of a longer proper name
            keep.add(h.canonical)
        return sorted(keep)

    def label(self, rec: dict) -> GeoExtraction:
        passage = rec["passage"]
        unit_name = (rec.get("unit_name") or "").strip() or None
        # The unit name must be the SUBJECT of this passage, not a passing
        # mention. Every passage is rendered as a lexicon entry that leads with
        # its unit's name (see fetch.py), so an early mention is required; a
        # passage that does not lead with it is about something else.
        if unit_name:
            m = re.search(rf"\b{re.escape(unit_name)}\b", passage, re.I)
            if m is None or m.start() > 150:
                unit_name = None

        text = self.mask_names(passage)
        # NOT_STATES entries ("Victoria River") map to "" and are dropped, and a
        # state name followed by another capitalised word is a place name
        # ("Victoria Bridge", "Queensland Museum"), not the state
        states = {h.canonical for h in self.states.find(text)
                  if h.canonical and not text[h.end:].lstrip()[:1].isupper()}
        states |= abbreviations(text)
        return GeoExtraction(
            unit_name=unit_name,
            rank=infer_rank(passage, unit_name),
            lithologies=self._common(self.lithologies, text),
            chronostrat=self.chronostrat.values(text),
            minerals=self._common(self.minerals, text),
            thickness=parse_thickness(passage),
            relations=extract_relations(passage, self.strat_names, unit_name),
            states=sorted(states),
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
# The labeller never *reads* ASUD's curated metadata, which leaves that
# metadata free to act as an independent check. ASUD curates far more than
# Geolex did -- rank, fine-grained ages, states, thickness, AND the unit's
# stratigraphic relations -- so every field but minerals has a tripwire.
# Run this every time you touch the rules.

#: ASUD rank -> the schema ranks a correct label may take
_RANK_OK = {
    "Formation, beds": {"Formation"}, "Member, phase": {"Member"},
    "Group, Suite": {"Group", "Suite"}, "Supergroup": {"Supergroup", "Supersuite"},
    "Subgroup": {"Subgroup"}, "Bed": {"Bed"},
}


def _tokens(strings: list[str]) -> set[str]:
    return {w.lower() for s in strings for w in re.findall(r"[A-Za-z]+", s) if len(w) > 3}


def _head(name: str) -> str:
    return name.split()[0].lower() if name else ""


def _canon_kind(kind: str) -> str:
    return "overlies" if kind == "unconformable_on" else kind


def agreement_report(records: list[dict], labeller: "Labeller") -> dict:
    """Compare rule output against ASUD curated metadata, field by field."""
    chrono_hit = chrono_tot = 0
    state_tp = state_fp = state_fn = 0
    rank_hit = rank_tot = 0
    rel_hit = rel_tot = 0
    thick_hit = thick_tot = 0
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

        # rank: does the name-derived rank agree with ASUD's?
        ok = _RANK_OK.get(rec.get("asud_rank") or "")
        if ok and g.rank != "Unknown":
            rank_tot += 1
            rank_hit += g.rank in ok

        # chronostrat: does any predicted interval appear in the curated ages?
        curated_age = _tokens(rec.get("age_names") or [])
        if curated_age and g.chronostrat:
            chrono_tot += 1
            if _tokens(g.chronostrat) & curated_age:
                chrono_hit += 1

        # states: overlap with the curated list. Expect LOW recall and do not
        # "fix" it: we extract states NAMED in this passage, while ASUD records
        # every jurisdiction the unit occurs in. The number is a drift
        # tripwire, not an accuracy score.
        curated_states = set(rec.get("states") or []) - {"OFF"}
        pred_states = set(g.states)
        if curated_states or pred_states:
            state_tp += len(curated_states & pred_states)
            state_fp += len(pred_states - curated_states)
            state_fn += len(curated_states - pred_states)

        # relations: is each extracted relation one ASUD also curates? This is
        # a PRECISION floor only -- ASUD's list is incomplete, so a correct
        # relation it lacks counts as a miss here.
        curated_rel = {(_canon_kind(r["kind"]), _head(r["unit"]))
                       for r in rec.get("curated_relations") or []}
        if curated_rel:
            for r in g.relations:
                rel_tot += 1
                rel_hit += (_canon_kind(r.kind), _head(r.unit)) in curated_rel

        # thickness: inside the curated range (with 10% slack for rounding)?
        lo, hi = rec.get("thickness_min_m"), rec.get("thickness_max_m")
        if g.thickness and g.thickness.max_m is not None and hi:
            thick_tot += 1
            thick_hit += (lo or 0) * 0.9 <= g.thickness.max_m <= hi * 1.1

    prec = state_tp / (state_tp + state_fp) if (state_tp + state_fp) else 0.0
    rec_ = state_tp / (state_tp + state_fn) if (state_tp + state_fn) else 0.0
    n = max(len(records), 1)

    def pct(a: int, b: int) -> float | None:
        return round(100 * a / b, 1) if b else None

    return {
        "n_passages": len(records),
        "unit_name_kept_pct": round(100 * name_kept / n, 1),
        "rank_agreement_pct": pct(rank_hit, rank_tot),
        "chronostrat_consistency_pct": pct(chrono_hit, chrono_tot),
        "relations_in_curated_pct": pct(rel_hit, rel_tot),
        "thickness_in_curated_range_pct": pct(thick_hit, thick_tot),
        "states_precision": round(prec, 3),
        "states_recall": round(rec_, 3),
        "field_fill_pct": {k: round(100 * v / n, 1) for k, v in sorted(field_counts.items())},
    }
