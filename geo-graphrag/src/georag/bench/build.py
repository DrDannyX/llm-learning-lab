"""Build the benchmark: questions with known answers, in eight categories.

WHY EIGHT CATEGORIES, NOT ONE SCORE
-----------------------------------
The three systems are good at different things, and a single averaged number
hides exactly the intuition this lab is for. The categories are chosen to pull
them apart:

  category    example                                               gold from  favours
  ----------- ----------------------------------------------------- ---------- --------
  age         What is the geologic age of the Alsace Quartzite?     asud       all
  states      In which Australian states does X occur?              asud       graph
  parent      What larger unit is X part of?                        asud       all
  members     Which units are part of the Mount Isa Group?          asud       graph
  relation    Which unit overlies X?                                asud       all
  multi_hop   Which Cambrian units in Tasmania consist of limestone? asud      graph
  count       How many Permian units occur in New South Wales?      asud       KG only
  descriptive Where was X named from? (LLM-written, 1 passage)      LLM        text

Question SUBJECTS (age, states, parent, members, relation, descriptive) are
drawn only from units that have passages, so RAG always has something to
find. Counts and filters range over the whole graph -- all ~18k ASUD units --
which is exactly the question retrieval of 8 passages cannot answer.

THE CIRCULARITY CAVEAT -- read before quoting any number
-------------------------------------------------------
Seven categories take their gold answers from the same curated metadata the
knowledge graph was built from. On those, the KG is being asked to read back
its own contents, and it will look better than it would against a truly
independent test. `descriptive` is the counterweight: its answers exist only
in prose. Unlike the Geolex version of this lab, `relation` gold is now
CURATED (ASUD records stratigraphic relations; Geolex did not), so it is no
longer the noisy category -- only curated OVERLIES edges become gold.

This is the same lesson as the SFT lab's gold review, from the other side:
whoever builds the test set decides what "better" means.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rich.progress import track

from geosft.data.states import STATE_NAMES

from ..config import Config
from ..ingest.extract import Graph, core_name
from ..llm import complete

CATEGORIES = ["age", "states", "parent", "members", "relation", "multi_hop", "count", "descriptive"]

#: Period- and era-level names a geologist would actually ask about. Much of
#: Australia is Precambrian, so the Proterozoic eras and the Archean count too.
PERIODS = ["Archean", "Paleoproterozoic", "Mesoproterozoic", "Neoproterozoic",
           "Cambrian", "Ordovician", "Silurian", "Devonian", "Carboniferous", "Permian",
           "Triassic", "Jurassic", "Cretaceous", "Paleogene", "Neogene", "Quaternary"]
#: Onshore jurisdictions a question can name ("Offshore Australia" is not a place)
ASK_STATES = {c: n for c, n in STATE_NAMES.items() if c != "OFF"}


def _q(qid: str, category: str, question: str, gold, match: str, source: str,
       provenance: str, reference: str | None = None) -> dict:
    ref = reference if reference is not None else (
        ", ".join(gold) if isinstance(gold, list) else str(gold))
    return {"id": qid, "category": category, "question": question, "gold": gold,
            "match": match, "reference": ref, "source": source, "provenance": provenance}


class Index:
    """Lookup tables over graph.json, so every gold answer is computed from data."""

    def __init__(self, g: Graph):
        self.g = g
        self.out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        self.inc: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for e in g.edges:
            self.out[e["src"]][e["type"]].append(e)
            self.inc[e["dst"]][e["type"]].append(e)
        cores = Counter(core_name(u["name"]) for u in g.units.values() if u["asud"])
        #: question subjects: curated, with passages, a unique core name (no
        #: homonym for the linker to guess between) and a rank word
        self.clean = [k for k, u in g.units.items() if u["asud"] and u.get("has_text")
                      and cores[core_name(u["name"])] == 1 and u["rank"] != "Unknown"]
        self.passages_of: dict[str, list[dict]] = defaultdict(list)
        for p in g.passages:
            self.passages_of[p["unit_key"]].append(p)

    def targets(self, key: str, type_: str, reverse: bool = False) -> list[str]:
        edges = (self.inc if reverse else self.out)[key].get(type_, [])
        return [e["src" if reverse else "dst"] for e in edges]

    def ancestors(self, interval_ref: str) -> set[str]:
        seen, stack = set(), [interval_ref]
        while stack:
            for parent in self.targets(stack.pop(), "WITHIN"):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen

    def periods(self, key: str) -> set[str]:
        names = set()
        for iv in self.targets(key, "HAS_AGE"):
            names |= {r.split(":", 1)[1] for r in self.ancestors(iv) | {iv}}
        return names & set(PERIODS)

    def name(self, key: str) -> str:
        return self.g.units[key]["full_name"]


def gen_age(ix: Index, rng: random.Random, n: int) -> list[dict]:
    keys = [k for k in ix.clean if ix.targets(k, "HAS_AGE")]
    out = []
    for k in rng.sample(keys, n):
        ages = sorted(r.split(":", 1)[1] for r in ix.targets(k, "HAS_AGE"))
        out.append(_q(f"age-{k}", "age", f"What is the geologic age of the {ix.name(k)}?",
                      ages, "all", k, "asud",
                      reference="; ".join(ix.g.units[k]["age_text"])))
    return out


def gen_states(ix: Index, rng: random.Random, n: int) -> list[dict]:
    keys = [k for k in ix.clean if 1 <= len(ix.g.units[k]["states"]) <= 4
            and all(s in ASK_STATES for s in ix.g.units[k]["states"])]
    return [_q(f"states-{k}", "states",
               f"In which Australian states or territories does the {ix.name(k)} occur?",
               [ASK_STATES[s] for s in ix.g.units[k]["states"]], "all", k, "asud")
            for k in rng.sample(keys, n)]


def gen_parent(ix: Index, rng: random.Random, n: int) -> list[dict]:
    keys = [k for k in ix.clean if ix.g.units[k]["rank"] in {"Member", "Bed"}
            and ix.targets(k, "PART_OF")]
    return [_q(f"parent-{k}", "parent", f"What larger stratigraphic unit is the {ix.name(k)} part of?",
               sorted({ix.name(p) for p in ix.targets(k, "PART_OF")}), "any", k, "asud")
            for k in rng.sample(keys, n)]


def gen_members(ix: Index, rng: random.Random, n: int) -> list[dict]:
    keys = [k for k in ix.clean
            if 3 <= len(set(ix.targets(k, "PART_OF", reverse=True))) <= 12]
    return [_q(f"members-{k}", "members", f"Which units are part of the {ix.name(k)}?",
               sorted({ix.name(c) for c in ix.targets(k, "PART_OF", reverse=True)}), "all", k,
               "asud")
            for k in rng.sample(keys, n)]


def gen_relation(ix: Index, rng: random.Random, n: int) -> list[dict]:
    def curated_above(k: str) -> list[str]:
        return sorted({ix.name(e["src"]) for e in ix.inc[k].get("OVERLIES", [])
                       if "asud" in e["props"].get("sources", [])})

    keys = [k for k in ix.clean if curated_above(k)]
    return [_q(f"relation-{k}", "relation", f"Which unit overlies the {ix.name(k)}?",
               curated_above(k), "any", k, "asud")
            for k in rng.sample(keys, n)]


def gen_multi_hop(ix: Index, rng: random.Random, n: int) -> list[dict]:
    groups: dict[tuple, set[str]] = defaultdict(set)
    for k, u in ix.g.units.items():
        if not u["asud"]:
            continue
        # curated lithology only: text-derived tags are noisier than the gold should be
        liths = {e["dst"].split(":", 1)[1] for e in ix.out[k].get("HAS_LITHOLOGY", [])
                 if "asud" in e["props"].get("sources", [])}
        for period in ix.periods(k):
            for st in u["states"]:
                for lith in liths:
                    groups[(period, st, lith)].add(k)
    combos = sorted(c for c, ks in groups.items() if 2 <= len(ks) <= 8 and c[1] in ASK_STATES)
    out = []
    for period, st, lith in rng.sample(combos, n):
        ks = groups[(period, st, lith)]
        out.append(_q(f"multi_hop-{period}-{st}-{lith}", "multi_hop",
                      f"Which {period} units in {ASK_STATES[st]} consist of {lith}?",
                      sorted({ix.name(k) for k in ks}), "all", f"{period}|{st}|{lith}", "asud"))
    return out


def gen_count(ix: Index, rng: random.Random, n: int) -> list[dict]:
    groups: dict[tuple, set[str]] = defaultdict(set)
    for k, u in ix.g.units.items():
        if u["asud"]:
            for period in ix.periods(k):
                for st in u["states"]:
                    groups[(period, st)].add(k)
    combos = sorted(c for c, ks in groups.items() if 5 <= len(ks) <= 80 and c[1] in ASK_STATES)
    return [_q(f"count-{p}-{s}", "count", f"How many {p} units occur in {ASK_STATES[s]}?",
               len(groups[(p, s)]), "count", f"{p}|{s}", "asud")
            for p, s in rng.sample(combos, n)]


DESCRIPTIVE_SYSTEM = """You write exam questions about geology passages.
Given a passage about a named geologic unit, write ONE question that:
- names the unit exactly as given,
- is answered by a specific detail stated in the passage (fossils, type locality,
  composition, what it was named after, who named it, where it is exposed),
- can be answered in a short phrase,
- is NOT about the unit's age, states, thickness, or what it overlies/underlies
  (those vary between reports, so one passage cannot be the gold answer).
Reply with JSON only: {"question": "...", "answer": "..."}"""


def gen_descriptive(ix: Index, rng: random.Random, n: int, cfg: Config) -> list[dict]:
    pool = [p for k in ix.clean for p in ix.passages_of[k] if 300 <= len(p["text"]) <= 1500]
    rng.shuffle(pool)
    out = []
    for p in track(pool, description="descriptive questions", total=n):
        if len(out) >= n:
            break
        text, _ = complete(cfg.llm, DESCRIPTIVE_SYSTEM,
                           f"Unit: {p['unit_name']}\nPassage: {p['text']}")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            qa = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            qa = None
        if not qa or not qa.get("question") or not qa.get("answer"):
            continue
        if p["unit_name"].split()[0].lower() not in qa["question"].lower():
            continue  # a question that does not name its unit is unanswerable by design
        out.append(_q(f"descriptive-{p['id']}", "descriptive", qa["question"].strip(),
                      qa["answer"].strip(), "judge", p["id"], "llm-generated from one passage"))
    return out


def build(cfg: Config, graph_path: Path, out_path: Path) -> list[dict]:
    g = Graph.load(graph_path)
    ix = Index(g)
    rng = random.Random(cfg.bench.seed)
    n = cfg.bench.per_category
    qs: list[dict] = []
    for gen in (gen_age, gen_states, gen_parent, gen_members, gen_relation, gen_multi_hop,
                gen_count):
        qs += gen(ix, rng, n)
    qs += gen_descriptive(ix, rng, n, cfg)
    out_path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
    return qs
