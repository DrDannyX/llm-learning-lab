"""Build the benchmark: questions with known answers, in eight categories.

WHY EIGHT CATEGORIES, NOT ONE SCORE
-----------------------------------
The three systems are good at different things, and a single averaged number
hides exactly the intuition this lab is for. The categories are chosen to pull
them apart:

  category    example                                            gold from   favours
  ----------- -------------------------------------------------- ----------- --------
  age         What is the geologic age of the Aarde Shale Member? geolex      all
  states      In which US states does X occur?                   geolex      graph
  parent      What larger unit is X part of?                     geolex      all
  members     Which units are part of the Y Group?               geolex      graph
  relation    Which unit overlies X?                             rule labels text
  multi_hop   Which Cretaceous units in Texas contain chalk?     geolex      graph
  count       How many Pennsylvanian units occur in Kansas?      geolex      KG only
  descriptive What fossils occur in X? (LLM-written, 1 passage)  LLM         text

THE CIRCULARITY CAVEAT -- read before quoting any number
-------------------------------------------------------
Seven categories take their gold answers from the same curated metadata the
knowledge graph was built from. On those, the KG is being asked to read back
its own contents, and it will look better than it would against a truly
independent test. `descriptive` is the counterweight: its answers exist only
in prose. `relation` gold comes from geo-sft's rule labeller -- the labeller
that lab found to be wrong on 83% of reviewed rows -- so treat it as noisy.

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

from ..config import Config
from ..ingest.extract import STATE_NAMES, Graph, core_name
from ..llm import complete

CATEGORIES = ["age", "states", "parent", "members", "relation", "multi_hop", "count", "descriptive"]

#: Period-level names a geologist would actually ask about.
PERIODS = ["Cambrian", "Ordovician", "Silurian", "Devonian", "Mississippian", "Pennsylvanian",
           "Permian", "Triassic", "Jurassic", "Cretaceous", "Tertiary", "Quaternary",
           "Precambrian"]


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
        cores = Counter(core_name(u["name"]) for u in g.units.values() if u["geolex"])
        #: units whose name identifies them: no homonym, and a rank word to read naturally
        self.clean = [k for k, u in g.units.items() if u["geolex"]
                      and cores[core_name(u["name"])] == 1
                      and u["rank"] != "Unknown" and u["full_name"] != u["name"]]
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
                      ages, "all", k, "geolex",
                      reference="; ".join(ix.g.units[k]["age_text"])))
    return out


def gen_states(ix: Index, rng: random.Random, n: int) -> list[dict]:
    keys = [k for k in ix.clean if 1 <= len(ix.g.units[k]["states"]) <= 4
            and all(s in STATE_NAMES for s in ix.g.units[k]["states"])]
    return [_q(f"states-{k}", "states", f"In which US states does the {ix.name(k)} occur?",
               [STATE_NAMES[s] for s in ix.g.units[k]["states"]], "all", k, "geolex")
            for k in rng.sample(keys, n)]


def gen_parent(ix: Index, rng: random.Random, n: int) -> list[dict]:
    keys = [k for k in ix.clean if ix.g.units[k]["rank"] in {"Member", "Bed", "Tongue", "Lentil"}
            and ix.targets(k, "PART_OF")]
    return [_q(f"parent-{k}", "parent", f"What larger stratigraphic unit is the {ix.name(k)} part of?",
               sorted({ix.name(p) for p in ix.targets(k, "PART_OF")}), "any", k, "geolex")
            for k in rng.sample(keys, n)]


def gen_members(ix: Index, rng: random.Random, n: int) -> list[dict]:
    keys = [k for k in ix.clean
            if 3 <= len(set(ix.targets(k, "PART_OF", reverse=True))) <= 12]
    return [_q(f"members-{k}", "members", f"Which units are part of the {ix.name(k)}?",
               sorted({ix.name(c) for c in ix.targets(k, "PART_OF", reverse=True)}), "all", k,
               "geolex")
            for k in rng.sample(keys, n)]


def gen_relation(ix: Index, rng: random.Random, n: int) -> list[dict]:
    out = []
    keys = [k for k in ix.clean if ix.targets(k, "OVERLIES", reverse=True)]
    for k in rng.sample(keys, n):
        above = sorted({ix.name(a) for a in ix.targets(k, "OVERLIES", reverse=True)})
        out.append(_q(f"relation-{k}", "relation", f"Which unit overlies the {ix.name(k)}?",
                      above, "any", k, "text (rule labeller)"))
    return out


def gen_multi_hop(ix: Index, rng: random.Random, n: int) -> list[dict]:
    groups: dict[tuple, set[str]] = defaultdict(set)
    for k, u in ix.g.units.items():
        if not u["geolex"]:
            continue
        # curated lithology only: text-derived tags are noisier than the gold should be
        liths = {e["dst"].split(":", 1)[1] for e in ix.out[k].get("HAS_LITHOLOGY", [])
                 if "geolex" in e["props"].get("sources", [])}
        for period in ix.periods(k):
            for st in u["states"]:
                for lith in liths:
                    groups[(period, st, lith)].add(k)
    combos = sorted(c for c, ks in groups.items() if 2 <= len(ks) <= 8 and c[1] in STATE_NAMES)
    out = []
    for period, st, lith in rng.sample(combos, n):
        ks = groups[(period, st, lith)]
        out.append(_q(f"multi_hop-{period}-{st}-{lith}", "multi_hop",
                      f"Which {period} units in {STATE_NAMES[st]} consist of {lith}?",
                      sorted({ix.name(k) for k in ks}), "all", f"{period}|{st}|{lith}", "geolex"))
    return out


def gen_count(ix: Index, rng: random.Random, n: int) -> list[dict]:
    groups: dict[tuple, set[str]] = defaultdict(set)
    for k, u in ix.g.units.items():
        if u["geolex"]:
            for period in ix.periods(k):
                for st in u["states"]:
                    groups[(period, st)].add(k)
    combos = sorted(c for c, ks in groups.items() if 5 <= len(ks) <= 80 and c[1] in STATE_NAMES)
    return [_q(f"count-{p}-{s}", "count", f"How many {p} units occur in {STATE_NAMES[s]}?",
               len(groups[(p, s)]), "count", f"{p}|{s}", "geolex")
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
