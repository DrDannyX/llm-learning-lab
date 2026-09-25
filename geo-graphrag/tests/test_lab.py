"""Tests that EXECUTE the code rather than inspect it (see geo-cpt RESULTS §9 for
what a source-grepping regression test once let through).

Default tier needs no services. `-m live` needs Neo4j (ingested) and LM Studio.
"""
from __future__ import annotations

import re

import pytest

from georag import db, paths
from georag.bench import score
from georag.config import Config
from georag.ingest import extract
from georag.retrievers import kg
from georag.retrievers.link import Linker
from georag.retrievers.pipeline import ANSWER_SYSTEM, SYSTEMS

# --- names --------------------------------------------------------------------

@pytest.mark.parametrize("raw, core", [
    ("Alsace Quartzite", "alsace"),
    ("Bulgonunna Volcanics", "bulgonunna"),     # plural rock word, not "volcanic"
    ("Mount Isa Group", "mount isa"),
    ("Nirranda Gp", "nirranda"),                # ASUD house abbreviation
    ("Mount Walsh Granite (G4)", "mount walsh"),
    ("Limestone", "limestone"),                 # never strip a name down to nothing
])
def test_core_name(raw, core):
    assert extract.core_name(raw) == core


@pytest.mark.parametrize("phrase, rank", [
    ("Pickwick Metabasalt Member", "Member"), ("Alsace Quartzite", "Formation"),
    ("Mount Isa Group", "Group"), ("Babel Island Suite", "Suite"),
    ("Moolayember Beds", "Formation"), ("Glyde Hill Volcanic Complex", "Unknown"),
])
def test_rank_of(phrase, rank):
    assert extract.rank_of(phrase) == rank


@pytest.mark.parametrize("asud_rank, name, rank", [
    ("Group, Suite", "Mount Isa Group", "Group"), ("Group, Suite", "Babel Island Suite", "Suite"),
    ("Supergroup", "Warakurna Supersuite", "Supersuite"), ("Formation, beds", "Alsace Quartzite", "Formation"),
])
def test_schema_rank(asud_rank, name, rank):
    assert extract.schema_rank(asud_rank, name) == rank


# --- the timescale --------------------------------------------------------------

@pytest.fixture(scope="module")
def graph():
    if not paths.PASSAGES.exists():
        pytest.skip("geo-sft corpus not built")
    return extract.build(paths.PASSAGES, paths.VOCAB, limit_units=400, seed=1)


def test_interval_containment():
    vocab = {"chronostrat": [
        {"name": "Cretaceous", "rank": "period", "t_age": 66, "b_age": 143.1},
        {"name": "Late Cretaceous", "rank": "epoch", "t_age": 66, "b_age": 100.5},
        {"name": "Cenomanian", "rank": "age", "t_age": 93.9, "b_age": 100.5},
        {"name": "Mesozoic", "rank": "era", "t_age": 66, "b_age": 251.9},
    ]}
    _, within = extract.build_intervals(vocab)
    assert ("Cenomanian", "Late Cretaceous") in within
    assert ("Late Cretaceous", "Cretaceous") in within
    assert ("Cretaceous", "Mesozoic") in within
    # only the NEXT populated rank: no shortcut edges
    assert ("Cenomanian", "Cretaceous") not in within


def test_build_invariants(graph):
    refs = {"asud", "name", "passage", "interval", "lithology", "mineral", "state", "province"}
    for e in graph.edges:
        assert e["src"].split(":", 1)[0] in refs and e["dst"].split(":", 1)[0] in refs
        assert e["src"] != e["dst"], "self-loops mean resolution matched a unit to itself"
    # one edge per (type, src, dst): evidence merges, it does not duplicate
    keys = [(e["type"], e["src"], e["dst"]) for e in graph.edges]
    assert len(keys) == len(set(keys))
    # there is no UNDERLIES: every relation is stored in canonical direction
    assert not any(e["type"] == "UNDERLIES" for e in graph.edges)
    # ages come from curated metadata only
    assert all(e["props"]["sources"] == ["asud"] for e in graph.edges if e["type"] == "HAS_AGE")
    # there is no INTRUDED_BY either
    assert not any(e["type"] == "INTRUDED_BY" for e in graph.edges)


def test_every_passage_describes_an_asud_unit(graph):
    described = {e["src"] for e in graph.edges if e["type"] == "DESCRIBES"}
    assert described == {f"passage:{p['id']}" for p in graph.passages}
    assert all(graph.units[p["unit_key"]]["asud"] for p in graph.passages)
    assert all(graph.units[p["unit_key"]]["has_text"] for p in graph.passages)


def test_curated_relations_are_in_the_graph(graph):
    """ASUD curates relations; most OVERLIES edges should carry an asud source."""
    over = [e for e in graph.edges if e["type"] == "OVERLIES"]
    assert over and sum("asud" in e["props"]["sources"] for e in over) > len(over) / 2


def _unit(key, name, states, rank="Formation"):
    return {"key": key, "name": name, "full_name": name, "rank": rank, "asud": True,
            "states": states}


def test_resolver_breaks_homonyms_by_name_then_state():
    g = extract.Graph()
    g.units["asud:1"] = _unit("asud:1", "Murchison Granite", ["TAS"])
    g.units["asud:2"] = _unit("asud:2", "Murchison Volcanics", ["TAS"])
    g.units["asud:3"] = _unit("asud:3", "Ada Formation", ["QLD"])
    g.units["asud:4"] = _unit("asud:4", "Ada Formation", ["WA"])
    r = extract.Resolver(g)
    assert r.resolve("Murchison Granite", ["TAS"]) == "asud:1"    # same core, exact name wins
    assert r.resolve("Ada Formation", ["WA"]) == "asud:4"
    assert r.resolve("Ada Formation", ["QLD"]) == "asud:3"
    assert r.resolve("Nowhere Shale", ["QLD"]) == "name:nowhere"
    assert g.units["name:nowhere"]["asud"] is False


# --- entity linking, without a database ------------------------------------------

@pytest.fixture
def linker(monkeypatch):
    units = [
        {"key": "asud:1", "name": "Alsace Quartzite", "full_name": "Alsace Quartzite",
         "rank": "Formation", "aliases": [], "asud": True, "states": ["QLD"]},
        {"key": "asud:2", "name": "Ada Formation", "full_name": "Ada Formation", "rank": "Formation",
         "aliases": [], "asud": True, "states": ["QLD"]},
        {"key": "asud:3", "name": "Ada Formation", "full_name": "Ada Formation", "rank": "Formation",
         "aliases": [], "asud": True, "states": ["WA"]},
        {"key": "asud:4", "name": "Red Member", "full_name": "Red Member", "rank": "Member",
         "aliases": [], "asud": True, "states": ["SA"]},
    ]

    def fake_read(cfg, query, **_):
        if "MATCH (u:Unit)" in query:
            return units
        if "Interval" in query:
            return [{"n": "Cambrian"}, {"n": "Late Cambrian"}]
        return [{"n": "shale"}, {"n": "limestone"}]

    monkeypatch.setattr(db, "read", fake_read)
    return Linker(None)


def test_link_absorbs_rank_word(linker):
    got = linker.link("What overlies the Alsace quartzites?")
    assert [u["key"] for u in got.units] == ["asud:1"]
    assert got.lithologies == []  # "quartzites" is part of the name here, not a filter


def test_link_homonym_uses_state(linker):
    assert [u["key"] for u in linker.link("Age of the Ada Formation in Western Australia?").units] == ["asud:3"]


def test_link_requires_capital_for_units(linker):
    got = linker.link("is there a red limestone unit in the Late Cambrian of Tas")
    assert got.units == []
    assert got.intervals == ["Late Cambrian"] and got.states == ["TAS"] and got.lithologies == ["limestone"]


# --- text-to-Cypher plumbing ---------------------------------------------------

def test_extract_cypher():
    assert kg.extract_cypher("Sure!\n```cypher\nMATCH (n) RETURN n;\n```") == "MATCH (n) RETURN n"
    assert kg.extract_cypher("MATCH (n) RETURN n") == "MATCH (n) RETURN n"


@pytest.mark.parametrize("q", [
    "MATCH (n) DETACH DELETE n", "CREATE (n:Unit)", "MATCH (n) SET n.x = 1",
    "MERGE (n:Unit {key: 'x'})", "LOAD CSV FROM 'file:///x' AS r RETURN r",
])
def test_write_guard(q):
    with pytest.raises(ValueError):
        db.read_untrusted(None, q)


def test_answer_prompt_is_shared(monkeypatch):
    """The controlled variable: every system's context goes to the same model with
    the same system prompt, and only the context differs. Executed, not grepped."""
    from georag.retrievers import pipeline
    from georag.retrievers.base import Retrieval

    calls = []
    monkeypatch.setattr(pipeline, "complete",
                        lambda cfg, system, prompt: (calls.append((system, prompt)) or "ok", {}))
    for name in SYSTEMS:
        monkeypatch.setitem(pipeline.SYSTEMS, name,
                            lambda cfg, q, n=name: Retrieval(system=n, context=f"ctx-{n}"))
    for name in SYSTEMS:
        pipeline.answer(Config(), name, "Q?")

    assert {system for system, _ in calls} == {ANSWER_SYSTEM}
    assert [p for _, p in calls] == [f"Context:\nctx-{n}\n\nQuestion: Q?" for n in SYSTEMS]
    assert set(SYSTEMS) == {"rag", "kg", "graphrag"}


# --- scoring --------------------------------------------------------------------

def test_mentions_core_names_and_state_codes():
    assert score.mentions("It is overlain by the Bortala formation.", "Bortala Formation")
    assert score.mentions("Found in QLD and the NT.", "Queensland")
    assert not score.mentions("Found in Queensland.", "Tasmania")
    assert score.mentions("Palaeoproterozoic in age.", "Paleoproterozoic")
    assert not score.mentions("the Adamstown Formation", "Ada Formation")  # word boundaries


def test_match_modes():
    q_all = {"match": "all", "gold": ["Statherian", "Calymmian"]}
    assert score.match(q_all, "Late Statherian") == 0.5
    assert score.match({"match": "all", "gold": ["Late Paleocene"]}, "top of the Paleocene") == 1.0
    q_any = {"match": "any", "gold": ["Mount Isa Group", "Ada Formation"]}
    assert score.match(q_any, "It is part of the Mount Isa.") == 1.0
    q_n = {"match": "count", "gold": 41}
    assert score.match(q_n, "41 units.") == 1.0
    assert score.match(q_n, "There are 7 units, not 41.") == 0.0  # the number must lead
    assert score.match({"match": "judge", "gold": "x"}, "x") is None


def test_bench_generators_are_data_derived(graph):
    import random

    from georag.bench import build as b

    ix = b.Index(graph)
    rng = random.Random(0)
    for gen in (b.gen_age, b.gen_states, b.gen_parent, b.gen_relation, b.gen_count):
        for q in gen(ix, rng, 2):
            assert q["gold"] not in ([], None, 0)
            assert re.search(r"\?$", q["question"])


# --- live tier -------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    from georag import config

    c = config.load()
    try:
        db.ping(c.neo4j)
    except Exception:  # noqa: BLE001
        pytest.skip("Neo4j not running")
    return c


@pytest.mark.live
def test_few_shot_examples_execute(cfg):
    """Every Cypher example in the text-to-Cypher prompt must run and return rows.
    A broken example teaches the model a broken query."""
    blocks = [b.strip() for b in kg.EXAMPLES.split("\n\n")]
    for block in blocks:
        cypher = "\n".join(line for line in block.splitlines() if not line.startswith("Q:"))
        rows = db.read_untrusted(cfg.neo4j, cypher)
        assert rows, f"example returned no rows:\n{cypher}"


@pytest.mark.live
def test_vector_search_finds_own_passage(cfg):
    from georag.retrievers import rag

    p = db.read(cfg.neo4j, "MATCH (p:Passage {id: '332:0'}) RETURN p.text AS t")[0]["t"]
    ids = [r["id"] for r in rag.search(cfg, p[:300], 5)]
    assert "332:0" in ids


# --- web UI -----------------------------------------------------------------------

def test_web_serves_page_and_validates():
    from starlette.testclient import TestClient

    from georag.web.app import app

    c = TestClient(app)
    assert "<title>geo-graphrag</title>" in c.get("/").text
    meta = c.get("/api/meta").json()
    assert set(meta["systems"]) == set(SYSTEMS)
    assert meta["examples"] and all(g["title"] and g["hint"] and g["questions"]
                                    for g in meta["examples"])
    assert 'id="about"' in c.get("/").text
    assert c.post("/api/ask", json={"question": "", "system": "rag"}).status_code == 400
    assert c.post("/api/ask", json={"question": "Q?", "system": "nope"}).status_code == 400
