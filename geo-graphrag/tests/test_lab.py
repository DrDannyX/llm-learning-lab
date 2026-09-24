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

# --- names and usages ----------------------------------------------------------

@pytest.mark.parametrize("raw, core", [
    ("Eagle Ford Clay", "eagle ford"),
    ("Howard Limestone", "howard"),
    ("Kansas City Group", "kansas city"),
    ("/Sacfox subgroup [informal]", "sacfox"),
    ("Aarde Shale Member", "aarde"),
    ("Limestone", "limestone"),          # never strip a name down to nothing
])
def test_core_name(raw, core):
    assert extract.core_name(raw) == core


def test_parse_usage_splits_of_and_in():
    assert extract.parse_usage(
        "Bloomfield limestone in Glenshaw Formation of Conemaugh Group", "Bloomfield"
    ) == ["Bloomfield limestone", "Glenshaw Formation", "Conemaugh Group"]


def test_parse_usage_rejects_notes():
    assert extract.parse_usage("Misspelled Critizer in early reports.", "Critzer") is None


@pytest.mark.parametrize("phrase, rank", [
    ("Aarde Shale Member", "Member"), ("Howard Limestone", "Formation"),
    ("Wabaunsee Group", "Group"), ("Sacfox subgroup [informal]", "Subgroup"), ("Foo", "Unknown"),
])
def test_rank_of(phrase, rank):
    assert extract.rank_of(phrase) == rank


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
    refs = {"geolex", "name", "passage", "interval", "lithology", "mineral", "state", "province"}
    for e in graph.edges:
        assert e["src"].split(":", 1)[0] in refs and e["dst"].split(":", 1)[0] in refs
        assert e["src"] != e["dst"], "self-loops mean resolution matched a unit to itself"
    # one edge per (type, src, dst): evidence merges, it does not duplicate
    keys = [(e["type"], e["src"], e["dst"]) for e in graph.edges]
    assert len(keys) == len(set(keys))
    # there is no UNDERLIES: every relation is stored in canonical direction
    assert not any(e["type"] == "UNDERLIES" for e in graph.edges)
    # ages come from curated metadata only
    assert all(e["props"]["sources"] == ["geolex"] for e in graph.edges if e["type"] == "HAS_AGE")


def test_every_passage_describes_a_geolex_unit(graph):
    described = {e["src"] for e in graph.edges if e["type"] == "DESCRIBES"}
    assert described == {f"passage:{p['id']}" for p in graph.passages}
    assert all(graph.units[p["unit_key"]]["geolex"] for p in graph.passages)


def test_resolver_breaks_homonyms_by_state():
    g = extract.Graph()
    for key, st in [("geolex:1", ["OK"]), ("geolex:2", ["VA"])]:
        g.units[key] = {"key": key, "name": "Ada", "geolex": True, "states": st}
    r = extract.Resolver(g)
    assert r.resolve("Ada Formation", ["VA"]) == "geolex:2"
    assert r.resolve("Ada Formation", ["OK"]) == "geolex:1"
    assert r.resolve("Nowhere Shale", ["OK"]) == "name:nowhere"
    assert g.units["name:nowhere"]["geolex"] is False


# --- entity linking, without a database ------------------------------------------

@pytest.fixture
def linker(monkeypatch):
    units = [
        {"key": "geolex:1", "name": "Eagle Ford", "full_name": "Eagle Ford Group", "rank": "Group",
         "aliases": ["Eagle Ford Shale"], "geolex": True, "states": ["TX"]},
        {"key": "geolex:2", "name": "Ada", "full_name": "Ada Formation", "rank": "Formation",
         "aliases": [], "geolex": True, "states": ["OK"]},
        {"key": "geolex:3", "name": "Ada", "full_name": "Ada Formation", "rank": "Formation",
         "aliases": [], "geolex": True, "states": ["VA"]},
        {"key": "geolex:4", "name": "Big", "full_name": "Big Member", "rank": "Member",
         "aliases": [], "geolex": True, "states": ["KS"]},
    ]

    def fake_read(cfg, query, **_):
        if "MATCH (u:Unit)" in query:
            return units
        if "Interval" in query:
            return [{"n": "Cretaceous"}, {"n": "Late Cretaceous"}]
        return [{"n": "shale"}, {"n": "chalk"}]

    monkeypatch.setattr(db, "read", fake_read)
    return Linker(None)


def test_link_absorbs_rank_word(linker):
    got = linker.link("What overlies the Eagle Ford Shale?")
    assert [u["key"] for u in got.units] == ["geolex:1"]
    assert got.lithologies == []  # "Shale" is part of the name here, not a filter


def test_link_homonym_uses_state(linker):
    assert [u["key"] for u in linker.link("Age of the Ada Formation in Virginia?").units] == ["geolex:3"]


def test_link_requires_capital_for_units(linker):
    got = linker.link("is there a big chalk unit in the Late Cretaceous of TX")
    assert got.units == []
    assert got.intervals == ["Late Cretaceous"] and got.states == ["TX"] and got.lithologies == ["chalk"]


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
    assert score.mentions("It is overlain by the Church limestone.", "Church Member")
    assert score.mentions("Found in TX and OK.", "Texas")
    assert not score.mentions("Found in Texas.", "Oklahoma")
    assert not score.mentions("the Adamstown Formation", "Ada Formation")  # word boundaries


def test_match_modes():
    q_all = {"match": "all", "gold": ["Pennsylvanian", "Virgilian"]}
    assert score.match(q_all, "Late Pennsylvanian") == 0.5
    assert score.match({"match": "all", "gold": ["Late Paleocene"]}, "top of the Paleocene") == 1.0
    q_any = {"match": "any", "gold": ["Howard Limestone", "Ada Formation"]}
    assert score.match(q_any, "It is part of the Howard.") == 1.0
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

    p = db.read(cfg.neo4j, "MATCH (p:Passage {id: '6304:0'}) RETURN p.text AS t")[0]["t"]
    ids = [r["id"] for r in rag.search(cfg, p[:300], 5)]
    assert "6304:0" in ids


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
