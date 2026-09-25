"""GraphRAG, local-search style: vectors find the way in, the graph fills in
what the passages leave out.

  1. vector search -> entry passages (as in RAG, fewer of them)
  2. entity linking -> the units the question actually names
  3. seed units = linked units first, then the units the entry passages describe
  4. for each seed, a fixed Cypher "unit card": its ages *and their parent
     intervals*, lithologies, states, hierarchy, stratigraphic neighbours and
     intrusive relations
  5. for linked units the vector search missed, pull their best passages
     directly through the graph (DESCRIBES), ranked by similarity

Step 5 is the part plain RAG cannot do: the graph *routes* text retrieval to
the right entity even when the question's wording sits far from the passage's.

This is NOT text-to-Cypher. The Cypher here is fixed and written by a human,
so it never fails to parse -- but it also cannot count or filter across the
whole graph. That trade-off is the most useful thing to watch in the benchmark.
(Microsoft-style *global* GraphRAG, with community summaries, is experiment #6.)
"""
from __future__ import annotations

import time

from .. import db
from ..config import Config
from ..llm import embedder
from .base import Retrieval
from .link import linker
from .rag import QUERY as VECTOR_QUERY
from .rag import format_passages

CARD = """
MATCH (u:Unit {key: $key})
RETURN u.full_name AS unit, u.rank AS rank, u.age_text AS asud_age,
       u.thickness_min_m AS thickness_min_m, u.thickness_max_m AS thickness_max_m,
       [(u)-[:HAS_AGE]->(i) | i.name] AS ages,
       apoc_free_ancestors AS age_context,
       [(u)-[:HAS_LITHOLOGY]->(l) | l.name] AS lithologies,
       [(u)-[:HAS_MINERAL]->(m) | m.name][..10] AS minerals,
       [(u)-[:OCCURS_IN]->(s) | s.name] AS states,
       [(u)-[:IN_PROVINCE]->(p) | p.name] AS provinces,
       [(u)-[:PART_OF]->(p) | p.full_name + ' (' + p.rank + ')'] AS part_of,
       [(c)-[:PART_OF]->(u) | c.full_name + ' (' + c.rank + ')'][..30] AS contains,
       [(u)-[:OVERLIES]->(x) | x.full_name] AS overlies,
       [(x)-[:OVERLIES]->(u) | x.full_name] AS overlain_by,
       [(u)-[:INTRUDES]->(x) | x.full_name] AS intrudes,
       [(x)-[:INTRUDES]->(u) | x.full_name] AS intruded_by,
       [(u)-[:EQUIVALENT_TO|GRADES_INTO|INTERTONGUES_WITH]-(x) | x.full_name] AS laterally_related
""".replace(
    "apoc_free_ancestors",
    "[(u)-[:HAS_AGE]->(:Interval)-[:WITHIN*1..6]->(a) | a.name]",
)

UNIT_PASSAGES = """
MATCH (p:Passage)-[:DESCRIBES]->(:Unit {key: $key})
RETURN p.id AS id, p.unit_name AS unit, p.year AS year, p.text AS text,
       vector.similarity.cosine(p.embedding, $v) AS score
ORDER BY score DESC LIMIT $n
"""


def unit_card(cfg: Config, key: str) -> dict | None:
    rows = db.read(cfg.neo4j, CARD, key=key)
    if not rows:
        return None
    card = rows[0]
    card["age_context"] = sorted(set(card["age_context"]) - set(card["ages"]))
    # drop empty fields: the answer model reads every token we send
    return {k: v for k, v in card.items() if v not in (None, [], "")}


def format_card(card: dict) -> str:
    head = f"* {card.pop('unit')} ({card.pop('rank', 'Unknown')})"
    body = "; ".join(
        f"{k.replace('_', ' ')}: {', '.join(map(str, v)) if isinstance(v, list) else v}"
        for k, v in card.items()
    )
    return f"{head}: {body}"


def retrieve(cfg: Config, question: str) -> Retrieval:
    t = time.perf_counter()
    r = cfg.retrieval
    v = embedder(cfg.llm.model_dump_json()).query(question)
    entry = db.read(cfg.neo4j, VECTOR_QUERY, k=r.graphrag_k, v=v)

    linked = linker(cfg.neo4j).link(question)
    described = db.read(cfg.neo4j, """
        UNWIND $ids AS id MATCH (p:Passage {id: id})-[:DESCRIBES]->(u:Unit) RETURN u.key AS key
    """, ids=[e["id"] for e in entry])
    seeds = list(dict.fromkeys([u["key"] for u in linked.units] + [d["key"] for d in described]))
    seeds = seeds[: max(r.graphrag_max_units, len(linked.units))]

    cards = [c for k in seeds if (c := unit_card(cfg, k))]

    # graph-routed text: linked units whose passages vector search did not surface
    have = {e["id"] for e in entry}
    routed: list[dict] = []
    for u in linked.units:
        for p in db.read(cfg.neo4j, UNIT_PASSAGES, key=u["key"], v=v, n=2):
            if p["id"] not in have:
                routed.append(p)
                have.add(p["id"])

    parts = []
    if cards:
        parts.append("Graph facts:\n" + "\n".join(format_card(dict(c)) for c in cards))
    if routed:
        parts.append("Passages about the named units (found via the graph):\n" + format_passages(routed))
    if entry:
        parts.append("Passages similar to the question:\n" + format_passages(entry))

    return Retrieval(
        system="graphrag",
        context="\n\n".join(parts),
        sources=seeds + [e["id"] for e in entry] + [p["id"] for p in routed],
        debug={"linked": linked.describe(), "seed_units": seeds,
               "routed_passages": [p["id"] for p in routed]},
        seconds=time.perf_counter() - t,
    )
