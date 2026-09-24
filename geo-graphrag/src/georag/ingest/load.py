"""graph.json + embeddings -> Neo4j.

One database holds everything: the property graph that the KG retriever
queries, the passage text that RAG retrieves, and the vector index that finds
it. GraphRAG is then just a query that crosses from one to the other -- which
is the whole idea, and why Neo4j rather than a separate vector store.

Writes are batched with UNWIND. Row-at-a-time MERGE over ~60k edges is the
classic slow-ingest mistake: one network round trip per fact.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from rich.progress import track

from .. import db
from ..config import Config
from ..llm import Embedder
from ..paths import EMBED_CACHE, EMBED_IDS
from .extract import STATE_NAMES, Graph

BATCH = 2000

SCHEMA = [
    "CREATE CONSTRAINT unit_key IF NOT EXISTS FOR (u:Unit) REQUIRE u.key IS UNIQUE",
    "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT interval_name IF NOT EXISTS FOR (i:Interval) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT lithology_name IF NOT EXISTS FOR (l:Lithology) REQUIRE l.name IS UNIQUE",
    "CREATE CONSTRAINT mineral_name IF NOT EXISTS FOR (m:Mineral) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT state_code IF NOT EXISTS FOR (s:State) REQUIRE s.code IS UNIQUE",
    "CREATE CONSTRAINT province_name IF NOT EXISTS FOR (p:Province) REQUIRE p.name IS UNIQUE",
    "CREATE INDEX unit_name IF NOT EXISTS FOR (u:Unit) ON (u.name)",
    (f"CREATE FULLTEXT INDEX {db.FULLTEXT_INDEX} IF NOT EXISTS "
     "FOR (u:Unit) ON EACH [u.name, u.full_name]"),
]

#: node key prefix in graph.json -> (label, key property)
LABELS = {
    "geolex": ("Unit", "key"), "name": ("Unit", "key"), "passage": ("Passage", "id"),
    "interval": ("Interval", "name"), "lithology": ("Lithology", "name"),
    "mineral": ("Mineral", "name"), "state": ("State", "code"), "province": ("Province", "name"),
}


def _node(ref: str) -> tuple[str, str, str]:
    prefix, _, value = ref.partition(":")
    label, prop = LABELS[prefix]
    return label, prop, (ref if label == "Unit" else value)


def _batches(rows: list, n: int = BATCH):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def reset(cfg: Config) -> None:
    """Drop everything. Batched, because one giant DETACH DELETE can exhaust heap.

    `CALL {...} IN TRANSACTIONS` only runs in an auto-commit transaction, so this
    goes through `session.run`, not the managed `execute_query` used elsewhere.
    """
    with db.driver(cfg.neo4j).session(database=cfg.neo4j.database) as s:
        s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()


def load_graph(cfg: Config, g: Graph) -> None:
    n4 = cfg.neo4j
    for stmt in SCHEMA:
        db.write(n4, stmt)

    for rows in _batches(list(g.units.values())):
        db.write(n4, """
            UNWIND $rows AS r
            MERGE (u:Unit {key: r.key})
            SET u.name = r.name, u.full_name = r.full_name, u.rank = r.rank,
                u.geolex = r.geolex, u.unit_id = r.unit_id, u.url = r.url,
                u.aliases = r.aliases, u.age_text = r.age_text,
                u.thickness_min_m = r.thickness_min_m, u.thickness_max_m = r.thickness_max_m
        """, rows=rows)

    for rows in _batches(g.passages):
        db.write(n4, """
            UNWIND $rows AS r
            MERGE (p:Passage {id: r.id})
            SET p.text = r.text, p.year = r.year, p.publication = r.publication,
                p.url = r.url, p.unit_name = r.unit_name
        """, rows=rows)

    db.write(n4, """
        UNWIND $rows AS r
        MERGE (i:Interval {name: r.name})
        SET i.rank = r.rank, i.top_ma = toFloat(r.t_age), i.base_ma = toFloat(r.b_age)
    """, rows=g.intervals)

    # leaf nodes are created on demand by their edges; states get a full name
    db.write(n4, "UNWIND $rows AS r MERGE (s:State {code: r.code}) SET s.name = r.name",
             rows=[{"code": c, "name": n} for c, n in STATE_NAMES.items()])

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for e in g.edges:
        sl, sp, sv = _node(e["src"])
        dl, dp, dv = _node(e["dst"])
        grouped[(e["type"], sl, sp, dl, dp)].append({"s": sv, "d": dv, "props": e["props"]})

    for (type_, sl, sp, dl, dp), rows in track(grouped.items(), description="edges"):
        # labels and types cannot be parameters in Cypher; they come from our
        # own LABELS table and edge types, never from user input
        q = f"""
            UNWIND $rows AS r
            MERGE (a:{sl} {{{sp}: r.s}})
            MERGE (b:{dl} {{{dp}: r.d}})
            MERGE (a)-[e:{type_}]->(b)
            SET e += r.props
        """
        for batch in _batches(rows):
            db.write(n4, q, rows=batch)


def embed_passages(cfg: Config, g: Graph) -> np.ndarray:
    """Embed every passage, reusing the on-disk cache when ids and model match.

    Each passage is embedded with its unit's name prepended ("contextual chunk
    headers"). Plenty of Geolex passages never name their own unit -- "Consists
    of erratic development of sandstones..." -- so without the header the
    vector has no idea what the passage is about.
    """
    ids = [p["id"] for p in g.passages]
    meta = {"model": cfg.llm.embed_model, "ids": ids}
    if EMBED_CACHE.exists() and EMBED_IDS.exists() and json.loads(EMBED_IDS.read_text()) == meta:
        return np.load(EMBED_CACHE)

    emb = Embedder(cfg.llm)
    texts = [f"{p['unit_name']}. {p['text']}" for p in g.passages]
    b = cfg.ingest.embed_batch
    out = [emb.documents(texts[i:i + b])
           for i in track(range(0, len(texts), b), description="embedding")]
    vecs = np.concatenate(out)
    np.save(EMBED_CACHE, vecs)
    EMBED_IDS.write_text(json.dumps(meta))
    return vecs


def load_embeddings(cfg: Config, g: Graph, vecs: np.ndarray) -> None:
    n4 = cfg.neo4j
    db.write(n4, f"""
        CREATE VECTOR INDEX {db.VECTOR_INDEX} IF NOT EXISTS
        FOR (p:Passage) ON (p.embedding)
        OPTIONS {{indexConfig: {{`vector.dimensions`: {cfg.llm.embed_dim},
                                 `vector.similarity_function`: 'cosine'}}}}
    """)
    rows = [{"id": p["id"], "v": v.tolist()} for p, v in zip(g.passages, vecs)]
    for batch in track(list(_batches(rows, 500)), description="vectors"):
        db.write(n4, """
            UNWIND $rows AS r
            MATCH (p:Passage {id: r.id})
            CALL db.create.setNodeVectorProperty(p, 'embedding', r.v)
        """, rows=batch)
    db.write(n4, "CALL db.awaitIndexes(300)")


def counts(cfg: Config) -> dict:
    nodes = db.read(cfg.neo4j, "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC")
    rels = db.read(cfg.neo4j, "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC")
    return {"nodes": {r["label"]: r["n"] for r in nodes}, "relationships": {r["type"]: r["n"] for r in rels}}
