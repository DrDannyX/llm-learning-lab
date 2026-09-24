"""Neo4j access. One driver per process; reads are enforced read-only."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from neo4j import READ_ACCESS, Driver, GraphDatabase, unit_of_work

from .config import Neo4jConfig

VECTOR_INDEX = "passage_embedding"
FULLTEXT_INDEX = "unit_names"

#: Belt and braces on top of READ_ACCESS: the KG retriever runs model-written
#: Cypher, and a read-only session is what makes that safe, not this regex.
_WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+\{)\b", re.IGNORECASE)


@lru_cache(maxsize=4)
def _driver(uri: str, user: str, password: str) -> Driver:
    return GraphDatabase.driver(uri, auth=(user, password))


def driver(cfg: Neo4jConfig) -> Driver:
    return _driver(cfg.uri, cfg.user, cfg.password)


def write(cfg: Neo4jConfig, query: str, **params: Any) -> None:
    driver(cfg).execute_query(query, params, database_=cfg.database)


def read(cfg: Neo4jConfig, query: str, **params: Any) -> list[dict]:
    records, _, _ = driver(cfg).execute_query(
        query, params, database_=cfg.database, routing_="r",
    )
    return [r.data() for r in records]


def read_untrusted(cfg: Neo4jConfig, query: str, limit: int = 200,
                   timeout_s: float = 15.0) -> list[dict]:
    """Run model-written Cypher in a read-only transaction with a timeout."""
    if _WRITE.search(query):
        raise ValueError("write clauses are not allowed")

    @unit_of_work(timeout=timeout_s)
    def work(tx) -> list[dict]:
        out = []
        for rec in tx.run(query):
            out.append(rec.data())
            if len(out) >= limit:
                break
        return out

    with driver(cfg).session(database=cfg.database, default_access_mode=READ_ACCESS) as s:
        return s.execute_read(work)


def ping(cfg: Neo4jConfig) -> str:
    info = driver(cfg).get_server_info()
    return f"{info.agent} at {info.address}"
