"""Vector RAG: embed the question, return the k nearest passages.

The baseline everyone builds first. It knows nothing about units, ages or
hierarchy -- only that some passages *sound like* the question. Strong on
"what does the text say about X"; blind to anything that requires combining
facts from passages it did not retrieve (lists, counts, multi-hop).
"""
from __future__ import annotations

import time

from .. import db
from ..config import Config
from ..llm import embedder
from .base import Retrieval

QUERY = f"""
CALL db.index.vector.queryNodes('{db.VECTOR_INDEX}', $k, $v) YIELD node, score
RETURN node.id AS id, node.unit_name AS unit, node.year AS year, node.text AS text, score
"""


def search(cfg: Config, question: str, k: int) -> list[dict]:
    v = embedder(cfg.llm.model_dump_json()).query(question)
    return db.read(cfg.neo4j, QUERY, k=k, v=v)


def format_passages(rows: list[dict]) -> str:
    return "\n\n".join(
        f"[{r['id']}] {r['unit']} ({r['year'] or 'n.d.'}): {r['text']}" for r in rows
    )


def retrieve(cfg: Config, question: str) -> Retrieval:
    t = time.perf_counter()
    rows = search(cfg, question, cfg.retrieval.rag_k)
    return Retrieval(
        system="rag",
        context=format_passages(rows),
        sources=[r["id"] for r in rows],
        debug={"scores": {r["id"]: round(r["score"], 4) for r in rows}},
        seconds=time.perf_counter() - t,
    )
