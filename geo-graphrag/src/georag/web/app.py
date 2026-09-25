"""A small local web UI: one question, three answers side by side.

The page sends three independent requests, one per system, so the columns fill
in as each finishes. LM Studio serves several requests in parallel, which means
wall-clock time is roughly the slowest system rather than the sum. Latencies
shown are therefore measured *under contention* and run higher than the
benchmark's sequential medians.

    georag web            ->  http://localhost:8000
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from .. import config, db
from ..retrievers.pipeline import LABELS, SYSTEMS, answer

CFG = config.load()
INDEX = Path(__file__).with_name("index.html")

#: Example questions, grouped by what they reveal. Every unit named here is an
#: ASUD unit with passages in this corpus and resolves in the entity linker
#: (checked when these were written), so a wrong answer means a wrong
#: retrieval, not a typo.
EXAMPLES = [
    {
        "title": "Counting and filtering",
        "hint": "The answer is spread across dozens of units. Only the knowledge graph "
                "can count or filter the whole lexicon; the others count what they retrieved.",
        "questions": [
            "How many Permian units occur in New South Wales?",
            "How many Cambrian units occur in Tasmania?",
            "Which Neoarchean units in Western Australia consist of basalt?",
            "Which Devonian units in Victoria consist of granite?",
            "How many Paleoproterozoic units occur in the Northern Territory?",
            "Which Jurassic units in Queensland consist of sandstone?",
            "How many Cretaceous units occur in South Australia?",
            "Which Ordovician units in New South Wales consist of limestone?",
        ],
    },
    {
        "title": "Hierarchy and neighbours",
        "hint": "Graph relationships: what a unit is part of, what lies above it, what "
                "intrudes it. The knowledge graph and GraphRAG should both do well.",
        "questions": [
            "Which units are part of the Mount Isa Group?",
            "Which units make up the Hamersley Group?",
            "What larger stratigraphic unit is the Breakaway Shale part of?",
            "Which unit overlies the Alsace Quartzite?",
            "What overlies the Hawkesbury Sandstone?",
            "Which units are part of the Illawarra Coal Measures?",
            "What overlies the Fortescue Group?",
            "What larger unit is the Brockman Iron Formation part of?",
        ],
    },
    {
        "title": "Details in the text",
        "hint": "Only in the prose. The knowledge graph has no passages and should say it "
                "doesn't know; RAG and GraphRAG should find it.",
        "questions": [
            "What is the Mount Isa Group named after?",
            "Where is the type section of the Tumblagooda Sandstone?",
            "What trace fossils occur in the Tumblagooda Sandstone?",
            "Where is the type section of the Hutton Sandstone?",
            "What fossils does the Pertatataka Formation contain?",
            "Where is the type section of the Mereenie Sandstone?",
        ],
    },
    {
        "title": "Unit profiles",
        "hint": "Facts that are in both the graph and the text. Compare how complete and "
                "how precise each answer is.",
        "questions": [
            "What is the geologic age of the Hawkesbury Sandstone?",
            "In which Australian states does the Toolebuc Formation occur?",
            "Tell me about the Brockman Iron Formation.",
            "What is the geologic age of the Precipice Sandstone?",
            "In which states does the Kanmantoo Group occur?",
            "What rock types make up the Newcastle Coal Measures?",
            "Tell me about the Mereenie Sandstone.",
        ],
    },
]


def _evidence(a) -> dict:
    r = a.retrieval
    ev: dict = {"context": r.context, "sources": r.sources[:20]}
    if a.system == "kg":
        ev["cypher_attempts"] = r.debug.get("attempts", [])
    if a.system in ("kg", "graphrag"):
        ev["linked"] = r.debug.get("linked", "")
    if a.system == "graphrag":
        ev["seed_units"] = r.debug.get("seed_units", [])
        ev["routed_passages"] = r.debug.get("routed_passages", [])
    return ev


async def index(_: Request) -> FileResponse:
    return FileResponse(INDEX)


async def meta(_: Request) -> JSONResponse:
    return JSONResponse({"systems": LABELS, "examples": EXAMPLES, "model": CFG.llm.chat_model})


async def ask(request: Request) -> JSONResponse:
    body = await request.json()
    question = str(body.get("question", "")).strip()
    system = body.get("system")
    if not question or system not in SYSTEMS:
        return JSONResponse({"error": "need a question and a system"}, status_code=400)
    try:
        # the pipeline is synchronous (Neo4j driver, Strands); keep the event loop free
        a = await run_in_threadpool(answer, CFG, system, question)
    except Exception as e:  # noqa: BLE001 -- show the failure in its column
        return JSONResponse({"system": system, "error": f"{type(e).__name__}: {e}"}, status_code=500)
    return JSONResponse({
        "system": system,
        "answer": a.answer,
        "seconds": round(a.seconds, 1),
        "retrieval_seconds": round(a.retrieval.seconds, 1),
        "context_chars": len(a.retrieval.context),
        "error": a.retrieval.error,
        "evidence": _evidence(a),
    })


@lru_cache(maxsize=1)
def corpus_stats() -> dict:
    """What is in the database, measured rather than asserted. Cached: it only
    changes on re-ingest (restart the server afterwards)."""
    n4 = CFG.neo4j

    def one(q: str) -> dict:
        return db.read(n4, q)[0]

    units = one("""
        MATCH (u:Unit) RETURN sum(CASE WHEN u.asud THEN 1 ELSE 0 END) AS asud,
                              sum(CASE WHEN u.has_text THEN 1 ELSE 0 END) AS with_text,
                              sum(CASE WHEN u.asud THEN 0 ELSE 1 END) AS placeholders""")
    names = one("MATCH (u:Unit {asud: true}) RETURN min(u.name) AS first, max(u.name) AS last")
    text = one("""
        MATCH (p:Passage)
        RETURN count(p) AS passages, sum(size(p.text)) AS chars,
               sum(size(split(p.text, ' '))) AS words,
               min(p.year) AS year_min, max(p.year) AS year_max,
               percentileDisc(p.year, 0.5) AS year_median""")
    per_unit = one("""
        MATCH (u:Unit {has_text: true})
        WITH COUNT { (:Passage)-[:DESCRIBES]->(u) } AS n
        RETURN avg(n) AS mean, max(n) AS max""")
    states = db.read(n4, """
        MATCH (:Unit {asud: true})-[:OCCURS_IN]->(s:State) WHERE s.name IS NOT NULL
        RETURN s.name AS state, count(*) AS units ORDER BY units DESC""")
    rels = db.read(n4, """
        MATCH ()-[r]->() WITH type(r) AS type, coalesce(r.sources, ['derived']) AS srcs
        UNWIND srcs AS source
        RETURN type, source, count(*) AS n ORDER BY type, source""")
    nodes = db.read(n4, "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC")
    return {
        "units": units, "names": names, "text": text,
        "passages_per_unit": {"mean": round(per_unit["mean"], 1), "max": per_unit["max"]},
        "states": {"count": len(states), "top": states[:6]},
        "relationships": rels,
        "nodes": nodes,
    }


async def about(_: Request) -> JSONResponse:
    try:
        return JSONResponse(await run_in_threadpool(corpus_stats))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=503)


async def health(_: Request) -> JSONResponse:
    try:
        return JSONResponse({"neo4j": db.ping(CFG.neo4j)})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"neo4j": None, "error": str(e)}, status_code=503)


app = Starlette(routes=[
    Route("/", index),
    Route("/api/meta", meta),
    Route("/api/ask", ask, methods=["POST"]),
    Route("/api/about", about),
    Route("/api/health", health),
])
