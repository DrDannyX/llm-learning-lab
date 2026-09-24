"""MCP server: the three systems, and the primitives under them, as tools.

Two levels of tool, on purpose:

  answer tools      ask_rag / ask_kg / ask_graphrag / compare_all
                    -- the full pipelines, for "which system answers this best?"
  primitive tools   vector_search / run_cypher / graph_schema / unit_facts
                    -- the raw retrieval steps, so an agent can do its own
                       retrieval and see why a pipeline succeeded or failed.

An agent with only the primitives is, in effect, an agentic GraphRAG system:
it decides per question whether to search text, query the graph, or both. Try
it (experiment #7).

Run standalone:   python -m georag.mcp_server          (stdio)
                  python -m georag.mcp_server --http   (streamable HTTP on :8765)
Anything printed to stdout corrupts the stdio protocol; log to stderr only.
"""
from __future__ import annotations

import json
import logging
import sys

from mcp.server.mcpserver import MCPServer

from . import config, db
from .retrievers import graphrag, kg, rag
from .retrievers.link import linker
from .retrievers.pipeline import LABELS, answer

CFG = config.load()

# the server's stderr is the agent's terminal: keep it to warnings
for name in ("httpx", "strands", "mcp"):
    logging.getLogger(name).setLevel(logging.WARNING)

mcp = MCPServer(
    name="geo-graphrag",
    log_level="WARNING",
    instructions=(
        "Question answering over the USGS Geolex lexicon of US geologic units, three ways: "
        "vector RAG (text passages), a knowledge graph (Neo4j, queried with Cypher), and "
        "GraphRAG (both). Use compare_all to see all three answers side by side."
    ),
)


def _summary(system: str, question: str) -> str:
    a = answer(CFG, system, question)
    r = a.retrieval
    lines = [f"## {LABELS[system]} ({a.seconds:.1f}s)", a.answer, "", "Evidence:"]
    if system == "kg":
        for att in r.debug.get("attempts", []):
            lines.append(f"- cypher ({att['rows']} rows{', ERROR ' + att['error'] if att['error'] else ''}):"
                         f" {' '.join(att['cypher'].split())}")
    else:
        lines.append(f"- sources: {', '.join(r.sources[:10])}")
        if system == "graphrag":
            lines.append(f"- expanded units: {', '.join(r.debug.get('seed_units', []))}")
    return "\n".join(lines)


@mcp.tool()
def compare_all(question: str) -> str:
    """Answer a question with all three systems (vector RAG, knowledge graph, GraphRAG)
    and return the three answers with their evidence, for side-by-side comparison."""
    return "\n\n".join(_summary(s, question) for s in ("rag", "kg", "graphrag"))


@mcp.tool()
def ask_rag(question: str) -> str:
    """Answer using vector RAG only: the most similar text passages from the lexicon.
    Best for descriptive details stated in prose (fossils, type localities, composition)."""
    return _summary("rag", question)


@mcp.tool()
def ask_kg(question: str) -> str:
    """Answer using the knowledge graph only: the model writes a Cypher query and answers
    from its rows. Best for counts, lists, filters and hierarchies. Sees no prose."""
    return _summary("kg", question)


@mcp.tool()
def ask_graphrag(question: str) -> str:
    """Answer using GraphRAG: vector search plus graph facts about the units involved."""
    return _summary("graphrag", question)


@mcp.tool()
def vector_search(query: str, k: int = 5) -> str:
    """Return the k lexicon passages most similar to the query, with similarity scores."""
    rows = rag.search(CFG, query, max(1, min(k, 20)))
    return "\n\n".join(f"[{r['id']}] score={r['score']:.3f} {r['unit']} ({r['year']}): {r['text']}"
                       for r in rows)


@mcp.tool()
def run_cypher(cypher: str) -> str:
    """Run a READ-ONLY Cypher query against the Neo4j graph and return up to 50 rows as
    JSON lines. Call graph_schema first to see labels, relationships and properties."""
    try:
        rows = db.read_untrusted(CFG.neo4j, cypher, limit=50)
    except Exception as e:  # noqa: BLE001 -- the error text is the useful output
        return f"ERROR {type(e).__name__}: {e}"
    return "\n".join(json.dumps(r, default=str) for r in rows) or "(no rows)"


@mcp.tool()
def graph_schema() -> str:
    """Describe the knowledge graph: node labels, relationship types, properties,
    and example Cypher queries."""
    return f"{kg.SCHEMA}\n\nExample queries:\n{kg.EXAMPLES}"


@mcp.tool()
def unit_facts(name: str) -> str:
    """Look up a geologic unit by name (e.g. 'Eagle Ford', 'Aarde Shale Member') and return
    its graph neighbourhood: age, lithology, states, hierarchy, over/underlying units,
    plus its graph key for use in run_cypher."""
    linked = linker(CFG.neo4j).link(name if name[:1].isupper() else name.title())
    if not linked.units:
        return f"No unit named {name!r} found in the graph."
    out = []
    for u in linked.units:
        card = graphrag.unit_card(CFG, u["key"])
        if card:
            out.append(f"key={u['key']}\n{graphrag.format_card(card)}")
    return "\n\n".join(out)


def main() -> None:
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http", port=8765)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
