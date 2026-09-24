"""A Strands agent that interrogates the three systems through the MCP server.

The agent runs the MCP server as a subprocess over stdio and gets its tools
from it -- it has no direct import of the retrievers. That separation is the
point of MCP: the same server works unchanged from Claude Desktop, Claude
Code, LM Studio's own MCP support, or any other client.
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

from mcp.client.stdio import StdioServerParameters, stdio_client
from strands import Agent
from strands.tools.mcp import MCPClient

from .config import Config
from .llm import model
from .paths import ROOT

SYSTEM = """You are a research assistant comparing three ways of answering questions about
US geologic units from the USGS Geolex lexicon:
  1. vector RAG      - retrieves similar text passages (ask_rag)
  2. knowledge graph - writes a Cypher query over a Neo4j graph (ask_kg)
  3. GraphRAG        - text passages plus graph facts (ask_graphrag)

When the user asks a domain question, call compare_all, then report what each system
answered and say which answer is best supported by its evidence, and why.
When you need to dig deeper, use the primitives: unit_facts, vector_search, graph_schema,
run_cypher (read-only). Never invent facts that no tool returned. Be concise."""


def mcp_client(config_path: str | None = None) -> MCPClient:
    env = {"GEORAG_CONFIG": config_path} if config_path else None
    params = StdioServerParameters(command=sys.executable, args=["-m", "georag.mcp_server"],
                                   env=env, cwd=str(ROOT))
    return MCPClient(lambda: stdio_client(params), startup_timeout=60)


@contextmanager
def session(cfg: Config, config_path: str | None = None,
            callback_handler=None) -> Iterator[Agent]:
    client = mcp_client(config_path)
    with client:
        tools = client.list_tools_sync()
        kwargs = {} if callback_handler is None else {"callback_handler": callback_handler}
        yield Agent(model=model(cfg.llm, max_tokens=2048), tools=tools, system_prompt=SYSTEM,
                    **kwargs)
