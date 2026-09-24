"""Typed config, loaded from YAML. Env vars override the connection settings,
so the MCP server can be pointed elsewhere without editing files."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from .paths import CONFIGS


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    chat_model: str = "google/gemma-4-12b"
    judge_model: str = "google/gemma-4-12b"
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"
    embed_dim: int = 768
    temperature: float = 0.0
    max_tokens: int = 1024
    reasoning_effort: str | None = "none"


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "geograph-lab"
    database: str = "neo4j"


class IngestConfig(BaseModel):
    limit_units: int | None = None
    embed_batch: int = 64


class RetrievalConfig(BaseModel):
    rag_k: int = 8
    graphrag_k: int = 5
    graphrag_max_units: int = 4
    kg_max_rows: int = 60
    kg_retries: int = 1


class BenchConfig(BaseModel):
    seed: int = 13
    per_category: int = 8
    name: str = "bench-v1"


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    ingest: IngestConfig = IngestConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    bench: BenchConfig = BenchConfig()


def load(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("GEORAG_CONFIG") or CONFIGS / "default.yaml")
    cfg = Config.model_validate(yaml.safe_load(path.read_text()) or {})
    env = {
        "GEORAG_LLM_URL": ("llm", "base_url"),
        "GEORAG_CHAT_MODEL": ("llm", "chat_model"),
        "GEORAG_NEO4J_URI": ("neo4j", "uri"),
        "GEORAG_NEO4J_PASSWORD": ("neo4j", "password"),
    }
    for var, (section, key) in env.items():
        if os.environ.get(var):
            setattr(getattr(cfg, section), key, os.environ[var])
    return cfg
