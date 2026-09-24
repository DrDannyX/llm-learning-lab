"""The two model endpoints this lab uses, both served by LM Studio.

* **Chat** goes through Strands. Every LLM step -- answering, writing Cypher,
  generating and grading benchmark questions -- is a Strands ``Agent`` over an
  ``OpenAIModel`` pointed at LM Studio's OpenAI-compatible server.
* **Embeddings** are a plain HTTP call. Strands is an agent framework and has
  no embedding API, and an embedding is not an agent step.

nomic-embed-text is trained with task prefixes. Documents are embedded as
``search_document: ...`` and queries as ``search_query: ...``; dropping the
prefixes costs retrieval quality silently (experiment #2).
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import httpx
import numpy as np
from strands import Agent
from strands.models.openai import OpenAIModel

from .config import LLMConfig

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Strands warns once per call that reasoning blocks are not replayed over the
# Chat Completions API. Our agents are single-turn, so that is expected.
logging.getLogger("strands.models.openai").setLevel(logging.ERROR)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def model(cfg: LLMConfig, model_id: str | None = None, max_tokens: int | None = None) -> OpenAIModel:
    params: dict = {"temperature": cfg.temperature, "max_tokens": max_tokens or cfg.max_tokens}
    if cfg.reasoning_effort:
        params["reasoning_effort"] = cfg.reasoning_effort
    return OpenAIModel(
        client_args={"base_url": cfg.base_url, "api_key": "lm-studio"},
        model_id=model_id or cfg.chat_model,
        params=params,
    )


def complete(cfg: LLMConfig, system: str, prompt: str, *, model_id: str | None = None,
             max_tokens: int | None = None) -> tuple[str, dict]:
    """One stateless LLM call. Returns (text, token usage).

    A fresh Agent per call is deliberate: a reused Agent keeps its message
    history, and a retriever that silently sees the previous question's
    context is exactly the kind of bug that produces plausible answers.
    """
    agent = Agent(model=model(cfg, model_id, max_tokens), system_prompt=system,
                  callback_handler=None)
    result = agent(prompt)
    text = _THINK.sub("", str(result)).strip()
    return text, dict(result.metrics.accumulated_usage)


class Embedder:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = httpx.Client(base_url=cfg.base_url, timeout=120)

    def _embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.post("/embeddings", json={"model": self.cfg.embed_model, "input": texts})
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([DOC_PREFIX + t for t in texts])

    def query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0].tolist()


@lru_cache(maxsize=4)
def embedder(cfg_json: str) -> Embedder:
    return Embedder(LLMConfig.model_validate_json(cfg_json))
