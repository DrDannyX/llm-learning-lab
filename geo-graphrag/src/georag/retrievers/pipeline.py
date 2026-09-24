"""Context -> answer, identically for all three systems.

Same model, same prompt, same temperature. If the systems differ in answer
quality, the only cause left is the context they retrieved. Change this prompt
for one system and every comparison in the benchmark becomes meaningless.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from ..config import Config
from ..llm import complete
from . import graphrag, kg, rag
from .base import Answer, Retrieval

SYSTEMS: dict[str, Callable[[Config, str], Retrieval]] = {
    "rag": rag.retrieve,
    "kg": kg.retrieve,
    "graphrag": graphrag.retrieve,
}

LABELS = {"rag": "Vector RAG", "kg": "Knowledge graph", "graphrag": "GraphRAG"}

ANSWER_SYSTEM = """You answer questions about US geologic units using ONLY the context below.
The context comes from a retrieval system: it may be incomplete, and some of it may be
irrelevant. Rules:
- If the context does not contain the answer, reply exactly:
  "I don't know based on the retrieved context."
- Do not use outside knowledge.
- Be concise: one to three sentences, or a short list.
- If the question asks for a list, give every matching item in the context.
- If the question asks "how many", answer with the number first."""

ABSTAIN = "i don't know based on the retrieved context"


def answer(cfg: Config, system: str, question: str) -> Answer:
    t = time.perf_counter()
    retrieval = SYSTEMS[system](cfg, question)
    if not retrieval.context.strip():
        text, usage = "I don't know based on the retrieved context.", {}
    else:
        prompt = f"Context:\n{retrieval.context}\n\nQuestion: {question}"
        text, usage = complete(cfg.llm, ANSWER_SYSTEM, prompt)
    return Answer(system=system, question=question, answer=text, retrieval=retrieval,
                  seconds=time.perf_counter() - t, usage=usage)


def compare(cfg: Config, question: str, systems: list[str] | None = None) -> list[Answer]:
    return [answer(cfg, s, question) for s in (systems or list(SYSTEMS))]
