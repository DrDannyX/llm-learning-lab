"""What a retriever returns. `context` is exactly what the answer model sees --
nothing else about the retriever reaches the answer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Retrieval:
    system: str
    context: str
    sources: list[str] = field(default_factory=list)   # passage ids / unit keys used
    debug: dict = field(default_factory=dict)          # cypher, linked entities, scores
    seconds: float = 0.0
    error: str | None = None


@dataclass
class Answer:
    system: str
    question: str
    answer: str
    retrieval: Retrieval
    seconds: float                                    # retrieval + generation
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
