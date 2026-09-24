"""Scoring. Deterministic wherever possible; an LLM judge only where not.

Every (question, system) pair gets four numbers:

  score           the headline. Structured categories: string-matched against
                  gold (below). Descriptive: the judge's verdict.
  judge           an LLM grader's 0/1 verdict against the reference, on EVERY
                  question -- a cross-check on the string matcher.
  context_recall  did the retrieved CONTEXT contain the answer? Scored with the
                  same matcher, on the context instead of the answer.
  abstained       did the system say "I don't know"?

context_recall is the diagnostic that matters most. When score is low:
  - context_recall low  -> RETRIEVAL failed; the answer model never saw it.
  - context_recall high -> GENERATION failed; the answer was there and was missed.
Those have completely different fixes, and one averaged score cannot tell them apart.

Match modes
  all    fraction of gold items mentioned (recall). Lists, ages, states.
  any    1 if any gold item is mentioned. "What is X part of?" has several true answers.
  count  1 if the first number in the answer equals the gold count, else 0.
  judge  the judge's verdict (descriptive questions have no string gold).

Names are matched on their core ("Church Member" matches "Church limestone"),
because an answer that names the right unit with a different rank word is right.
"""
from __future__ import annotations

import json
import re

from ..config import Config
from ..ingest.extract import STATE_NAMES, core_name
from ..llm import complete
from ..retrievers.pipeline import ABSTAIN

_CODE_OF = {v.lower(): k for k, v in STATE_NAMES.items()}
_NUM = re.compile(r"\b\d[\d,]*\b")
_QUALIFIER = re.compile(r"^ (early|middle|late) ")


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "


def mentions(text: str, item: str) -> bool:
    t = _norm(text)
    forms = {_norm(item).strip(), core_name(item)}
    # "Paleocene" for gold "Late Paleocene" is less precise, not wrong
    forms.add(_QUALIFIER.sub("", _norm(item)).strip())
    if item.lower() in _CODE_OF:  # a state: its postal code counts too
        forms.add(_CODE_OF[item.lower()].lower())
    return any(f and f" {f} " in t for f in forms)


def first_number(text: str) -> int | None:
    m = _NUM.search(text)
    return int(m.group(0).replace(",", "")) if m else None


def match(q: dict, text: str) -> float | None:
    gold = q["gold"]
    if q["match"] == "all":
        return sum(mentions(text, g) for g in gold) / len(gold)
    if q["match"] == "any":
        return float(any(mentions(text, g) for g in gold))
    if q["match"] == "count":
        # the answer must lead with the number; context is searched anywhere
        return float(first_number(text) == gold)
    return None


def context_recall(q: dict, context: str, sources: list[str]) -> float:
    if q["match"] == "judge":
        return float(q["source"] in sources)  # was the source passage retrieved?
    if q["match"] == "count":
        # a count is only "in" the context if the context computed it
        return float(re.search(rf"\b{q['gold']}\b", context) is not None
                     and "count" in context.lower())
    return match(q, context) or 0.0


JUDGE_SYSTEM = """You grade answers to geology questions against a reference answer.
The candidate is CORRECT (1) if it states the same fact as the reference. Extra detail is
fine; different wording is fine. For list questions, it is correct if it gives most of the
reference items and no clearly wrong ones. It is INCORRECT (0) if it contradicts the
reference, omits the key fact, or says it does not know.
Reply with JSON only: {"correct": 0 or 1, "reason": "<one short sentence>"}"""


def judge(cfg: Config, q: dict, answer: str) -> tuple[int, str]:
    prompt = (f"Question: {q['question']}\nReference answer: {q['reference']}\n"
              f"Candidate answer: {answer}")
    text, _ = complete(cfg.llm, JUDGE_SYSTEM, prompt, model_id=cfg.llm.judge_model,
                       max_tokens=200)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        v = json.loads(m.group(0)) if m else {}
        return int(bool(int(v.get("correct", 0)))), str(v.get("reason", ""))
    except (ValueError, TypeError):
        return int('"correct": 1' in text), text[:200]


def score(cfg: Config, q: dict, answer: str, context: str, sources: list[str]) -> dict:
    j, reason = judge(cfg, q, answer)
    m = match(q, answer)
    return {
        "score": float(j) if m is None else m,
        "judge": j,
        "judge_reason": reason,
        "context_recall": context_recall(q, context, sources),
        "abstained": ABSTAIN in answer.lower(),
    }
