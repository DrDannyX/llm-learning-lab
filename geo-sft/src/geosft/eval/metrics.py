"""Scoring predicted JSON against gold JSON, field by field.

A single accuracy number hides everything you need to know. A model can score
well overall while never once getting `relations` right, because the list
fields are easy and the relational field is hard. So every field is scored on
its own terms:

  unit_name    exact match after normalisation   -> accuracy
  rank         closed 9-way label                -> accuracy
  list fields  unordered sets                    -> micro P / R / F1
  thickness    numbers with tolerance            -> accuracy
  relations    set of (kind, unit) pairs         -> micro P / R / F1

Two gates come before any of that, and they are reported separately because
they fail differently:

  parse_rate   did the model emit parseable JSON at all?
  schema_rate  did that JSON satisfy the schema?

An untuned instruct model typically scores well on content and badly on those
two gates -- it wraps JSON in prose, adds commentary, invents fields. A large
part of what SFT buys you here is format compliance, and you want to see that
separated from genuine extraction skill.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from ..schema import GeoExtraction

LIST_FIELDS = ("lithologies", "chronostrat", "minerals", "states")
THICK_REL_TOL = 0.05  # 5% -- unit conversion and rounding noise, not laxity

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)
#: Reasoning models (Qwen3 base variants, not the -Instruct-2507 ones) emit a
#: <think> block before the answer. Strip it rather than letting the greedy
#: JSON search pick up braces from inside the reasoning.
_THINK = re.compile(r"<think>.*?</think>", re.S)


def extract_json(text: str) -> dict | None:
    """Best-effort recovery of a JSON object from a model response.

    We are deliberately lenient here and record leniency separately via
    `parse_rate`/`strict_json_rate`. Being strict would conflate "cannot
    extract" with "chatty", and those want different fixes.
    """
    if not text:
        return None
    cleaned = _FENCE.sub("", _THINK.sub("", text).strip())
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _norm(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, pred: set, gold: set) -> None:
        self.tp += len(pred & gold)
        self.fp += len(pred - gold)
        self.fn += len(gold - pred)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {"precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4), "tp": self.tp, "fp": self.fp, "fn": self.fn}


def _thickness_match(pred: dict | None, gold: dict | None) -> bool:
    if gold is None and pred is None:
        return True
    if gold is None or pred is None:
        return False
    for key in ("min_m", "max_m"):
        g, p = gold.get(key), pred.get(key)
        if g is None and p is None:
            continue
        if g is None or p is None:
            return False
        try:
            g, p = float(g), float(p)
        except (TypeError, ValueError):
            return False
        if abs(g - p) > max(THICK_REL_TOL * abs(g), 0.05):
            return False
    return True


def _relation_set(obj: dict) -> set[tuple[str, str]]:
    out = set()
    for r in obj.get("relations") or []:
        if isinstance(r, dict) and r.get("kind") and r.get("unit"):
            out.add((str(r["kind"]).strip().lower(), _norm(str(r["unit"]))))
    return out


@dataclass
class Scorer:
    n: int = 0
    parsed: int = 0
    strict_json: int = 0
    schema_ok: int = 0
    name_correct: int = 0
    name_total: int = 0
    rank_correct: int = 0
    thick_correct: int = 0
    lists: dict[str, PRF] = field(default_factory=lambda: {f: PRF() for f in LIST_FIELDS})
    relations: PRF = field(default_factory=PRF)
    rank_confusion: Counter = field(default_factory=Counter)
    empty_output: int = 0

    def add(self, raw_output: str, gold: dict) -> dict:
        self.n += 1
        if not (raw_output or "").strip():
            self.empty_output += 1

        try:
            json.loads((raw_output or "").strip())
            self.strict_json += 1
        except Exception:
            pass

        pred = extract_json(raw_output)
        if pred is None:
            # An unparseable answer is scored as an empty prediction: every
            # gold item becomes a false negative. Skipping it instead would
            # silently reward a model for refusing to answer.
            pred = {}
        else:
            self.parsed += 1
            try:
                GeoExtraction.model_validate(pred)
                self.schema_ok += 1
            except Exception:
                pass

        if gold.get("unit_name"):
            self.name_total += 1
            if _norm(pred.get("unit_name")) == _norm(gold.get("unit_name")):
                self.name_correct += 1

        g_rank = gold.get("rank", "Unknown")
        p_rank = pred.get("rank", "Unknown")
        if p_rank == g_rank:
            self.rank_correct += 1
        else:
            self.rank_confusion[f"{g_rank}->{p_rank}"] += 1

        for f in LIST_FIELDS:
            gset = {_norm(x) for x in (gold.get(f) or []) if x}
            pv = pred.get(f) or []
            pset = {_norm(x) for x in pv if isinstance(x, str)} if isinstance(pv, list) else set()
            self.lists[f].add(pset, gset)

        self.relations.add(_relation_set(pred), _relation_set(gold))

        if _thickness_match(pred.get("thickness"), gold.get("thickness")):
            self.thick_correct += 1

        return pred

    def report(self) -> dict:
        n = max(self.n, 1)
        per_field = {f: self.lists[f].as_dict() for f in LIST_FIELDS}
        per_field["relations"] = self.relations.as_dict()
        # A field with no gold items AND no predictions anywhere in the eval
        # set is undefined, not zero. Counting it as 0.0 would punish a model
        # for a field the data never exercises.
        scored = {f: d for f, d in per_field.items() if (d["tp"] + d["fp"] + d["fn"]) > 0}
        f1s = [d["f1"] for d in scored.values()]
        return {
            "n": self.n,
            "gates": {
                "parse_rate": round(self.parsed / n, 4),
                "strict_json_rate": round(self.strict_json / n, 4),
                "schema_valid_rate": round(self.schema_ok / n, 4),
                "empty_output_rate": round(self.empty_output / n, 4),
            },
            "scalar_fields": {
                "unit_name_accuracy": round(self.name_correct / max(self.name_total, 1), 4),
                "unit_name_n": self.name_total,
                "rank_accuracy": round(self.rank_correct / n, 4),
                "thickness_accuracy": round(self.thick_correct / n, 4),
            },
            "set_fields": per_field,
            "macro_f1": round(sum(f1s) / max(len(f1s), 1), 4),
            "macro_f1_fields": sorted(scored),
            "rank_confusion_top": dict(self.rank_confusion.most_common(6)),
        }
