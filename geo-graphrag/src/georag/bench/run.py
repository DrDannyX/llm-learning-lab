"""Run every question through every system, score it, and write a report.

Results are appended to `results.jsonl` one line at a time and the run resumes
where it stopped: a benchmark over a local LLM takes long enough that you will
interrupt it at least once.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from rich.progress import Progress

from ..config import Config
from ..retrievers.pipeline import LABELS, SYSTEMS, answer
from .build import CATEGORIES
from .score import score


def run(cfg: Config, questions: list[dict], out_dir: Path,
        systems: list[str] | None = None) -> Path:
    systems = systems or list(SYSTEMS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    results = out_dir / "results.jsonl"
    done = set()
    if results.exists():
        done = {(r["id"], r["system"]) for r in map(json.loads, results.open())}

    todo = [(q, s) for q in questions for s in systems if (q["id"], s) not in done]
    with Progress() as prog, results.open("a") as f:
        task = prog.add_task("benchmark", total=len(todo))
        for q, s in todo:
            a = answer(cfg, s, q["question"])
            sc = score(cfg, q, a.answer, a.retrieval.context, a.retrieval.sources)
            row = {
                "id": q["id"], "category": q["category"], "system": s,
                "question": q["question"], "gold": q["gold"], "answer": a.answer,
                **sc,
                "seconds": round(a.seconds, 2),
                "retrieval_seconds": round(a.retrieval.seconds, 2),
                "context_chars": len(a.retrieval.context),
                "retrieval_error": a.retrieval.error,
                "debug": a.retrieval.debug,
            }
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            prog.advance(task)
    report(out_dir)
    return out_dir


def _mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 3) if xs else float("nan")


def report(out_dir: Path) -> dict:
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").open()]
    systems = [s for s in SYSTEMS if any(r["system"] == s for r in rows)]
    cats = [c for c in CATEGORIES if any(r["category"] == c for r in rows)]
    by = defaultdict(list)
    for r in rows:
        by[(r["system"], r["category"])].append(r)
        by[(r["system"], "ALL")].append(r)

    def agg(key: str) -> dict:
        return {s: {c: _mean([r[key] for r in by[(s, c)]]) for c in cats + ["ALL"]}
                for s in systems}

    rep = {
        "n_questions": len({r["id"] for r in rows}),
        "systems": systems,
        "categories": cats,
        "score": agg("score"),
        "judge": agg("judge"),
        "context_recall": agg("context_recall"),
        "abstain_rate": {s: _mean([float(r["abstained"]) for r in by[(s, "ALL")]]) for s in systems},
        "median_seconds": {s: round(statistics.median(r["seconds"] for r in by[(s, "ALL")]), 2)
                           for s in systems},
        "median_context_chars": {s: int(statistics.median(r["context_chars"] for r in by[(s, "ALL")]))
                                 for s in systems},
        "kg_query_errors": sum(1 for r in rows if r["system"] == "kg" and r["retrieval_error"]),
        # judge vs string-matcher agreement on structured questions: how far
        # to trust the judge on the category where it is the only grader
        "judge_agreement": _mean([float((r["score"] >= 0.5) == bool(r["judge"]))
                                  for r in rows if r["category"] != "descriptive"]),
    }
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "report.md").write_text(to_markdown(rep))
    return rep


def to_markdown(rep: dict) -> str:
    systems, cats = rep["systems"], rep["categories"]
    head = "| category | " + " | ".join(LABELS[s] for s in systems) + " |\n"
    sep = "|---|" + "---|" * len(systems) + "\n"

    def table(metric: str) -> str:
        body = ""
        for c in cats + ["ALL"]:
            vals = [rep[metric][s][c] for s in systems]
            best = max(vals)
            cells = [f"**{v:.2f}**" if v == best and len(systems) > 1 else f"{v:.2f}" for v in vals]
            name = f"**{c}**" if c == "ALL" else c
            body += f"| {name} | " + " | ".join(cells) + " |\n"
        return head + sep + body

    ops = "| | " + " | ".join(LABELS[s] for s in systems) + " |\n" + sep
    ops += "| abstain rate | " + " | ".join(f"{rep['abstain_rate'][s]:.2f}" for s in systems) + " |\n"
    ops += "| median seconds | " + " | ".join(f"{rep['median_seconds'][s]:.1f}" for s in systems) + " |\n"
    ops += ("| median context chars | "
            + " | ".join(f"{rep['median_context_chars'][s]:,}" for s in systems) + " |\n")
    return (
        f"# Benchmark report ({rep['n_questions']} questions)\n\n"
        "## Score (headline)\n\nString-matched against gold for structured categories; "
        "LLM-judged for `descriptive`.\n\n" + table("score") +
        "\n## Context recall\n\nDid the retrieved context contain the answer at all? "
        "Low here = retrieval failure; high here but low score = generation failure.\n\n"
        + table("context_recall") +
        "\n## LLM judge (all questions)\n\n" + table("judge") +
        f"\nJudge agrees with the string matcher on {rep['judge_agreement']:.0%} of "
        "structured questions.\n\n## Cost\n\n" + ops +
        f"\nKG queries that still errored after repair: {rep['kg_query_errors']}\n"
    )
