"""Cloze probes: does the model actually KNOW more geoscience?

Perplexity is a fluency measure over whole passages, and it can improve simply
because the model learned the *style* of lexicon prose -- the hedging, the
abbreviations, the sentence rhythm. That is real but it is not knowledge.

A cloze probe isolates knowledge. Take a held-out sentence containing a domain
term, hide the term, and ask the model to score the continuation:

    "The Austin Chalk is of Late ______ age."   ->  Cretaceous

We score by the model's mean log-probability of the correct completion, and
compare against distractors drawn from the same vocabulary class (other
periods). Accuracy = how often the true answer beats every distractor.

This is a weak proxy for knowledge, and it should be read as one: it measures
whether the right token is *more likely*, not whether the model can reason.
But it separates "learned the vocabulary distribution" from "learned the
style", which perplexity alone cannot do.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def sequence_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Mean log-probability the model assigns to `completion` after `prompt`."""
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    c_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not c_ids:
        return float("-inf")
    ids = mx.array([p_ids + c_ids])
    logits = model(ids[:, :-1]).astype(mx.float32)
    targets = ids[:, 1:]
    lp = -nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    return float(lp[-len(c_ids):].mean().item())


def build_probes(texts: list[str], vocabulary: list[str], n: int,
                 n_distractors: int = 3, seed: int = 17) -> list[dict]:
    """Find held-out sentences containing a vocabulary term and blank it out."""
    rng = random.Random(seed)
    vocab_set = {v.lower(): v for v in vocabulary}
    probes: list[dict] = []
    for text in texts:
        for sent in re.split(r"(?<=[.;])\s+", text):
            words = re.findall(r"[A-Za-z][A-Za-z\-]+", sent)
            hits = [w for w in words if w.lower() in vocab_set]
            if not hits or len(words) < 8:
                continue
            answer = rng.choice(hits)
            idx = sent.find(answer)
            if idx <= 20:                       # need real left context
                continue
            distractors = [v for v in vocabulary if v.lower() != answer.lower()]
            if len(distractors) < n_distractors:
                continue
            probes.append({
                "prompt": sent[:idx].strip(),
                "answer": answer,
                "distractors": rng.sample(distractors, n_distractors),
            })
            break
        if len(probes) >= n:
            break
    return probes


def run_probes(model, tokenizer, probes: list[dict]) -> dict:
    correct = 0
    margins: list[float] = []
    for p in probes:
        true_lp = sequence_logprob(model, tokenizer, p["prompt"], " " + p["answer"])
        d_lps = [sequence_logprob(model, tokenizer, p["prompt"], " " + d)
                 for d in p["distractors"]]
        best_d = max(d_lps) if d_lps else float("-inf")
        if true_lp > best_d:
            correct += 1
        margins.append(true_lp - best_d)
    n = max(len(probes), 1)
    return {"n": len(probes), "accuracy": correct / n,
            "mean_margin": float(np.mean(margins)) if margins else 0.0}


def compare(base_model: str, adapted_path: str | Path, stage: str, n: int = 200) -> dict:
    """Probe accuracy before and after continued pretraining."""
    from mlx_lm.utils import load

    from .. import paths
    from ..corpus.build import load_split

    # the label vocabulary from the SFT project -- chronostratigraphic interval
    # names make good probes: closed set, unambiguous, genuinely domain knowledge
    from geosft.data import vocab as sft_vocab
    v = sft_vocab.load()
    vocabulary = [c["name"] for c in v["chronostrat"] if " " not in c["name"]][:200]

    texts = load_split(stage, "val")
    probes = build_probes(texts, vocabulary, n)
    console.print(f"built [cyan]{len(probes)}[/cyan] cloze probes from held-out text")
    if not probes:
        return {"n": 0}

    out: dict[str, dict] = {}
    for label, path in (("base", base_model), ("adapted", str(adapted_path))):
        model, tok = load(path)
        model.eval()
        out[label] = run_probes(model, tok, probes)
        del model

    t = Table(title="cloze probe: geoscience term recovery")
    t.add_column("model"); t.add_column("accuracy", justify="right")
    t.add_column("mean margin", justify="right")
    for k, r in out.items():
        t.add_row(k, f"{r['accuracy']:.3f}", f"{r['mean_margin']:+.3f}")
    console.print(t)
    console.print("[dim]accuracy = true term beats all distractors. Chance is "
                  "1/(1+n_distractors) = 0.25.[/dim]")
    return {"stage": stage, "probes": len(probes), "results": out}
