# Results

Everything this project measured, including the two methodology errors it
made and what they taught. Reproducible via `make all` then
`geocpt transfer`.

---

## 1. Setup

| | |
|---|---|
| Hardware | Apple M4 Pro, 48 GB unified memory |
| Base model | `Qwen/Qwen3-1.7B` (bf16, **post-trained** — see §7) |
| Method | full fine-tune, all 1.72B parameters |
| Stage | TAPT — CPT on geo-sft's unlabelled task text |
| Corpus | 1.57M tokens (6,318 domain docs + 1,114 replay, 15%) |
| Schedule | 600 iters, batch 1 × accum 16, LR 1e-5 cosine, warmup 40 |
| Wall time | 55 min |

## 2. Corpus

| stage | count |
|---|---|
| Raw documents (geo-sft train passages) | 6,635 |
| Quality filter | **skipped** — task text is the target distribution (§6) |
| Near-duplicates removed (MinHash/LSH) | 17 |
| Domain train | 6,318 |
| Replay (wikitext, 15%) | 1,114 |
| Domain val / forgetting probe | 300 / 300 |
| **Approx. train tokens** | **1.57M** |

## 3. Packing

| mode | retention | efficiency |
|---|---|---|
| packed | 99.98% | **100%** |
| `no_cross_document` | 100% | **61.8%** |

Avoiding cross-document attention would cost 38% of the compute budget on
this corpus.

## 4. Training

| | before | after | change |
|---|---|---|---|
| **Domain perplexity** | 33.27 | **13.87** | **−58.3%** |

```
iter   0   33.27
iter  75   22.42
iter 150   17.05
iter 300   14.65
iter 450   13.95
iter 600   13.87     <- flattening
```

The model roughly halved its uncertainty about the next token in geological
prose, and had largely converged by iteration 450 on 1.57M tokens.

### Memory: peak is set by the optimizer, not the batch

| iteration | peak |
|---|---|
| 10 (before first optimizer step) | 12.3 GB |
| 20 (after it) | **49.0 GB** |

The optimizer step lands at iteration 16. Everything under 12 GB is
activations and gradients; the remaining ~37 GB appears when AdamW allocates
its moments. Dropping batch 2 → 1 barely moved peak (48.7 → 49.0 GB) but
stopped the machine swapping: **9.1 → 3.4 s/iter**.

Compare geo-sft: **8.5 GB for a 4B model**, because LoRA's gradients and
optimizer state scale with the adapter, not the model.

## 5. Evaluation

### Cloze probes — knowledge, or style?

| model | accuracy | mean margin |
|---|---|---|
| base | 0.815 | +2.130 |
| adapted | 0.833 | +2.330 |

54 probes, chance 0.25. **0.815 → 0.833 is one extra probe correct.** Not
significant. Large fluency gain, no detectable knowledge gain — exactly what
the probe exists to separate, and the expected outcome at 1.57M tokens.

### Downstream transfer — the number that matters

Three arms, all scored by geo-sft's own trainer and evaluator on identical
data, 200 held-out examples:

| arm | zero-shot | after SFT |
|---|---|---|
| 1.7B + SFT (no CPT) | 0.001 | **0.823** |
| **1.7B + TAPT + SFT** | 0.002 | **0.842** |
| 4B + SFT (geo-sft baseline) | 0.314 | 0.831 |

**TAPT contributed +0.019 macro F1** with model size held constant.

Per field, the gains land on the hardest, most domain-dependent ones:

| field | no CPT | + TAPT | Δ |
|---|---|---|---|
| minerals | 0.677 | 0.713 | **+0.037** |
| relations | 0.674 | 0.703 | **+0.030** |
| rank | 0.845 | 0.860 | +0.015 |
| thickness | 0.905 | 0.920 | +0.015 |
| lithologies | 0.915 | 0.928 | +0.013 |
| chronostrat | 0.907 | 0.920 | +0.013 |
| states | 0.942 | 0.946 | +0.004 |
| unit_name | 0.905 | 0.899 | −0.006 |

The two weakest fields improved most; the near-saturated ones barely moved.
A coherent pattern rather than scattered noise.

> **Be sceptical of +0.019.** One seed, 200 examples, no confidence interval.
> Suggestive, not established. A real claim needs several seeds.

This was mildly **better than predicted** — the README warned to expect no
downstream gain at this scale. The technique is slightly more useful at 1.5M
tokens than the literature's scale guidance implies.

## 6. Methodology errors, and what they taught

### Error 1 — the forgetting probe overlapped the replay corpus

The first run replayed wikitext into training and measured forgetting on
*held-out wikitext*: different documents, same distribution. It reported
**−40.5% forgetting**, i.e. general perplexity improving 40%.

It was measuring "did we also learn wikitext". Fixed with a separate
`forgetting_dataset` (`ag_news`), never replayed, plus a test asserting the
two corpora differ.

### Error 2 — perplexity was the wrong instrument

With the independent probe, forgetting still came out **−32.9%**:

| corpus | base | adapted |
|---|---|---|
| geoscience (domain) | 33.27 | 13.87 (−58.3%) |
| general (ag_news, independent) | 40.12 | 26.94 (**−32.9%**) |

Perplexity improved everywhere. Yet the same checkpoint scored **macro F1
0.002, parse rate 0.015** on the downstream task without SFT.

**Next-token perplexity on plain prose cannot detect the loss of a
behaviour.** For an instruction-tuned starting model the forgetting probe has
to be a capability evaluation.

### Error 3 — comparing across model sizes

On seeing 0.002, this project initially concluded "CPT destroyed
instruction-following, 0.314 → 0.002". **That was wrong**: 0.314 was the *4B*
model. The control shows the untouched 1.7B scores **0.001** zero-shot. CPT
went 0.001 → 0.002 and destroyed nothing; the 1.7B model never had the
capability.

The error was comparing across model sizes — the same confound flagged one
paragraph earlier in the project's own notes, then walked into anyway. It is
why the control arms exist, and why `geocpt transfer` no longer claims its
comparison is apples-to-apples without them.

## 7. What these numbers do not establish

**No forgetting was observed, but it was barely tested.** LR 1e-5, 15%
replay, 1.2M tokens seen — too gentle to do damage. Run
[experiment 3](LEARNING.md#11-the-experiment-grid) (`replay_fraction: 0.0`)
and [experiment 5](LEARNING.md#11-the-experiment-grid) (LR 1e-4) to find the
cliff deliberately.

**The base model was the post-trained one**, not `Qwen3-1.7B-Base`. Running
the pretraining objective over instruction tuning is not the textbook recipe;
`-Base` would be more principled given SFT follows anyway.

**DAPT was never run.** Only TAPT (1.57M tokens). The ~20M-token DAPT corpus
and the `dapt_tapt` sequence remain open.

**Vocabulary extension was never A/B'd** — [experiment 8](LEARNING.md#11-the-experiment-grid),
the question geo-sft left open, and the one CPT is uniquely able to answer
because full fine-tuning trains the embedding matrix.

**Single seed throughout.**

## 8. Run summary

| run | stage | tokens | domain ppl | downstream F1 | wall |
|---|---|---|---|---|---|
| `tapt-v1` | TAPT | 1.57M | 33.27 → 13.87 | 0.842 (with SFT) | 55 min |
| `sft-1.7b-nocpt` | control | – | – | 0.823 | 41 min |
| `sft-after-tapt-v1` | transfer | – | – | 0.842 | 44 min |

## 9. Bugs found while building

| bug | symptom | cost |
|---|---|---|
| `mx.utils` does not exist | `AttributeError` on first parameter count | caught pre-run |
| `save_weights` does not exist | `ImportError` at the first checkpoint | **48-min run lost** |
| `mlx_lm.utils.save()` needs a complete HF snapshot | `IncompleteSnapshotError` on `.gitattributes` | caught by the new test |
| forgetting probe drawn from the replay corpus | −40.5% "forgetting" | wrong conclusion |
| perplexity used as the forgetting metric | missed a capability question entirely | wrong methodology |
| compared 1.7B against 4B | "CPT destroyed instruction-following" | **wrong published claim** |

The first three were caught by testing. The last three were only caught by
running controls — which is the argument for controls.
