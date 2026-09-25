# Results

Everything this project measured on Geoscience Australia data. The earlier
USGS Geolex version — including the two methodology errors it made and what
they taught — is archived in
[`runs/geolex-archive/RESULTS-geolex.md`](../runs/geolex-archive/RESULTS-geolex.md)
and compared against here: same code, same schedule, different corpus.

---

## 1. Setup

| | |
|---|---|
| Hardware | Apple M4 Pro, 48 GB unified memory |
| Base model | `Qwen/Qwen3-1.7B` (bf16, post-trained) |
| Method | full fine-tune, all 1.72B parameters |
| Stage | TAPT — CPT on geo-sft's unlabelled train passages |
| Schedule | 600 iters, batch 1 × accum 16, LR 1e-5 cosine (identical to Geolex) |
| Wall time | 72 min (57 on Geolex; see §4 memory) |

## 2. Corpora

| | TAPT (trained) | DAPT (built, **not trained**) | Geolex TAPT |
|---|---|---|---|
| source | geo-sft train passages | GA eCat abstracts + all of ASUD | geo-sft train passages |
| raw documents | 14,691 | 57,045 | 6,635 |
| quality filter | skipped (task text is the target) | 32.6% kept | skipped |
| near-/exact duplicates removed | 20 | ~3,560 | 17 |
| domain train | 14,371 | 14,650 | 6,318 |
| replay (wikitext, 15%) | 2,536 | 2,585 | 1,114 |
| **approx. train tokens** | **3.26M** | **7.37M** | 1.57M |

The TAPT corpus is **2.1× the Geolex one**. With the schedule held fixed, the
run sees the same ~614k tokens, so it covers 0.19 of an epoch rather than
0.4. More text, same compute.

The DAPT corpus drops every document about geo-sft's 1,442 held-out
(valid/test/gold) units: the whole lexicon contains the test passages, and
pretraining on them before scoring SFT on them is invisible leakage.

Packing: 3,292 blocks of 1,024 tokens, **100% efficiency**, 99.99% retention.

## 3. Training

| iter | domain ppl | general ppl (ag_news) |
|---|---|---|
| 0 | 29.24 | 40.12 |
| 75 | 21.69 | 25.65 |
| 150 | 16.22 | 25.61 |
| 300 | 13.98 | 26.89 |
| 450 | 13.35 | 26.99 |
| 600 | **13.27** | 26.96 |

**Domain perplexity −54.6%** (Geolex: −58.3%), largely converged by
iteration 450 — the same curve on a corpus twice the size.

General perplexity on the independent forgetting probe improved **−32.8%**
(Geolex: −32.9%). As the archived results explain (Error 2), that is not
"no forgetting": next-token perplexity on plain prose cannot detect the loss
of a behaviour, which is why the downstream arms below exist.

## 4. Memory

Peak **53.4 GB** on a 48 GB machine (Geolex: 49.0). The run was meant to
start with LM Studio's models unloaded; `lms unload --all` timed out waiting
for the LM Studio daemon, so the graph lab's 12B chat model stayed resident
and the run swapped. It completed, 26% slower. Unload by hand before a CPT
run.

## 5. Evaluation

### Cloze probes — knowledge, or style?

| model | accuracy | mean margin |
|---|---|---|
| base | 0.720 | +1.38 |
| TAPT | 0.720 | +1.26 |

No change. Only **25 probes** could be built from the held-out text (a
domain term the probe vocabulary recognises, in a usable sentence), so this is
weak evidence — but it matches the Geolex result (0.815 → 0.833 on 54
probes, one extra probe correct): a large fluency gain with no detectable
knowledge gain.

### Downstream transfer — the number that matters

Two arms, both trained and scored by geo-sft's own trainer and evaluator on
identical data (200 held-out test rows, rule labels), with model size held
constant:

| arm | zero-shot | after SFT |
|---|---|---|
| 1.7B + SFT (no CPT) | 0.000 | **0.778** |
| **1.7B + TAPT + SFT** | 0.000 | **0.801** |
| *4B + SFT (geo-sft)* | *0.311* | *0.806* |

**TAPT contributed +0.023 macro F1.** On Geolex the same comparison gave
+0.019 at one seed and **+0.025 mean across three seeds** (95% CI
[+0.012, +0.039]). The effect replicates on a different corpus.

Per field:

| field | no CPT | + TAPT | Δ |
|---|---|---|---|
| chronostrat | 0.759 | 0.811 | **+0.052** |
| minerals | 0.798 | 0.833 | **+0.035** |
| lithologies | 0.836 | 0.855 | +0.019 |
| relations | 0.534 | 0.544 | +0.010 |
| unit_name | 0.985 | 0.995 | +0.010 |
| rank | 0.970 | 0.975 | +0.005 |
| states | 0.963 | 0.963 | 0.000 |
| thickness | 0.965 | 0.965 | 0.000 |

The gains land on the vocabulary-heavy fields — ages and minerals, the
fields whose correctness depends on recognising domain terms — and the
near-saturated fields do not move. On Geolex the largest gains were minerals
and relations; here relations barely moves, and it remains the weakest field
in every arm.

**1.7B + TAPT (0.801) nearly matches 4B without CPT (0.806).** The Geolex
version found the same: the task is not capacity-limited at this scale.

Neither 1.7B model can do the task zero-shot (0.000 — it does not emit JSON),
so the zero-shot column says nothing about what CPT did to instruction
following; the SFT arms are the only valid comparison.

## 6. What these numbers do not establish

- **One seed.** The Geolex effect needed three to be convincing; this is its
  single-seed replication, consistent with it but not independently
  significant.
- **Scored against rule labels, not gold.** The 1.7B arms were not scored on
  geo-sft's reviewed gold set. Relative comparisons between arms are sound;
  the absolute numbers carry the rule-label caveats in geo-sft's RESULTS §3.
- **DAPT was built but not trained.** The 7.4M-token corpus is ready
  (`configs/dapt.yaml`, ~4 h). Whether broad domain text beats the task's own
  text is the lab's most interesting open question.
- **Forgetting was barely tested**: LR 1e-5 and 15% replay are gentle.
  Experiments 3 and 5 in LEARNING.md find the cliff deliberately.

## 7. Run summary

| run | what | wall | result |
|---|---|---|---|
| `runs/tapt-v1` | TAPT, 600 iters | 72 min | domain ppl 29.24 → 13.27 |
| `../geo-sft/runs/sft-after-tapt-v1` | SFT on TAPT model | 67 min | macro F1 0.801 |
| `../geo-sft/runs/sft-1.7b-nocpt` | SFT on plain 1.7B | 67 min | macro F1 0.778 |
