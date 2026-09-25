# geo-cpt

A **continued pre-training** (CPT) lab for geoscience, running on an Apple
Silicon Mac. Sister project to [geo-sft](../geo-sft), and best done after it.

**This repo exists to teach the process of continued pre-training**, not to
ship a model. Corpus acquisition at scale, deduplication, quality filtering,
replay mixing, document packing, full fine-tuning, and the evaluation of
learning *and forgetting* — every stage is a command that writes inspectable
files to disk.

### ▶ Start here: **[docs/LEARNING.md](docs/LEARNING.md)**

## Why this, after geo-sft

geo-sft established that **SFT teaches form far better than facts**. CPT is the
technique for the other half — and it answers two questions that project left
open:

1. **Does more domain text help the downstream task?** geo-sft reached macro F1
   0.806 from SFT alone. `geocpt transfer` runs geo-sft's own trainer and
   scorer on a CPT'd model, so the comparison is apples to apples.
2. **Can vocabulary extension finally pay off?** geo-sft grafted 512 geoscience
   tokens for a 4.89% context saving, but MLX LoRA never touches embeddings so
   the new rows never trained. **Full fine-tuning does train them.**

## CPT, DAPT, TAPT

Not three techniques — one technique pointed at different text.

| | corpus | size here |
|---|---|---|
| **TAPT** | the unlabelled text of the downstream task (geo-sft's ASUD passages) | ~3.3M tokens |
| **DAPT** | broad geoscience domain text (GA eCat abstracts + all of ASUD) | ~7.4M tokens |
| **DAPT→TAPT** | both, in sequence | the full recipe |

One `stage` setting selects which. Run **TAPT first** — it is an evening on
data you already have, and if it moves nothing the big corpus probably will
not either.

## Quick start

```bash
make setup          # venv + deps (pulls in ../geo-sft as an editable dep)
make doctor         # environment, disk, geo-sft data, stage status
make smoke          # tiny end-to-end run on Qwen3-0.6B
make all            # the real TAPT run
```

Requires geo-sft's dataset to exist: run `geosft build` next door first.

## The workflow

| # | stage | command | output |
|---|-------|---------|--------|
| 1 | Corpus | `geocpt corpus` | `data/corpus/<stage>/{train,val,val_general}.jsonl` |
| 2 | Packing | `geocpt pack` | `data/packed/<stage>/*.npy` |
| 3 | Training | `geocpt train` | `runs/<name>/checkpoint-NNNNNN/` |
| 4 | Perplexity | `geocpt eval` | learn vs forget |
| 5 | Probes | `geocpt probe` | knowledge vs style |
| 6 | Transfer | `geocpt transfer` | macro F1 vs the no-CPT 1.7B arm |

## What differs from the SFT lab

| | geo-sft (SFT) | **geo-cpt (CPT)** |
|---|---|---|
| Data | 6,635 labelled pairs | ~10⁴–10⁵ raw documents |
| Labels | rule-derived JSON | **none** |
| Loss | answer only (72% masked) | **every token** |
| Batching | padded pairs | **packed blocks** |
| Method | LoRA, 0.365% of params | **full fine-tune, 100%** |
| LR | 1e-4 | **1e-5** |
| Model | Qwen3-4B (4-bit) | **Qwen3-1.7B (bf16)** |
| Checkpoints | 58 MB adapters | **3.4 GB full models** |
| Signature failure | overfitting | **catastrophic forgetting** |
| Hard part | label quality | **corpus hygiene** |
| Peak memory | 8.5 GB | **41.8 GB** |

## The two numbers that define a run

```
domain perplexity   did it LEARN?
general perplexity  what did it FORGET?
```

Reporting only the first is how people convince themselves a CPT run worked
while quietly degrading the model at everything else. Both are printed at
every evaluation step, with general-set drift coloured red when it grows.

## Documentation

| # | document | contents |
|---|---|---|
| 1 | **[docs/LEARNING.md](docs/LEARNING.md)** | CPT as a subject: objective, packing, forgetting, replay, full-FT vs LoRA, corpus hygiene, evaluation, an 8-experiment grid, guided session, failure modes, glossary |
| 2 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how each stage works in code, what is reused from geo-sft, module reference |
| 3 | [docs/RESULTS.md](docs/RESULTS.md) | every measurement, the three methodology errors, what the numbers do *not* establish |

## Measured results

TAPT on 3.26M tokens of Geoscience Australia lexicon text, Qwen3-1.7B full
fine-tune, 72 minutes.

| | before | after |
|---|---|---|
| Domain perplexity | 29.24 | **13.27** (−54.6%) |
| Cloze probe accuracy | 0.720 | 0.720 (25 probes, no change) |

Downstream, scored by geo-sft's own trainer and evaluator on identical data:

| arm | macro F1 |
|---|---|
| 1.7B + SFT (no CPT) | 0.778 |
| **1.7B + TAPT + SFT** | **0.801** |
| 4B + SFT (geo-sft baseline) | 0.806 |

**TAPT contributed +0.023** with model size held constant, concentrated in
the vocabulary-heavy fields (`chronostrat` +0.052, `minerals` +0.035). One
seed — but it replicates the Geolex version of this lab (+0.019 at one seed,
+0.025 mean over three) on a different corpus.

The warning below still stands for the general case, but at a few million
tokens the technique is slightly more useful than the literature's scale
guidance implies. The DAPT corpus is built and has not been trained.

> Real domain adaptation uses **billions** of tokens. At this scale expect
> small effects, and treat a null result as a finding about scale rather than
> a failure.

### Three methodology errors worth reading

Full detail in [docs/RESULTS.md](docs/RESULTS.md) §6. In short:

1. **The forgetting probe overlapped the replay corpus** — reported −40.5%
   "forgetting" while measuring whether the model had learned wikitext.
2. **Perplexity was the wrong instrument.** With an independent probe,
   general perplexity still *improved* 33% — while the same checkpoint scored
   0.002 macro F1 on the task. Perplexity cannot see the loss of a behaviour.
3. **A comparison across model sizes** produced the wrong published claim
   ("CPT destroyed instruction-following"). The control showed the 1.7B model
   never had the capability: 0.001 before CPT, 0.002 after.

The first was caught by inspection. The other two were only caught by running
controls.

## Hardware notes (M4 Pro, 48 GB)

- Qwen3-1.7B full FT, block 1024: **41.8 GB peak** at batch 2 in the real loop
  (gradient accumulation holds a full extra gradient tree). Use batch 1 for
  headroom; MLX's recommended working set is 40.2 GB.
- ~266 tokens/sec, so a 600-iteration TAPT run is roughly 75 minutes.
- Checkpoints are full models. `keep_last_n` prunes automatically;
  `make clean-ckpt` reclaims the rest.
- The HuggingFace cache is shared with geo-sft — base models download once.

## Licence

[MIT](LICENSE) © 2026 Daniel Bongiorno. Corpus sources: Geoscience Australia's
ASUD and eCat (CC BY 4.0, © Commonwealth of Australia (Geoscience Australia)),
Macrostrat (CC-BY 4.0), wikitext (CC-BY-SA), ag_news. None are redistributed
here. The Geolex-era runs and results are in `runs/geolex-archive/`.
