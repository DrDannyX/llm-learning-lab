# geo-sft

A complete, runnable supervised fine-tuning (SFT) workflow for a **geoscience
structured-extraction** task, built to run end to end on an Apple Silicon Mac.

**This repo exists to teach the process of supervised fine-tuning**, not to ship a
model. Corpus acquisition, training-pair extraction, tokenizer work, QLoRA
training, live monitoring, and honest evaluation — every stage is a separate
command that writes inspectable files to disk, and every non-obvious decision is
explained where it is made.

### ▶ Start here: **[docs/LEARNING.md](docs/LEARNING.md)**, then follow the
[learning path](#how-to-learn-from-this-repo) below.

```
messy Australian lexicon prose  ──►  strict JSON
```

> *"Culvida Sandstone. Up to 210m thick; fine to coarse sandstone; siltstone;
> granule and pebble conglomerate (poorly sorted); fluvial. Conformably overlies
> Erskine Sandstone. Type section 1km south of Culvida Soak (CORNISH: 20deg
> 14' 00" S, 126deg 56' 00" E). Plant fossils."*

```json
{"unit_name": "Culvida Sandstone", "rank": "Formation",
 "lithologies": ["conglomerate", "sandstone", "siltstone"], "chronostrat": [],
 "minerals": [], "thickness": {"min_m": 210.0, "max_m": 210.0},
 "relations": [{"kind": "overlies", "unit": "Erskine Sandstone"}],
 "states": []}
```

## Why this task

Extraction was chosen over geoscience Q&A deliberately: **it is objectively
gradable.** Field-level precision/recall against a fixed schema tells you
whether the fine-tune worked. A Q&A fine-tune would leave you grading style with
an LLM judge and guessing.

## Quick start

```bash
make setup          # uv venv + all dependencies
make doctor         # hardware, libraries, disk, data
make smoke          # tiny end-to-end run (Qwen3-1.7B, 60 iters, a few minutes)
make all            # the real run  (Qwen3-4B, 900 iters, ~2 hours)
```

Run `make smoke` first. It exercises every stage on a small model so a typo
costs you three minutes instead of two hours.

## How to learn from this repo

**Do not read the docs front to back.** Most of the material is reference, and
reading about a loss curve teaches you far less than watching one. Interleave
reading with running — four sessions, most of the time unattended.

### Session 1 — concepts, and prove the wiring (~1 h, mostly waiting)

Read **[LEARNING.md](docs/LEARNING.md) §1–5**. This is the conceptual core and
the only part worth reading carefully before you touch anything:

| § | topic |
|---|---|
| 1 | what SFT can and cannot teach (vs prompting, RAG, RLHF, pretraining) |
| 2–3 | what LoRA and QLoRA physically do |
| **4** | **the anatomy of a training example — the most important section here** |
| 5 | why loss is a diagnostic, not a metric |

Then skim this README for the shape of the project, and run:

```bash
make doctor
make smoke                 # ~5 min, every stage on a 1.7B model
geosft inspect --index 3
```

Sit with that `inspect` output next to §4 until the masked/trained split
clicks. That one screen is where most fine-tuning failures live.

### Session 2 — hyperparameters and the real run (~2 h, unattended)

Read **§6–8** (steps vs iterations vs epochs vs optimizer steps; what each knob
physically does; how to read a curve). Then start the run and *watch the live
panel* for the first few minutes — you want to see the steep phase, the model
learning JSON shape, happen in real time.

```bash
make train
```

While it runs, read **[RESULTS.md](docs/RESULTS.md) §1–5** and compare its
curve against yours.

### Session 3 — evaluation (~30 min)

Read **§9** (five rules for honest evaluation), then:

```bash
geosft eval --adapter runs/<name>
```

Read the **delta** column, not the tuned column. Then **RESULTS.md §6–7**.

Now do the step most people skip: open `runs/<name>/predictions_tuned.jsonl`,
find a case where the model and the gold label disagree, and decide which is
right. **Sometimes the model is.** That is generalisation past the rules, and
it is the most encouraging thing you will see.

### Session 4 — your own experiment

Read **[EXPERIMENTS.md](docs/EXPERIMENTS.md) #1** (prompt masking) and run it.
It is the most instructive hour available: the unmasked run reaches *lower*
training loss and *worse* task scores, which makes §5 permanent.

### Read differently

| document | how to read it |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | **Lookup, not cover to cover.** Go when you want a specific stage, or before changing code. |
| LEARNING.md §10–11 | Data/weak supervision and tokenizers — read when you reach those stages, not upfront. |
| LEARNING.md §13–14 | Failure-mode table and glossary. Bookmarks. Go to §13 the moment something looks wrong. |
| [RESULTS.md](docs/RESULTS.md) §7 | "What these numbers do *not* establish." Read early — it is the antidote to believing the 0.807. |

LEARNING.md §12 is the same journey as sessions 1–3 but command-by-command
rather than reading-first. Use it as the detailed companion, not a second path.

### If you would rather read code than prose

Four files, in this order:

1. `src/geosft/schema.py` — the contract everything else points at (~130 lines)
2. `src/geosft/data/label.py` — where data quality is won or lost
3. `src/geosft/train/mlx_train.py` — the training loop and its two documented traps
4. `src/geosft/eval/metrics.py` — what "good" actually means here

## The workflow

| # | Stage | Command | Output |
|---|-------|---------|--------|
| 1 | Corpus + gazetteers | `geosft fetch` | `data/interim/passages.jsonl`, `vocab.json` |
| 2 | Pair extraction + splits | `geosft build` | `data/processed/{train,valid,test}.jsonl`, `data/gold/` |
| 3 | Tokenizer + fertility | `geosft tokenizer` | `artifacts/tokenizers/`, `tokenizer_report.json` |
| 4 | Vocabulary extension | `geosft extend` | `artifacts/models/*-geo-ext`, `*-mlx-4bit` |
| 5 | QLoRA training | `geosft train` | `runs/<name>/adapters.safetensors`, `loss_curve.png` |
| 6 | Evaluation | `geosft eval --adapter runs/<name>` | `eval_report.json`, `predictions_*.jsonl` |

Between steps 2 and 5, run **`geosft inspect`**. It prints the prompt (masked)
against the loss region (trained) for one example and asserts the two
invariants that silently wreck fine-tunes: the completion ends with EOS, and
the training prompt is byte-identical to the inference prompt. It takes a
second and catches a whole family of bugs.

### 1. Corpus

**Geoscience Australia's Australian Stratigraphic Units Database (ASUD)**
(CC BY 4.0) — the national authority on Australian stratigraphic names, 18,387
units. The lab reads its documented WFS service for the unit index and its
weekly state-report downloads for the prose: definition cards written by the
units' authors, and ~350,000 short notes on how each published reference uses
a unit. Telegraphic, abbreviated (`qtz-rich`, `Sltst`, `Gp`), full of house
style: exactly the text a domain model has to survive and a general instruct
model handles badly. **22,311 passages from 8,094 units** — 2.5× the Geolex
version of this lab.

**Macrostrat API** (CC-BY 4.0) supplies the closed vocabularies: 214 lithologies,
514 chronostratigraphic intervals, 6,350 minerals. Australian spellings
(*Palaeozoic*, *Archaean*, *Lower Devonian*) are mapped onto them.

The ASUD download is **snapshotted** with SHA-256 hashes
(`data/raw/asud/manifest.json`): GA rebuilds the reports weekly, so
re-downloading is not reproducing.

### 2. Training pairs

A rule-based labeller (`src/geosft/data/label.py`) turns each passage into a
`GeoExtraction` object. Two rules govern it:

- **Only facts present in the passage.** ASUD offers curated ages, states,
  thickness and even stratigraphic relations, and using them would teach the
  model to state things it cannot see — i.e. to hallucinate confidently. The
  metadata is used only to *audit* the labeller (`agreement_report`).
- **Canonical targets.** Lists are sorted and de-duplicated. If identical facts
  could serialise two ways, you are training the model to predict a coin flip.

Splitting is **group-wise on `unit_id`**. ASUD files up to a dozen overlapping
notes under one unit; a random split would put near-duplicates on both sides of
the wall and inflate the test score. Once the gold set is reviewed it is
**frozen**: later labeller fixes cannot reshuffle it.

### 3. Tokenizer

> **A from-scratch tokenizer cannot be dropped into a pretrained LLM.** Token IDs
> are row indices into the embedding matrix. Swapping the tokenizer re-points
> every index at an unrelated vector and destroys the model; recovering needs
> continued *pretraining*, not SFT.

So the tokenizer does the two jobs it legitimately can:

1. **Measurement.** Fertility (tokens per word) on geoscience text vs general
   English. Measured on this corpus against Qwen3's tokenizer:

   | corpus | base tokenizer | geo tokenizer |
   |---|---|---|
   | geoscience passages | 1.921 | **1.610** (−16%) |
   | general English | **1.123** | 1.877 (+67%) |

   That second row is the whole argument. The domain tokenizer wins modestly on
   domain text and loses catastrophically everywhere else — which is what a
   wholesale tokenizer swap would cost you.

   1,923 of 2,228 domain terms cost the base tokenizer 3+ tokens. Ranked by
   tokens actually wasted (frequency × pieces−1): `siltstone` (3 tokens,
   2,951 uses), `granodiorite` (4×1,858), `Biotite` (3×1,406), `rhyolite`
   (4×1,135), `gneiss`, `monzogranite`, `mafic`, `dolerite` — Australian
   lexicon prose is dense with igneous rock names. Ranking by raw
   fragmentation instead surfaces junk like
   `Clino-ferro-ferri-fluoro-holmquistite` (16 tokens, zero occurrences).

2. **Candidate mining** for step 4.

### 4. Vocabulary extension (the part that actually feeds training)

Rather than replacing the tokenizer, `geosft extend` **adds** the top-K domain
terms to the base vocabulary and initialises each new embedding row as the
**mean of the sub-word embeddings it replaces**. Nothing existing is disturbed.

Measured on this corpus (512 tokens grafted onto Qwen3-4B):

```
 base    (18 tokens): Penn|s|ylv|anian| s|ilt|stone| uncon|form|ably| over|lying| Ord|ov|ician| g|ne|iss
 extended (7 tokens): Pennsylvanian| siltstone| unconformably| over|lying| Ordovician| gneiss
```

Across 200 held-out passages that is **4.89% fewer tokens**. Real, but modest —
and notice it is a *cost* saving, not a *quality* gain. Keeping those two
straight is the point of experiment #5.

Four traps this handles, all of which fail *silently*:

1. **Shrinking the embedding matrix.** Qwen3-4B declares `vocab_size=151936` but
   its tokenizer has 151,669 entries — 267 spare rows. The reflex
   `model.resize_token_embeddings(len(tok))` would **delete** 267 rows including
   live special tokens. This code only ever grows.
2. **Tied embeddings.** Qwen3 ties `lm_head` to `embed_tokens`; treating them as
   separate matrices double-writes or leaves `lm_head` stale.
3. **Random init.** Default new rows land off-manifold. Mean-of-subwords starts
   each token at the centroid of the pieces it replaces.
4. **Wrong casing.** Added tokens match case-sensitively (`normalized=False`).
   Mining candidates from a lowercased frequency table grafts
   `pennsylvanian`, which never fires — the word is a proper noun and appears
   as `Pennsylvanian` every single time. The highest-value terms
   (`Cretaceous`, `Ordovician`, `Mississippian`) are all proper nouns, so this
   one mistake halves the benefit: 2.54% saving before the fix, 4.89% after.
   We graft the dominant surface form actually observed in the corpus.

A fifth trap is environmental rather than conceptual: transformers 5.x rewrites
`rope_theta` into a nested `rope_parameters` on save, and mlx-lm 0.31 still
reads the flat key, so quantising a re-saved model dies with a misleading
`ModelArgs.__init__() missing 1 required positional argument: 'rope_theta'`.
The weights are fine; only metadata moved. `patch_config_for_mlx` restores the
legacy keys automatically.

**Expect this step to be a null result, and treat that as a finding.** Vocabulary
extension is a pretraining-scale technique; on a few thousand SFT examples the new
rows get very little gradient. Worse, on the MLX path they get *none* — MLX LoRA
adapts attention and MLP projections, not embeddings, so the grafted rows keep
their mean-init values. Only the HF path can train them
(`--train-embeddings`, via PEFT `modules_to_save`). Running the A/B and reading
the result is the lesson; `make all --skip-extend` gives you the control arm.

### 5. Training

**MLX is the primary path.** The base model is loaded already quantised to 4 bits
and stays frozen; only the LoRA matrices train. Memory scales with the adapter,
not the model — that is what QLoRA buys you, and why a 4-bit base is the point
rather than a compromise. MLX is typically 2–3× faster than PyTorch/MPS here.

**The HF PEFT/TRL path is *not* QLoRA on a Mac.** bitsandbytes has no working MPS
backend, so it runs bf16 LoRA on a full-precision base. It is kept because the
API is what you will meet everywhere else, and because it is the only path that
can train grafted embeddings. The CUDA QLoRA block is in the file, commented.

Knobs that matter, roughly in order:

| Knob | Why |
|---|---|
| `mask_prompt: true` | Loss on the JSON only. Without it most of your gradient teaches the model to reproduce the input passage. **Biggest single lever.** |
| `num_layers` | Blocks (from the top) that get adapters. 16 is a good default; `-1` is everything. |
| `lora_rank` | 8–16 is plenty for a format/extraction task. Raise only if train loss plateaus high. |
| `learning_rate` | `1e-4` is the LoRA default. `1e-3` often diverges; `1e-5` looks like nothing is happening. |
| `batch_size × grad_accumulation_steps` | Your effective batch. Prefer raising accumulation — it costs time, not RAM. |

> **The gradient-accumulation trap.** MLX indexes the LR schedule by *optimizer
> steps*, and with accumulation there is one optimizer step every
> `grad_accumulation_steps` iterations. Passing iteration counts straight
> through stretches warmup by that factor. With `grad_accumulation_steps: 8`, a
> `warmup_steps: 30` config warms up over **240** iterations, so a 600-iteration
> run spends 40% of its life at an effectively zero LR and the loss sits flat —
> looking exactly like "LoRA doesn't work on my data". This repo hit that bug
> for real: loss was stuck at 2.26 for 40 iterations, and after the fix it fell
> to 0.52 over the same span. Config values here are in **iterations** and
> converted internally.

### 6. Monitoring

A live panel shows iteration, train/val loss with sparklines, best checkpoint,
train↔val gap, tokens/sec, peak memory and ETA. It raises alarms for the four
failure modes worth catching early:

- loss is NaN → LR too high
- train loss flat from step 1 → adapters not attached, or LR ≈ 0
- val loss not improving for N evals → overfitting, stop and take the best checkpoint
- val − train gap widening → memorisation

Everything lands in `runs/<name>/metrics.jsonl` plus a `loss_curve.png`.

### 7. Evaluation

**The headline number is the delta, not the score.** `geosft eval` always scores
the untuned base on the same prompts. A tuned macro-F1 of 0.71 means nothing
until you know the base scored 0.42 (a real win) or 0.70 (you burned an afternoon).

Two gates are reported separately from content, because they fail differently
and a large part of what SFT buys is format compliance:

- `parse_rate` — did it emit parseable JSON at all?
- `schema_valid_rate` — did that JSON satisfy the schema?

Then per field: exact match for `unit_name`, accuracy for `rank`, tolerance-based
match for `thickness`, and micro P/R/F1 for the set-valued fields and `relations`.
An unparseable answer is scored as an empty prediction, never skipped — otherwise
a model is rewarded for refusing.

## Measured results

Qwen3-4B-Instruct-2507 4-bit, rank-16 LoRA on 16 layers, 14,691 training
pairs from ASUD, scored against the **untuned base on the same prompts**, on
the held-out test split (group-split by unit) and on the 150 reviewed gold rows.

| metric | base (gold) | tuned (gold) | delta |
|---|---|---|---|
| schema valid rate | 0.867 | **0.993** | +0.127 |
| unit_name accuracy | 0.993 | **0.993** | +0.000 |
| rank accuracy | 0.887 | **0.973** | +0.087 |
| thickness accuracy | 0.687 | **0.947** | +0.260 |
| lithologies F1 | 0.359 | **0.825** | +0.466 |
| chronostrat F1 | 0.120 | **0.667** | +0.546 |
| minerals F1 | 0.554 | **0.854** | +0.300 |
| states F1 | 0.126 | **1.000** | +0.874 |
| relations F1 | 0.438 | **0.691** | +0.253 |
| **macro F1** | **0.320** | **0.807** | **+0.488** |

Against rule labels on 200 test rows: **0.311 → 0.806**. The Geolex version of
this lab scored 0.314 → 0.831 against rules but only **0.705** against its
reviewed gold; see [RESULTS §3 and §5](docs/RESULTS.md) for why the gap closed
here and why that is partly leakage from the gold review itself.

167 minutes on an M4 Pro (GPU shared with the graph lab), 13.5 GB peak. Best
validation at iter 824; a false "overfitting" alarm fired at 674 on noisy
20-batch validation.

`relations` is the weakest field and `chronostrat` the most improved. Both are
limited by the labeller: coordinated lists ("overlain by A, B and C") defeat
the relation rules, and neighbouring units' ages defeat the chronostrat
gazetteer. Against gold, the tuned model scores slightly *below* the fixed
rules on every field — it imitates its teacher, imperfectly.

### The bug this table found (Geolex-era run)

The first full run of the Geolex version of this lab scored **strict JSON rate 1.000 on the base and 0.000 on the
tuned model** while macro F1 went *up*. Cause: Qwen3's chat template injects
`<think>\n\n</think>\n\n` before assistant content in a full conversation but
not in the generation prompt MLX derives the loss mask from, so the scaffolding
landed inside the trained region and the model faithfully learned to emit it.

Masking it is not enough — masked tokens stay in the *input*, so the model would
be trained to produce JSON conditioned on a prefix absent at inference.
`PromptCompletionDataset` fixes it by building `prompt + answer + EOS` directly,
byte-identical to what evaluation feeds the model. After the fix, strict JSON
rate is **1.000** and output is bare JSON.

Note the two runs' *loss values* are not comparable (different token sets in the
denominator — initial val is 2.266 vs 1.766). Compare the eval metrics, not the
loss.

## The honesty problem, and what to do about it

Rule-derived labels have a ceiling: **a model trained on them learns to imitate
the rules**, so its score against the rules approaches 100% while proving nothing.

Two things keep this project honest:

1. The rules are high-precision / low-recall by design, so the model must
   *generalise* past the gazetteer to score well on unseen units.
2. `data/gold/gold.jsonl` — 150 rows drawn from test, never trained on, meant to
   be **corrected by hand**:

```bash
geosft review --n 20        # print rows for checking
# edit data/gold/gold.jsonl, set "reviewed": true
geosft eval --adapter runs/<name> --gold
```

Until rows are marked reviewed, `--gold` warns you that it is scoring rules
against rules. **Read the gold numbers, not the rule-matched numbers.**

## Documentation

See [How to learn from this repo](#how-to-learn-from-this-repo) for the
recommended path through these. In brief:

| # | document | contents |
|---|---|---|
| 1 | **[docs/LEARNING.md](docs/LEARNING.md)** | **SFT as a subject** — concepts, mental models, the anatomy of a training example, reading a loss curve, honest evaluation, guided first session, failure-mode table, glossary |
| 2 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how every stage works in code, the design decisions, module reference |
| 3 | [docs/RESULTS.md](docs/RESULTS.md) | full training outcomes, evaluation tables, the four bugs and their fixes |
| 4 | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | eight experiments worth running, and what to expect from each |

### The four bugs are teaching material

Building this hit four bugs that each produced a **plausible-looking loss curve
and no error message** — which is the normal failure mode in this field. Each is
documented where it lives, carries a regression test, and is written up in
LEARNING.md and RESULTS.md:

| bug | symptom |
|---|---|
| `re.IGNORECASE` defeating `[A-Z]` | relation names swallowed clauses |
| gradient-accumulation LR trap | loss flat at 2.26 for 40 iters; 0.52 after the fix |
| lowercase vocabulary grafting | added tokens existed but never fired |
| template scaffolding inside the loss region | strict JSON rate 1.000 → 0.000 **while macro F1 rose** |

## Layout

```
configs/         default.yaml (full run), smoke.yaml (fast wiring test)
src/geosft/
  schema.py      the extraction contract + prompt construction (one source of truth)
  config.py      typed config for every stage
  data/          ASUD access + snapshot, gazetteers, passages, matcher, labeller, splits
  tokenizer/     train from scratch, fertility analysis, vocabulary extension
  train/         mlx_train.py (QLoRA, primary), hf_train.py (PEFT/TRL, portable)
  monitor/       live tracker, alarms, JSONL metrics, loss curves
  eval/          per-field metrics, base-vs-tuned driver
tests/           21 tests, incl. regressions for four real bugs found while building
docs/            ARCHITECTURE.md, RESULTS.md, EXPERIMENTS.md
runs/            one directory per run: metrics.jsonl, loss_curve.png, eval_report.json
```

## Hardware notes (M4 Pro, 48 GB)

- Qwen3-4B 4-bit + rank-16 LoRA on 16 layers: comfortable, ~10–14 GB peak.
- `make smoke` on Qwen3-1.7B: a few GB.
- Budget ~40–60 GB of disk for weights, quantised copies and checkpoints.
  `make clean-models` reclaims it. Model weights live in `artifacts/hf-cache`
  (override with `GEOSFT_HF_HOME`).
- To go bigger: Qwen3-8B 4-bit fits, but drop `batch_size` to 1 and raise
  `grad_accumulation_steps` to keep the effective batch.

## Licences

**This code: [MIT](LICENSE)** © 2026 Daniel Bongiorno.

Everything it downloads carries its own licence, and none of it is
redistributed here:

| what | licence | note |
|---|---|---|
| ASUD (Geoscience Australia) | CC BY 4.0 | attribute "© Commonwealth of Australia (Geoscience Australia)" |
| Macrostrat gazetteers | CC-BY 4.0 | attribute Macrostrat if you republish derived data |
| Qwen3 model weights | Apache-2.0 | fetched from HuggingFace at run time |

The committed `data/gold/gold.jsonl` is derived from ASUD (CC BY 4.0,
© Commonwealth of Australia (Geoscience Australia)). The Geolex-era gold set
and runs are kept for reference in `runs/geolex-archive/`.
Training splits, model weights and quantised copies are gitignored and
regenerated by the pipeline.
