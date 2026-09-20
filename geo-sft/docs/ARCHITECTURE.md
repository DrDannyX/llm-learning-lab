# Architecture

How geo-sft works, stage by stage. Read [README.md](../README.md) first for the
what and why; this document is the how.

---

## Data flow

```
 USGS Geolex API                 Macrostrat API
 (16,684 stratigraphic units)    (lithologies, intervals, minerals, strat names)
        │                                │
        │  data/fetch.py                 │  data/vocab.py
        ▼                                ▼
 data/interim/passages.jsonl      data/interim/vocab.json
        │                                │
        └──────────────┬─────────────────┘
                       │  data/label.py   (rule-based weak supervision)
                       ▼
                GeoExtraction objects
                       │  data/build.py   (dedupe, filter, group-split)
                       ▼
     data/processed/{train,valid,test}.jsonl   +  .meta.jsonl
     data/gold/gold.jsonl  (150 rows, hand-correctable)
                       │
       ┌───────────────┼───────────────────────┐
       │               │                       │
       ▼               ▼                       ▼
 tokenizer/train  tokenizer/extend        train/mlx_train
 (measurement)    (embedding surgery)     train/hf_train
       │               │                       │
       ▼               ▼                       ▼
 tokenizer_report  *-geo-ext-mlx-4bit     runs/<name>/adapters.safetensors
                                                │
                                                ▼  eval/evaluate.py
                                          runs/<name>/eval_report.json
```

Every arrow is a file on disk. Any stage can be re-run without repeating the
ones before it, which is the property that makes iteration bearable.

---

## The central contract: `schema.py`

Everything in the project points at one object, `GeoExtraction`. The labeller
produces it, the prompt builder serialises it, the trainer learns it, the
evaluator scores it. Keeping it in one file is what stops those four from
drifting apart.

```python
class GeoExtraction(BaseModel):
    unit_name:   str | None          # exact match
    rank:        RankT               # closed 9-way label
    lithologies: list[str]           # closed vocab, set-scored
    chronostrat: list[str]           # closed vocab, set-scored
    minerals:    list[str]           # closed vocab, set-scored
    thickness:   Thickness | None    # numeric, tolerance-scored
    relations:   list[Relation]      # (kind, unit) pairs, set-scored
    states:      list[str]           # postal codes, set-scored
```

Two properties are load-bearing:

**Canonical serialisation.** Every list field is sorted and de-duplicated by a
Pydantic validator. If the same facts could serialise two ways, you are
training the model to predict a coin flip; determinism in the target is free
accuracy. `test_lists_are_canonical` guards this.

**One prompt builder.** `build_messages(passage, target=None)` is called by
both the dataset builder and the evaluator. Train/inference prompt asymmetry is
the single most common silent failure in fine-tuning, so there is exactly one
function that can produce a prompt. `test_prompt_symmetry` guards this.

The field mix is deliberate: a scalar string, a closed categorical, four
set-valued fields, a nullable numeric, and a nested relational list. Each needs
a different metric, which is the point — it teaches you that "accuracy" is not
one number.

---

## Stage 1 — corpus (`data/fetch.py`, `data/vocab.py`, `data/http.py`)

**Source: USGS Geologic Names Lexicon (Geolex).** A US Government work, so
public domain. Each of ~16,700 stratigraphic units carries several *reference
summaries* — prose written by geologists between roughly 1890 and 1990,
abbreviated and inconsistent, e.g.

> *Aarde Shale Member (restricted) of Howard Limestone of Wabaunsee Group. The
> hard dense limestone and black fissile shale in the upper part of the Aarde
> are reallocated to newly named Wauneta Limestone… 2 to 7 feet thick…*

This is deliberately *not* clean text. A general instruct model handles it
badly, which is exactly what leaves room for a fine-tune to show a real gain.

**Source: Macrostrat** (CC-BY 4.0) supplies four closed vocabularies: 214
lithologies, 509 chronostratigraphic intervals, 6,350 minerals, 45,347
stratigraphic names.

**`CachedClient`** wraps both with a disk cache keyed by URL+params hash, a
politeness delay, exponential-backoff retries, and atomic writes (a `.tmp`
file then `replace`, so concurrent workers never read a torn file). Detail
fetches run through a 6-worker thread pool: serially it is ~1.7 s per unit,
nearly two hours for 4,000 units, almost all of it network wait.

The cache is why you can iterate on extraction rules a hundred times without
touching the network again — and the rules are the part you actually iterate on.

**Gotcha handled:** Macrostrat carries planetary timescales. `Amazonian`,
`Noachian` and `Hesperian` are *Martian* periods and would be nonsense labels
on a USGS corpus. They are filtered by inspecting the nested `timescales[].name`
— note that the naive substring check for `"mars"` fails, because the string is
`"Martian"`.

## Stage 2 — training pairs (`data/label.py`, `data/match.py`, `data/build.py`)

### The labelling rule

> **A target field may only contain facts present in the passage itself.**

Geolex offers tempting curated metadata (authoritative ages, state lists), but
training on facts the model cannot see teaches it to state things confidently
without evidence. The metadata is therefore used *only* to audit the labeller,
never to write labels.

`agreement_report()` exploits that separation: because the labeller never reads
the curated fields, they act as independent ground truth. On the full corpus:

| check | value | reading |
|---|---|---|
| unit_name retained | 85.5% | name appears early in its own passage |
| chronostrat consistency | 89.5% | rule-extracted ages agree with curated ages |
| states precision | 0.581 | **expected to be low — see below** |
| states recall | 0.233 | **expected to be low — see below** |

The states numbers are low *by design*, not by defect. We extract states
*named in the passage*; Geolex curates every state the unit occurs in across
all references. Different questions. The number is a drift tripwire, not an
accuracy score — if it moves sharply after you edit the rules, something broke.

### How the rules work

`Gazetteer` (`data/match.py`) tokenises once and looks up word n-grams in a
dict, longest match first. A 6,000-entry regex alternation would be slow and
fragile; this handles multi-word terms (`lime mudstone`) and gives longest-match
semantics for free, so `Late Cretaceous` beats `Cretaceous`.

`parse_thickness` only trusts a measurement in a sentence containing
`thick`/`thickness` — lexicon prose is full of distances, elevations and page
numbers, and that keyword is what separates a thickness from a road log. Feet
are converted to metres.

`extract_relations` matches trigger phrases (`overlies`, `underlies`, `below`,
`above`, `grades into`, `intertongues with`, …) followed by a capitalised name,
then **validates the head word against the 45k-name Macrostrat gazetteer** so
we do not capture ordinary capitalised English.

> **Bug found here.** These patterns are compiled *without* `re.IGNORECASE`.
> The name group relies on `[A-Z]` to find proper nouns, and a global
> ignorecase flag silently turns that into "any letter" — so the group
> swallowed whole clauses, yielding `"Church member of Howard limestone"`
> instead of `"Church Member"`. Trigger words that genuinely need
> case-insensitivity use scoped `(?i:...)` groups instead. Guarded by
> `test_relation_names_are_not_greedy`.

`infer_rank` reads the rank word following the unit name, with explicit ranks
(`Member`, `Group`) beating the lithology-as-rank fallback (`Austin chalk` →
Formation). Order matters: `Aarde shale member of Howard limestone` must
resolve to Member, not Formation.

### Splitting

**Group-wise on `unit_id`.** Geolex carries several reference summaries per
unit and they overlap heavily — seven passages all describing the Aarde Shale
Member. A random split puts near-duplicates on both sides of the wall and
inflates the test score badly. Every passage about a unit lands in exactly one
split.

Passages are also de-duplicated by normalised-text SHA1, and rows where the
labeller found fewer than `min_filled_fields` are dropped — a target that is
almost all nulls teaches the model to say nothing.

Output is `{"messages": [...]}` chat rows, a format both mlx-lm and TRL accept,
so one file feeds both backends and they cannot diverge on data. A parallel
`.meta.jsonl` keeps the passage, the parsed target and the source URL for
evaluation and review.

## Stage 3 — tokenizer (`tokenizer/train.py`, `tokenizer/analyze.py`)

A from-scratch tokenizer **cannot** be dropped into a pretrained LLM: token IDs
are embedding-matrix row indices, so swapping the tokenizer re-points every
index at an unrelated vector. Recovery needs continued *pretraining*, not SFT.

So this stage does the two jobs it legitimately can:

1. **Measurement** — fertility (tokens per word) on domain vs general text.
2. **Candidate mining** — its merges shortlist terms worth grafting in stage 4.

It trains on the **train split only**. A tokenizer fitted on test text has seen
test text and its compression numbers become self-congratulatory.

The fragmentation report ranks by **tokens wasted = frequency × (pieces − 1)**,
not raw fragmentation. Raw fragmentation surfaces junk like
`Clino-ferro-ferri-fluoro-holmquistite` (16 tokens, zero occurrences); the
weighted ranking surfaces `Pennsylvanian` (4 tokens × 340 uses).

## Stage 4 — vocabulary extension (`tokenizer/extend.py`)

Adds the top-K domain terms to the base vocabulary and initialises each new
embedding row as the **mean of the sub-word embeddings it replaces**. Existing
tokens are untouched, so nothing is destroyed.

Five traps, all of which fail silently:

| # | Trap | Handling |
|---|---|---|
| 1 | **Shrinking embeddings.** Qwen3-4B declares `vocab_size=151936` but its tokenizer has 151,669 entries — 267 spare rows. `resize_token_embeddings(len(tok))` would *delete* 267 rows including live special tokens. | Only ever grow; pad to a multiple of 64. |
| 2 | **Tied embeddings.** Qwen3 ties `lm_head` to `embed_tokens`. | Detect `tie_word_embeddings` and write once. |
| 3 | **Random init.** New rows land off-manifold. | Mean-of-subwords, computed in fp32. |
| 4 | **Wrong casing.** Added tokens match case-sensitively (`normalized=False`). Mining from a lowercased frequency table grafts `pennsylvanian`, which never fires — the term is a proper noun. | Graft the dominant surface form observed in the corpus. Worth 2.54% → 4.89%. |
| 5 | **Config drift.** transformers 5.x rewrites `rope_theta` into nested `rope_parameters` and `torch_dtype` into `dtype`; mlx-lm 0.31 reads the flat keys and dies with a misleading `ModelArgs.__init__() missing 'rope_theta'`. | `patch_config_for_mlx` restores the legacy keys. |

`AddedToken(term, lstrip=True, single_word=True, normalized=False)` is the
correct construction: `lstrip` absorbs the leading space (1 token instead of
2), `single_word` stops `sand` matching inside `sandstone`.

## Stage 5 — training (`train/mlx_train.py`, `train/hf_train.py`)

### MLX (primary)

The base is loaded already quantised to 4 bits and stays frozen; only the LoRA
matrices train. Memory scales with the adapter, not the model — that is what
QLoRA buys, and why a 4-bit base is the point rather than a compromise.

Two traps this file exists to document:

> **The gradient-accumulation trap.** MLX indexes the LR schedule by
> **optimizer steps**, and with accumulation there is one optimizer step every
> `grad_accumulation_steps` iterations. Passing iteration counts straight
> through stretches warmup by that factor: `grad_accumulation_steps: 8` with
> `warmup_steps: 30` warms up over **240** iterations, so a 600-iteration run
> spends 40% of its life at effectively zero LR. Observed here: loss pinned at
> 2.26 for 40 iterations; after the fix it fell to 0.52 over the same span.
> Config values are in **iterations** and converted internally.

> **The template-prefix trap.** Qwen3's chat template injects
> `<think>\n\n</think>\n\n` before assistant content in a *full conversation*
> but not in the *generation prompt* MLX derives the mask offset from — so the
> scaffolding lands inside the trained region and the model learns to emit it.
> Masking is not enough: masked tokens remain in the *input*, so the model
> would be trained to produce JSON conditioned on a prefix absent at
> inference. `PromptCompletionDataset` builds `prompt + answer + EOS`
> directly, byte-identical to what evaluation feeds the model.

`CacheDataset` must wrap the dataset before `train()` — it memoises
`apply_chat_template` per example and exposes the tuple interface the trainer's
length-sorting needs. Without it you get a bare `KeyError: 0`.

### HuggingFace PEFT/TRL (portable)

**Not QLoRA on a Mac.** bitsandbytes has no working MPS backend, so this runs
bf16 LoRA on a full-precision base. The CUDA QLoRA block is present, commented.
It is kept for two reasons: PEFT/TRL is the API you meet everywhere else, and
it is the **only path that can train grafted embedding rows**
(`modules_to_save=["embed_tokens", "lm_head"]`). MLX LoRA adapts attention and
MLP projections only.

Two MPS-specific fixes: `enable_input_require_grads()` (gradient checkpointing
plus a frozen base otherwise crashes with *element 0 of tensors does not
require grad*), and runtime bf16 probing rather than assuming support.

Uses prompt/completion rows rather than `assistant_only_loss`, which requires
`{% generation %}` markers many templates lack and fails mid-run.

## Stage 6 — monitoring (`monitor/tracker.py`)

Backend-agnostic: both trainers feed one `RunTracker`. A loss number scrolling
past at 10 Hz teaches nothing; the panel shows iteration, train/val loss with
sparklines, best checkpoint, train↔val gap, tokens/sec, peak memory and ETA.

Four alarms, for the four failure modes worth catching early:

| symptom | alarm |
|---|---|
| loss is NaN | diverged — lower the LR |
| train loss flat from step 1 | adapters not attached, or LR ≈ 0 |
| val not improving for `patience` evals | overfitting — take the best checkpoint |
| val − train gap > 0.5 | memorising |

Everything is appended to `runs/<name>/metrics.jsonl`, so runs are comparable
after the fact; `loss_curve.png` and `summary.json` are written on close —
including on failure, via a `finally` block.

## Stage 7 — evaluation (`eval/metrics.py`, `eval/evaluate.py`)

**The headline number is the delta, not the score.** Evaluation always scores
the untuned base on identical prompts. A tuned macro-F1 of 0.83 means nothing
until you know the base scored 0.31.

Two gates are reported separately from content, because they fail differently
and a large part of what SFT buys is format compliance:

- `parse_rate` — did it emit parseable JSON at all (after stripping fences and
  `<think>` blocks)?
- `strict_json_rate` — does `json.loads(raw)` succeed with no recovery?

Then per field: exact match for `unit_name`, accuracy for `rank`, 5%-tolerance
match for `thickness`, micro P/R/F1 for the four set fields and for `relations`
(as `(kind, unit)` pairs, normalised).

Two scoring decisions worth knowing:

- **An unparseable answer is scored as an empty prediction**, never skipped.
  Every gold item becomes a false negative. Skipping would reward a model for
  refusing to answer.
- **A field with no gold and no predictions anywhere is excluded from the
  macro average**, because it is undefined rather than zero.

`eval` recovers the base model from the run's `summary.json` when not given
one. An adapter paired with the wrong base produces plausible-looking garbage
rather than an error — on the vocabulary-extended path the base is *not*
`cfg.train.base_model`.

---

## Module reference

| module | responsibility |
|---|---|
| `schema.py` | `GeoExtraction`, `SYSTEM_PROMPT`, `build_messages` |
| `config.py` | typed config for every stage, YAML in/out |
| `paths.py` | every path, `ensure()`, `GEOSFT_HF_HOME` override |
| `data/http.py` | cached, retrying, atomic-write HTTP |
| `data/vocab.py` | Macrostrat gazetteers |
| `data/fetch.py` | Geolex corpus, concurrent detail fetch |
| `data/match.py` | longest-match n-gram gazetteer tagger |
| `data/label.py` | rules, `Labeller`, `agreement_report` |
| `data/build.py` | dedupe, filter, group-split, gold slice |
| `tokenizer/train.py` | BPE/Unigram from scratch |
| `tokenizer/analyze.py` | fertility, weighted fragmentation |
| `tokenizer/extend.py` | candidate mining, embedding surgery, config patch |
| `train/mlx_train.py` | QLoRA, `PromptCompletionDataset`, LR schedule, quantise |
| `train/hf_train.py` | PEFT/TRL path, trainable embeddings |
| `monitor/tracker.py` | metrics, alarms, live panel, plots |
| `eval/metrics.py` | `Scorer`, JSON recovery, per-field P/R/F1 |
| `eval/evaluate.py` | generation, base-vs-tuned driver |
| `cli.py` | the `geosft` command |

## Configuration

One `Config` object drives every stage and is copied into each run directory,
so a run is reproducible from a single file. `configs/default.yaml` is the full
run; `configs/smoke.yaml` is a few-minute wiring test on Qwen3-1.7B.

Knobs that matter most, in order: `mask_prompt`, `num_layers`, `lora_rank`,
`learning_rate`, then `batch_size × grad_accumulation_steps`.

## Extending this

- **A different task:** change `GeoExtraction` and the labeller. Metrics adapt
  automatically — set fields are scored generically.
- **A different corpus:** implement a fetcher producing
  `{unit_id, unit_name, passage, ...}` rows.
- **A different base model:** set `base_model` (MLX 4-bit) and `hf_base_model`.
  The traps in `extend.py` are general, but the spare-row count is not.
- **Cloud GPU:** uncomment the bitsandbytes block in `hf_train.py` for real
  4-bit QLoRA.
