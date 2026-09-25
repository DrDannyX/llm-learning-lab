# Architecture

How geo-sft works, stage by stage. Read [README.md](../README.md) first for the
what and why; this document is the how.

---

## Data flow

```
 Geoscience Australia (ASUD)     Macrostrat API
 WFS unit index + weekly         (lithologies, intervals, minerals)
 state-report ZIPs, 18,387 units        │
        │  data/asud.py, data/fetch.py   │  data/vocab.py (+ ASUD unit names)
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
    states:      list[str]           # ASUD codes (NSW, QLD, ...), set-scored
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

## Stage 1 — corpus (`data/asud.py`, `data/fetch.py`, `data/vocab.py`)

**Source: Geoscience Australia's Australian Stratigraphic Units Database
(ASUD)**, the national authority on Australian stratigraphic names, released
under CC BY 4.0 (attribution required). It is reached two ways, and the lab
needs both:

| | what | why |
|---|---|---|
| **WFS** `services.ga.gov.au/gis/stratunits` | the documented OGC service: 18,354 units with rank, lithology class, state, hierarchy | the unit index. But `DESCRIPTION` is **truncated at 255 characters** and ages are era-level only |
| **state reports** (ZIPs linked from asud.ga.gov.au) | six pipe-delimited tables per jurisdiction, rebuilt weekly | the prose (definition cards, 350k per-reference notes), **curated relations**, thickness in metres, fine-grained ages |

`asud.py` pages the WFS (always with `count` and `sortBy=STRATNO` — the
server default page is a million rows, and GA warns paging is not
transaction-safe), checks the attribute names against `DescribeFeatureType` at
run time (GA has renamed them before), and snapshots the report ZIPs into
`data/raw/asud/` with a manifest of SHA-256 hashes and server dates. The
reports change weekly, so re-downloading is not reproducing: every downstream
step reads the snapshot, and only `--force` refetches.

The prose is two kinds of passage, each rendered as a lexicon *entry* —
`"<Unit Name>. <text>"` — because a note like *"Conformably overlain by
Cygnet Coal Measures."* has no subject without its heading:

> *Wilton Formation. Overlies Woonona Coal. Overlain by Appin Formation.
> Thickness: ~250 m. Of Illawarra Coal Measures. Possibly Stratigraphically
> equivalent to part of Four Mile Creek Subgroup. Geological Province: Sydney
> Basin.*

- **definition cards** (2,307 passages): the author's sections — lithology,
  relationships and boundaries, thickness, extent, age reasons, name source —
  with administrative sections (proposer, reservation) dropped.
- **reference notes** (20,004 passages): one published reference's comment on
  the unit, with page-location boilerplate (*"Location in text includes p325
  Fig.3"*) stripped. A note filed under **more than one unit** is dropped
  entirely: it is about none of them, and it would put identical text on both
  sides of the group split.

At most 12 passages per unit, so a few heavily-cited units cannot dominate the
loss. Result: **22,311 passages from 8,094 units**.

This is deliberately *not* clean text — telegraphic, abbreviated
(`qtz-rich`, `Sltst`, `Gp`), full of house style. A general instruct model
handles it badly, which is exactly what leaves room for a fine-tune to show a
real gain.

**Source: Macrostrat** (CC-BY 4.0) supplies three closed vocabularies: 214
lithologies, 514 chronostratigraphic intervals and 6,350 minerals. The unit
names the relation rules validate against come from ASUD itself.

**Australian spellings.** ASUD writes *Palaeozoic*, *Archaean* and series
names (*Lower Devonian*) where Macrostrat has *Paleozoic*, *Archean* and the
epoch (*Early Devonian*). `vocab.chrono_aliases` maps every such surface form
to one canonical name, so the label space stays closed. Precambrian and the
ICS's unnamed Cambrian series and stages, which Macrostrat lacks, are added.

**Gotcha handled:** Macrostrat carries planetary timescales. `Amazonian`,
`Noachian` and `Hesperian` are *Martian* periods and would be nonsense labels
on an Australian corpus. They are filtered by inspecting the nested
`timescales[].name` — note that the naive substring check for `"mars"` fails,
because the string is `"Martian"`.

**Gotcha handled:** a newline inside a report's free-text field splits one
record over two lines. When it falls in the *last* field, the fragment has
too few fields to be a record — and gluing it to the *next* line instead
silently corrupts that record's STRATNO. `parse_table` attaches a short line
to the previous record. Guarded by
`test_report_table_stitches_split_records_and_folds_pipes`.

## Stage 2 — training pairs (`data/label.py`, `data/match.py`, `data/build.py`)

### The labelling rule

> **A target field may only contain facts present in the passage itself.**

ASUD offers tempting curated metadata (authoritative ages, states, thickness,
even the unit's stratigraphic relations), but training on facts the model
cannot see teaches it to state things confidently without evidence. The
metadata is therefore used *only* to audit the labeller, never to write labels.

`agreement_report()` exploits that separation: because the labeller never reads
the curated fields, they act as independent ground truth. ASUD curates far more
than Geolex did, so almost every field has a tripwire. On the full corpus:

| check | value | reading |
|---|---|---|
| unit_name retained | 99.9% | every passage leads with its unit's name |
| rank agreement | 98.3% | name-derived rank matches ASUD's |
| thickness in curated range | 89.2% | extracted thickness within ASUD's min–max |
| relations in curated list | 59.7% | a **precision floor**: ASUD's list is incomplete |
| chronostrat consistency | 51.9% | token overlap only; `Sakmarian` vs curated `Cisuralian` counts as a miss |
| states precision | 0.841 | named states are real |
| states recall | 0.041 | **expected to be low — see below** |

The states recall is low *by design*, not by defect. We extract states
*named in the passage*; ASUD records every jurisdiction the unit occurs in,
and a reference note rarely names one. Different questions. These numbers are
drift tripwires, not accuracy scores — if one moves sharply after you edit the
rules, something broke.

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
then **validates the head word against ASUD's 18k unit names** so we do not
capture ordinary capitalised English. ASUD's own templates are covered too:
*"Overlying unit: X"*, *"Overlies: X"*, *"Conformable on X"*, and the
shorthand *"Over X; under Y"*.

> **Bug found here (direction).** *"the overlying Bortala Formation"* puts the
> named unit above the subject, but *"an interval overlying Oolloo Dolostone"*
> puts the *subject* above it. The first version of the rule treated both the
> same and reversed every participle. The article is what disambiguates them.
> Found in the gold review; guarded by `test_relation_templates`.

> **Bug found here (self-relations).** ASUD names carry their rank, so the
> Geolex-era check "is the head word the unit's own name?" no longer fired —
> *"Breakfast Sandstone overlies Breakfast Sandstone"*. Comparing *cores*
> fixed that and broke something else: *Murchison Granite* and *Murchison
> Volcanics* share a core, and the real intrusive relation between them was
> discarded as a self-reference. Only the full name or the bare core counts as
> self.

> **Bug found here.** These patterns are compiled *without* `re.IGNORECASE`.
> The name group relies on `[A-Z]` to find proper nouns, and a global
> ignorecase flag silently turns that into "any letter" — so the group
> swallowed whole clauses, yielding `"Church member of Howard limestone"`
> instead of `"Church Member"`. Trigger words that genuinely need
> case-insensitivity use scoped `(?i:...)` groups instead. Guarded by
> `test_relation_names_are_not_greedy`.

`infer_rank` reads the unit name's own last word, then the words following it,
with explicit ranks (`Member`, `Suite`) beating the lithology-as-rank fallback
(`Tumblagooda Sandstone` → Formation). The rock words that stand in for a rank
were *measured* on ASUD: units named "X Granite", "X Volcanics" or "X Beds"
are Formation rank 95–100% of the time, while "Complex" and "Sequence" split
evenly between Group and Formation and are left as Unknown.

> **Bug found here.** `rstrip("s")` de-pluralised "Beds" into "bed" — a Bed —
> when ASUD's "X Beds" are Formation rank, and turned "Volcanics" into
> "volcanic", which matched nothing. Rank fill was 1.1% until the exact word
> was tried before the de-pluralised one.

**Unit names are masked before tagging rock, mineral, age and state terms.**
*Tumblagooda Sandstone* names a unit; it does not report sandstone. The
Geolex-era gold review found rock words inside proper names to be the single
most common rule error, and every ASUD passage leads with one. Both current
ASUD names and anything that *looks* like a name (capitalised words ending in a
capitalised rank or rock word, or a house abbreviation — *Carcoar Granite*,
*Nirranda Gp*) are blanked first. Relations still read the unmasked text.

### Splitting

**Group-wise on `unit_id`.** ASUD carries many reference notes per unit and
they overlap heavily — a dozen notes all describing the Mathinna Supergroup.
A random split puts near-duplicates on both sides of the wall and inflates the
test score badly. Every passage about a unit lands in exactly one split.

**A reviewed gold set is frozen.** Every labeller fix changes which rows pass
the filter, which reshuffles the unit split, which draws a different gold
slice — silently discarding hours of review. Once `gold.jsonl` has reviewed
rows, `build` keeps it verbatim and forces its units into test.

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

> The measurements in this section (`Pennsylvanian`, 2.54% → 4.89%) are from
> the Geolex-era version of this lab; the extension experiment has not been
> re-run on ASUD. The traps are properties of the model and libraries, not
> the data, and all still apply.

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
| `data/asud.py` | ASUD access: WFS unit index, report snapshot, table parsing |
| `data/fetch.py` | ASUD passages (definition cards, reference notes) |
| `data/vocab.py` | Macrostrat gazetteers, Australian spellings, ASUD names |
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
