# Architecture

How geo-cpt works, stage by stage. Read [LEARNING.md](LEARNING.md) first for
the concepts; this is the code tour.

---

## Data flow

```
 USGS Publications Warehouse     geo-sft's Geolex cache      geo-sft's task text
 (abstracts, public domain)      (16,684 units)              (train split only)
        │                               │                            │
        │  corpus/sources.py            │                            │
        └───────────────┬───────────────┴────────────────────────────┘
                        ▼
                  raw documents
                        │  corpus/quality.py   heuristic rejection (by reason)
                        ▼
                        │  corpus/dedup.py     MinHash + LSH near-duplicates
                        ▼
                        │  corpus/build.py     replay mixing, clean val splits
                        ▼
     data/corpus/<stage>/{train,val,val_general}.jsonl
                        │  pack.py             tokenise, concatenate, slice
                        ▼
     data/packed/<stage>/{train,val,val_general}.npy    (n_blocks, block_size)
                        │  train/cpt.py        full fine-tune, no masking
                        ▼
     runs/<name>/checkpoint-NNNNNN/     ← FULL models, ~3.4 GB each
                        │
        ┌───────────────┼────────────────────────┐
        ▼               ▼                        ▼
 eval/perplexity   eval/probe              cli.py transfer
 learn vs forget   knowledge vs style      → geo-sft SFT → macro F1
```

Every arrow is a file. Any stage re-runs independently.

## Relationship to geo-sft

geo-cpt is a separate package with its own environment, but declares geo-sft as
an editable path dependency:

```toml
[tool.uv.sources]
geo-sft = { path = "../geo-sft", editable = true }
```

Four things are reused rather than reimplemented, and each is a deliberate
choice about what is *the same problem*:

| reused | why |
|---|---|
| `geosft.data.fetch` + its HTTP cache | ~3,300 Geolex units are already on disk; only the remaining 13,000 hit the network |
| `geosft.paths.PROCESSED` task text | TAPT *is* pretraining on the SFT task's text — reading it from source keeps them provably identical |
| `geosft.monitor.tracker.RunTracker` | monitoring is backend-agnostic; live panel, alarms, JSONL metrics and loss curves all apply unchanged |
| `geosft.train.mlx_train` + `geosft.eval` | the downstream-transfer experiment must use the *same* trainer and scorer, or the comparison against 0.831 is meaningless |

The HuggingFace cache is shared too (`paths.HF_HOME` points at geo-sft's), so
multi-GB base models are not downloaded twice.

## Stage 1 — sources (`corpus/sources.py`)

Three functions, one per corpus type.

**`fetch_usgs`** — abstracts from the USGS Publications Warehouse. Real
technical geoscience prose, public domain as a US Government work, ~500 tokens
each. Paginated per query with de-duplication by record id, cached to
`data/raw/usgs.jsonl`. Abstracts arrive as HTML fragments, so `strip_html`
(selectolax) runs first.

**`fetch_geolex_full`** — the entire lexicon, calling geo-sft's own fetcher so
its disk cache applies.

**`load_task_text`** — the TAPT corpus. Two rules enforced in code, both fatal
if broken:

1. **Train split only.** Pretraining on test-set text is leakage, and it is
   invisible: downstream numbers just quietly improve.
2. **Text only, labels discarded.** Pretraining on the JSON answers would be a
   sloppy unmasked SFT wearing a CPT costume.

**`load_replay`** — general English (wikitext) for §6 of LEARNING.md. Degrades
gracefully with a warning if unavailable rather than failing the run.

## Stage 2 — quality (`corpus/quality.py`)

Heuristics in the Gopher/C4 lineage, adapted to scientific prose: `too_short`,
`too_long`, `low_alpha_ratio` (tables, coordinate dumps), `digit_heavy` (assay
appendices), `too_few_words`, `repetitive`, `no_sentences`.

The design decision that matters: **every rejection is counted by reason and
printed with an example**. That table is how we discovered the filter was
inappropriate for TAPT — 21.6% of geo-sft's passages rejected as
`no_sentences`/`too_short`, all of them valid lexicon fragments the downstream
task must handle. Hence `quality_filter: false` for pure TAPT.

## Stage 3 — deduplication (`corpus/dedup.py`)

Two passes:

1. **Exact** — hash of normalised text. Cheap, catches the obvious.
2. **Near** — MinHash signatures in an LSH index. First occurrence wins.

`shingles(text, k)` builds overlapping word k-grams (k=5 default), so
reordered text produces different shingles. `signature()` hashes them into a
128-permutation MinHash whose collision probability equals Jaccard similarity.
`MinHashLSH(threshold=0.8)` buckets signatures so `query()` returns only
plausible near-neighbours — turning O(n²) into roughly O(n).

`DedupStats` reports exact and near removals separately, because they mean
different things: lots of exact duplicates suggests a fetching bug, lots of
near duplicates is genuine corpus redundancy.

## Stage 4 — assembly (`corpus/build.py`)

Order is deliberate: **gather → filter → dedup → split → mix**.

Validation is held out **before** replay mixing, so domain perplexity is
measured on domain text alone. The replay pool is split too — the first
`val_docs` become held-out general validation, the rest go into training. If
you train on your forgetting probe it stops measuring forgetting.

`corpus_card.json` records every count, including `replay_fraction_actual` so
you can confirm the mix is what you asked for.

## Stage 5 — packing (`pack.py`)

Covered conceptually in [LEARNING.md §4](LEARNING.md#4-packing). Mechanically:
tokenise each document, append EOS, extend a running stream, and emit
`block_size` slices. The trailing partial block is discarded rather than
padded.

`no_cross_document=True` instead starts each document in a fresh block and pads
the tail — 100% retention, ~62% efficiency on this corpus.

Output is a single `(n_blocks, block_size)` int32 `.npy` plus a stats JSON.
Memory-mapped loading means the corpus never has to fit in RAM.

## Stage 6 — training (`train/cpt.py`)

Hand-written rather than delegated to mlx-lm, because the objective is the
lesson:

```python
inputs, targets = blocks[:, :-1], blocks[:, 1:]
logits = model(inputs).astype(mx.float32)
loss = cross_entropy(logits.reshape(-1, V), targets.reshape(-1)).mean()
```

No mask. Compare the SFT lab, where 72% of tokens were excluded.

Specific mechanics:

- **`build_schedule`** converts warmup from iterations to **optimizer steps**
  and prints the resolved numbers. This is the trap that cost the SFT project
  a full run; it is not repeated here.
- **Gradient accumulation** sums grads in a `tree_map`, divides, then clips
  global norm via `optim.clip_grad_norm`. Clipping matters far more in full FT
  than in LoRA: one bad batch can damage every weight.
- **Baseline before training.** Both perplexities are measured at iteration 0,
  before any weight moves. Forgetting is a delta and needs an origin.
- **Dual evaluation** at every `steps_per_eval`, with general-set drift
  coloured red when it grows.
- **`save_checkpoint`** writes a full model and `_prune_checkpoints` deletes
  all but `keep_last_n`. At ~3.4 GB per checkpoint this is not optional.
- **`grad_checkpoint`** patches `model.layers[0]`'s class via mlx-lm's helper,
  which covers every block.

Monitoring is geo-sft's `RunTracker`, unchanged: live panel, sparklines,
divergence alarms, `metrics.jsonl`, `loss_curve.png`.

## Stage 7 — evaluation (`eval/`)

**`perplexity.compare`** loads base and adapted models in turn and scores both
on *identical* packed blocks — perplexity is only comparable across models
sharing a tokenizer on the same text. Reports domain improvement and
forgetting as signed percentages.

**`probe.compare`** builds cloze probes from held-out text using geo-sft's
chronostratigraphic vocabulary as the answer set (closed, unambiguous, genuine
domain knowledge). `sequence_logprob` scores the mean log-probability of a
completion; accuracy is how often the true term beats all distractors. Chance
is 0.25 with three distractors.

**`cli.transfer`** is the payoff: quantise the CPT checkpoint to 4-bit MLX
(geo-sft's `quantize_for_mlx`), then run geo-sft's trainer and evaluator on it.
Same data, same hyperparameters, same metrics — the only variable is the
starting model.

## Module reference

| module | responsibility |
|---|---|
| `paths.py` | paths; shares geo-sft's HF cache |
| `config.py` | typed config for corpus, packing, training, eval |
| `corpus/sources.py` | USGS, full Geolex, task text, replay |
| `corpus/quality.py` | heuristic filtering with per-reason reporting |
| `corpus/dedup.py` | MinHash + LSH near-duplicate removal |
| `corpus/build.py` | assembly, replay mixing, clean splits |
| `pack.py` | tokenise and pack into fixed blocks |
| `train/cpt.py` | the CPT loop, schedule, clipping, checkpoints |
| `eval/perplexity.py` | domain vs general: learn vs forget |
| `eval/probe.py` | cloze probes: knowledge vs style |
| `cli.py` | the `geocpt` command |

## Configuration

`configs/tapt.yaml` is the cheap first run. `configs/dapt.yaml` is the big
corpus. `configs/smoke.yaml` proves the wiring in minutes.

The `stage` field (`tapt` / `dapt` / `dapt_tapt`) selects which sources are
gathered, and everything downstream keys off it — corpus, packed blocks and
evaluation all live under `<stage>/` directories so stages never collide.

## Extending this

- **Another domain:** replace `corpus/sources.py`. Everything downstream is
  domain-agnostic.
- **Another base model:** set `train.base_model` to any unquantised MLX-loadable
  repo. Memory scales ~8× parameter count for full FT.
- **Vocabulary-extended CPT** ([experiment 8](LEARNING.md#11-the-experiment-grid)):
  point `base_model` at geo-sft's `artifacts/models/*-geo-ext`. Full FT trains
  the embedding matrix, which LoRA never touched — this is where geo-sft's
  grafted tokens can finally earn their place.
