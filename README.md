# llm-learning-lab

Two hands-on labs for learning how to **modify LLMs for a domain**, worked
end to end on a single Apple Silicon Mac, using geoscience as the domain.

They are meant to be read and run in order. Together they cover the two
techniques that do most of the work in practice — and, just as importantly,
they show what each one *cannot* do.

| | [geo-sft](geo-sft/) | [geo-cpt](geo-cpt/) |
|---|---|---|
| Technique | **Supervised fine-tuning** (QLoRA) | **Continued pre-training** (DAPT/TAPT) |
| Teaches the model | *form* — a task, a schema, a convention | *content* — domain language and vocabulary |
| Data | 6,635 labelled pairs | ~10⁴ raw documents, no labels |
| Loss on | the answer only (72% of tokens masked) | every token |
| Method | LoRA, 0.365% of parameters | full fine-tune, 100% |
| Signature failure | overfitting | catastrophic forgetting |
| The hard part | label quality | corpus hygiene |
| Result | macro F1 **0.314 → 0.831** | TAPT contributes **+0.019** |

## The task

Both labs point at one problem: turning messy geological prose into a
structured database record.

> *"Austin chalk. The present generally accepted definition applies to the
> beds below Taylor marl and above Eagle Ford clay. Thickness 200 to 400
> feet."*

```json
{"unit_name": "Austin", "rank": "Formation",
 "lithologies": ["chalk", "clay", "marl"], "chronostrat": ["Late Cretaceous"],
 "thickness": {"min_m": 60.96, "max_m": 121.92},
 "relations": [{"kind": "overlies", "unit": "Eagle Ford Clay"},
               {"kind": "underlies", "unit": "Taylor Marl"}],
 "states": ["TX"]}
```

Extraction was chosen deliberately over geoscience Q&A because it is
**objectively gradable**. Field-level precision and recall against a fixed
schema tell you whether the fine-tune worked; a Q&A task would leave you
grading style with an LLM judge and guessing.

## How the two fit together

```
  pretraining          CPT              SFT            deploy
  (someone else's      geo-cpt          geo-sft
   36T tokens)         1.5M tokens      6,635 pairs
       │                  │                │
   general           domain              task
   language          language            behaviour
```

**SFT teaches form far better than facts.** CPT is the technique for the
other half. And CPT is never the deliverable on its own — a CPT'd model is
better at *continuing* geological prose and no better at answering you. It
must be followed by SFT.

`geo-cpt` imports `geo-sft` as a dependency and reuses its corpus, its
monitoring and — for the transfer experiment — its trainer and scorer, so
"did CPT help?" is measured through identical machinery.

---

## Learning path

Roughly a week of evenings, mostly unattended compute. **Do not read the docs
front to back** — most are reference, and reading about a loss curve teaches
far less than watching one.

### Stage 1 — SFT (start here)

Everything about CPT is easier to understand by contrast with SFT, so do this
first even if CPT is what you came for.

Follow the [four-session learning path](geo-sft/README.md#how-to-learn-from-this-repo)
in geo-sft. In outline:

| session | read | run |
|---|---|---|
| 1 | [LEARNING.md](geo-sft/docs/LEARNING.md) §1–5 — what SFT is, LoRA/QLoRA, the anatomy of a training example | `make smoke`, `geosft inspect` |
| 2 | §6–8 — steps vs epochs, the knobs, reading a curve | `make train` (~2 h) |
| 3 | §9 — honest evaluation | `geosft eval` |
| 4 | [EXPERIMENTS.md](geo-sft/docs/EXPERIMENTS.md) #1 | prompt-masking ablation |

The single most important screen in either project is `geosft inspect`: it
shows exactly which tokens carry loss, and asserts that the training prompt is
byte-identical to the inference prompt.

### Stage 2 — CPT

| session | read | run |
|---|---|---|
| 5 | [LEARNING.md](geo-cpt/docs/LEARNING.md) §1–5 — CPT vs SFT, packing, forgetting | `make smoke` |
| 6 | §6–9 — replay, full-FT vs LoRA, corpus hygiene, hyperparameters | `make all` (~1 h) |
| 7 | §10 — the three evaluation questions | `geocpt eval`, `geocpt probe` |
| 8 | — | `geocpt transfer`, compare against 0.831 |

### Stage 3 — your own experiment

Both labs ship an experiment grid ([SFT](geo-sft/docs/EXPERIMENTS.md),
[CPT](geo-cpt/docs/LEARNING.md#11-the-experiment-grid)). The most instructive
single run in either is **SFT #1, prompt masking**: the unmasked run reaches
*lower* training loss and *worse* task scores, which makes "loss is not your
metric" permanent.

### Reading differently

| document | how to read it |
|---|---|
| `LEARNING.md` (both) | concepts — the only part worth reading before running |
| `ARCHITECTURE.md` (both) | **lookup, not cover to cover** |
| `RESULTS.md` (both) | what was measured, and what the numbers do *not* establish |
| `EXPERIMENTS.md` | what to run next |

---

## Results

**Against hand-corrected gold labels** — the defensible numbers:

| arm | macro F1 |
|---|---|
| 4B zero-shot, no training | 0.393 |
| **4B + SFT** | **0.705** |

Against the *rule-derived* labels the same model scores 0.831. That gap is
finding 3 below, and it is the single most useful result here.

The arm comparisons were run against rule labels throughout, so they are
internally consistent but share that ~40% inflation. Read them as *relative*:

| arm | macro F1 (rule labels) |
|---|---|
| 1.7B + SFT | 0.823 |
| 4B + SFT | 0.831 |
| **1.7B + TAPT + SFT** | **0.842** |
| 1.7B + *vocab-extended* TAPT + SFT | 0.758 |

Domain perplexity under CPT: 33.27 → 13.87 (−58%).

### Three findings worth the compute

**1. TAPT helps, and it replicates.** Three seeds, both arms retrained each
time: **+0.025 mean, sd 0.005, 95% CI [+0.012, +0.039]**. The arms do not
overlap (no-CPT 0.816–0.823, TAPT 0.842–0.847). This is *better* than the CPT
lab predicted — it warned to expect nothing at 1.5M tokens.

**2. Vocabulary extension makes things worse.** −0.084, three times TAPT's
gain in the opposite direction. Fragmentation is not pure loss:
`Penn|s|ylv|anian` uses four embeddings trained on trillions of tokens; the
grafted `Pennsylvanian` is one row starting at their mean and trained on 1.2M.
The 4.89% context saving costs real quality at this scale.

**3. The rule labels were inflating everything by ~40%.** All 150 gold rows
were hand-reviewed; **125 (83%) had wrong labels**. Scored against corrected
gold, the SFT result is **0.705, not 0.831**, and the delta over base falls
from +0.516 to **+0.312**.

Also: **the task is not capacity-limited** — a 1.7B model matched a 4B one, so
the ceiling is label quality, not model size.

## The honest caveats

**0.705 is the defensible SFT headline**, not 0.831. The gold review was a
*machine* review, not a geologist's, with one row flagged for expert eyes.

`states` fell hardest under review (0.946 → 0.664) because the model had
faithfully learned the labeller's blind spot — it never extracted postal
abbreviations, because it was never taught to. It scored 0.946 for reproducing
an error. Meanwhile the **base model improved** against gold (0.314 → 0.393):
it was being penalised for extracting things the rules had missed.

**Seeds:** the TAPT result is three seeds; everything else is one.

**Nine bugs are documented rather than hidden**, across
[SFT RESULTS §10](geo-sft/docs/RESULTS.md) and
[CPT RESULTS §6/§9](geo-cpt/docs/RESULTS.md). The common thread is worth more
than any result here: **every one produced a plausible loss curve and no error
message.** Three were caught only by running control arms, and one was masked
by a regression test that grepped source instead of executing the code — it
passed while the function was completely broken.

## Where to go next

Ordered by return on effort. The first group closes out claims this repo has
already made; everything after it is new ground.

### 1. Finish what is started (hours, not days)

| | why | cost |
|---|---|---|
| **Three seeds on the vocab-extension arm** | the −0.084 result is one seed. TAPT's +0.025 only became a claim at n=3; this deserves the same. | ~3 h unattended |
| **Geologist spot-check of ~20 gold rows** | the review was a *machine* review. Twenty rows tells you how much to trust the other 130. | an hour of your time |
| **Run DAPT** | only TAPT ran (1.5M tokens). The ~20M-token DAPT corpus is already configured in `geo-cpt/configs/dapt.yaml`, and `dapt_tapt` runs both in sequence. Does 13× more domain text help further? | ~4 h |
| **Experiment #1, prompt masking** | the most instructive hour in either lab: the unmasked run reaches *lower* training loss and *worse* task scores. | ~2 h |

### 2. The next technique: preference optimisation

SFT is **imitation** — it can never exceed the quality of your labels. That is
the 0.705 ceiling. Preference methods learn from *comparisons*, which are both
cheaper to collect and able to surpass the demonstrations.

- **DPO for calibrated abstention.** Teach the model to prefer `null` over a
  plausible guess. This directly addresses the biggest production gap: the
  model currently has no way to say "I'm not sure", and a confident wrong
  formation name is worse than a blank field.
- **GRPO against your own F1 metric.** Structured extraction is unusually
  well-suited to RL with verifiable rewards, because `eval/metrics.py` already
  *is* a computable reward function — no human labelling, no reward model.

`trl` ships `DPOTrainer`, `GRPOTrainer`, `KTOTrainer`. Note mlx-lm has none of
these, so this is the HF/MPS path: slower, and no real 4-bit.

### 3. Differently-shaped skills

- **A domain embedding model.** Contrastive learning on (query, passage) pairs
  is a completely different objective from anything here, and it is the other
  half of a real system: semantic search over a report archive. Pairs
  naturally with extraction — retrieve, then extract.
- **Constrained decoding** (`outlines`, grammar-constrained generation). A
  weekend. It guarantees schema-valid JSON at *decode* time, making the
  0.833 → 1.000 schema-validity win obsolete by construction. The lesson is
  knowing when **not** to train.
- **Distillation.** Train a small model on a large one's outputs and reasoning
  traces — how you get 1.7B behaving like something far larger.
- **A quantisation study.** Everything here is 4-bit. What does 8-bit or bf16
  actually buy on this task, and at what memory cost?

### 4. Production-shaped work

- **Span-grounded extraction.** The model outputs `"lithologies": ["chalk"]`
  but not *where it saw it*. For any geoscience QA workflow a reviewer needs
  the supporting span. This means a schema change (character offsets) and
  relabelling — the largest single gap between this and something deployable.
- **Calibration.** Logprob-based confidence to route uncertain records to a
  human review queue.
- **The data flywheel.** Deploy → geologist corrects → corrections become new
  SFT data. Unglamorous, and almost always the highest-return move: it is the
  only thing that escapes the rule-label ceiling with certainty.
- **Serving.** `mlx_lm.server --adapter-path` runs on a Mac; `mlx_lm.fuse`
  merges the adapter into one artifact. For Linux/GPU (vLLM) you would retrain
  on the HF path — MLX adapters do not transfer.

### 5. Staying in geoscience

- **A harder, messier corpus.** Open-file exploration reports (WAMEX, state
  surveys) and well completion reports are PDFs, not clean API text. Ingest
  becomes the work, which is realistic.
- **Multimodal.** Core photographs, well logs, scanned maps. A large jump —
  vision encoders — but it is where the domain actually lives.
- **A different task on the same corpus.** Lithology-description
  classification, or summarisation, to see how much of the pipeline transfers
  when only the schema changes.

### If you only do one thing

**Run DAPT.** It is already configured, it answers the question the CPT lab
was built for at 13× the scale, and you now have a three-seed baseline to
measure it against. If 20M tokens moves the needle further than 1.5M did, you
have located the scaling curve for yourself — which is worth more than any
single number in this repo.

## Add to the repo
I would love if you want to contribute and add in a lesson of your own on something related to geoscience and AI. To do this follow these rough steps:
1. Create a new branch
2. Add a new folder
3. In your folder add a README.md explaining what you have done
4. Commit and raise a PR

## Requirements

Apple Silicon with 32 GB+ (48 GB for CPT full fine-tuning), Python 3.12,
[uv](https://docs.astral.sh/uv/). Each project owns its virtualenv:

```bash
cd geo-sft && make setup && make doctor && make smoke
cd ../geo-cpt && make setup && make doctor && make smoke
```

Budget ~40 GB of disk for model weights; `make clean-models` / `make clean-ckpt`
reclaim it.

## Licence

[MIT](LICENSE) © 2026 Daniel Bongiorno. Data sources carry their own licences —
USGS Geolex and Publications Warehouse (public domain), Macrostrat (CC-BY 4.0),
wikitext and ag_news (their own terms). None are redistributed here.
