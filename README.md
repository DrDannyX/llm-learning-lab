# llm-scratchpad

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

| arm | macro F1 |
|---|---|
| 4B zero-shot, no training | 0.314 |
| 1.7B + SFT | 0.823 |
| 4B + SFT | **0.831** |
| 1.7B + TAPT + SFT | **0.842** |

Domain perplexity under CPT: 33.27 → 13.87 (−58%).

Two things worth noticing. **The task is not capacity-limited** — a 1.7B model
matched a 4B one, so the ceiling is label quality, not model size. And **TAPT's
gains land on the hardest fields** (`minerals` +0.037, `relations` +0.030)
while near-saturated ones barely move.

## The honest caveats

Both labs state these in their own docs; they belong up front too.

**Every score is against rule-derived labels.** A model trained on regex output
learns to imitate regexes, so scoring it against those same rules partly
measures itself. A 150-row gold set is included for hand-correction and is
still unreviewed. Read the gold numbers, not these.

**Single seed, 200 examples.** The +0.019 from TAPT is suggestive, not
established.

**Seven bugs are documented rather than hidden**, across
[SFT RESULTS §10](geo-sft/docs/RESULTS.md) and
[CPT RESULTS §6/§9](geo-cpt/docs/RESULTS.md). The common thread is worth more
than any result here: **every one produced a plausible loss curve and no error
message.** Three were caught only by running control arms.

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
