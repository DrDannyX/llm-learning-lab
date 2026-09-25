# llm-learning-lab

Three hands-on labs for learning how to **adapt LLMs to a domain**, worked
end to end on a single Apple Silicon Mac, using geoscience as the domain —
specifically **Geoscience Australia's open stratigraphic data**: the
Australian Stratigraphic Units Database (ASUD) and GA's eCat publication
catalogue, both CC BY 4.0.

The first two **change the model's weights**. Together they cover the two
training techniques that do most of the work in practice, and, just as
importantly, they show what each one *cannot* do. The third changes **what the
model is shown**: knowledge graphs, RAG and GraphRAG, compared side by side.

| | [geo-sft](geo-sft/) | [geo-cpt](geo-cpt/) |
|---|---|---|
| Technique | **Supervised fine-tuning** (QLoRA) | **Continued pre-training** (DAPT/TAPT) |
| Teaches the model | *form* — a task, a schema, a convention | *content* — domain language and vocabulary |
| Data | 14,691 labelled pairs | 3.3M tokens (TAPT); 7.4M built for DAPT |
| Loss on | the answer only (72% of tokens masked) | every token |
| Method | LoRA, 0.365% of parameters | full fine-tune, 100% |
| Signature failure | overfitting | catastrophic forgetting |
| The hard part | label quality | corpus hygiene |
| Result | macro F1 **0.320 → 0.807** (reviewed gold) | TAPT contributes **+0.023** |

| | [geo-graphrag](geo-graphrag/) |
|---|---|
| Technique | **Retrieval**: knowledge graph vs vector RAG vs GraphRAG, side by side |
| Changes | the context the model is shown, not its weights |
| Data | all 18,387 ASUD units, 22,311 passages → a 42k-node, 180k-relationship Neo4j graph |
| Stack | Neo4j (graph + vector index), Strands agents, MCP, local LLM via LM Studio |
| Signature failure | RAG: cannot see facts that live in tables, not prose. KG: a query that silently returns nothing |
| The hard part | extraction and entity resolution (graph); recall over many passages (RAG) |
| Result | overall **0.29 / 0.88 / 0.74** (RAG / KG / GraphRAG); each wins the question types its design predicts |

## The task

The two training labs point at one problem: turning messy geological prose into a
structured database record.

> *"Culvida Sandstone. Up to 210m thick; fine to coarse sandstone; siltstone;
> granule and pebble conglomerate (poorly sorted); fluvial. Conformably
> overlies Erskine Sandstone. Type section 1km south of Culvida Soak."*

```json
{"unit_name": "Culvida Sandstone", "rank": "Formation",
 "lithologies": ["conglomerate", "sandstone", "siltstone"], "chronostrat": [],
 "thickness": {"min_m": 210.0, "max_m": 210.0},
 "relations": [{"kind": "overlies", "unit": "Erskine Sandstone"}],
 "states": []}
```

Extraction was chosen deliberately over geoscience Q&A because it is
**objectively gradable**. Field-level precision and recall against a fixed
schema tell you whether the fine-tune worked; a Q&A task would leave you
grading style with an LLM judge and guessing.

The retrieval lab *does* answer questions, and it inherits that concern. Seven
of its eight benchmark categories have deterministic gold answers taken from
curated ASUD metadata ("How many Ordovician units occur in Tasmania?" → 58).
Only the prose-detail category needs an LLM judge, and the judge is calibrated
against the string matcher (97% agreement).

## How the labs fit together

```
  pretraining          CPT              SFT            deploy
  (someone else's      geo-cpt          geo-sft
   36T tokens)         3.3M tokens      14,691 pairs
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

### And where retrieval fits

Training puts knowledge *in* the weights, where it is compressed, hard to
update and impossible to cite. Retrieval leaves it *outside*, where it can be
queried, updated and shown as evidence. `geo-graphrag` imports geo-sft too: its
rule labeller extracts the knowledge graph's text relations, so geo-sft's
finding that 57% of rule-labelled rows had an error reappears as a
graph-quality problem — and because ASUD also *curates* relations, the graph
can measure it: 62% of the rules' OVERLIES edges are confirmed by curation.
And the SFT model is a natural next extractor for that graph
([experiment #1](geo-graphrag/docs/EXPERIMENTS.md)).

```
                    ┌──────────────── Neo4j ────────────────┐
  question ─► RAG ──┤ vector index ─► 8 similar passages    ├─┐
           ─► KG ───┤ LLM-written Cypher ─► rows            ├─┼─► same model, same prompt ─► 3 answers
           ─► GraphRAG  passages + graph facts around units ├─┘
                    └───────────────────────────────────────┘
```

All three systems send their context to **the same model with the same
prompt**, so the only thing that differs between the answers is what was
retrieved. It is the same controlled-comparison discipline as geo-cpt reusing
geo-sft's scorer.

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
| 8 | — | `geocpt transfer`, compare against the no-CPT arm (0.778) |

### Stage 3 — knowledge graphs, RAG and GraphRAG

Independent of stages 1–2 except for the corpus: needs geo-sft's `make fetch`,
Docker and LM Studio. No training: ingest takes minutes and the benchmark about
40 minutes. Follow the
[six-session path](geo-graphrag/README.md#how-to-learn-from-this-repo). In outline:

| session | read | run |
|---|---|---|
| 9 | [LEARNING.md](geo-graphrag/docs/LEARNING.md) §1–3: what a KG is, extraction, entity resolution | `make ingest`, the Neo4j browser |
| 10 | §4–6: Cypher and text-to-Cypher, embeddings, GraphRAG | `georag inspect`, `georag ask` |
| 11 | §7–9: measuring retrieval, context recall, LLM judges | `make bench` |
| 12 | §10: MCP and agents | `make agent` |

Four ways to ask a question, from quickest to most flexible:

| | command | |
|---|---|---|
| web UI | `make web` → http://localhost:8000 | three answers side by side, per-system evidence, 33 example questions grouped by what they reveal, and an *About the data* panel |
| CLI | `make ask Q="..."` | the same, in the terminal |
| inspection | `make inspect Q="..."` | exactly what each system retrieved, before any answer is written |
| agent | `make agent` | a Strands agent that calls the three systems through an MCP server and explains which answer is best supported |

The equivalent of `geosft inspect` here is `georag inspect` (or the web UI's
*Evidence* panels): most wrong answers are decided at retrieval, before any
text is generated.

### Stage 4 — your own experiment

Every lab ships an experiment grid ([SFT](geo-sft/docs/EXPERIMENTS.md),
[CPT](geo-cpt/docs/LEARNING.md#11-the-experiment-grid),
[retrieval](geo-graphrag/docs/EXPERIMENTS.md)). The most instructive single
training run is **SFT #1, prompt masking**: the unmasked run reaches *lower*
training loss and *worse* task scores, which makes "loss is not your metric"
permanent. The most instructive retrieval experiment is **geo-graphrag #1, LLM
extraction**: rebuild the graph with an LLM instead of regex rules, and watch
both the edge count and the error rate go up.

### Reading differently

| document | how to read it |
|---|---|
| `LEARNING.md` (all) | concepts — the only part worth reading before running |
| `ARCHITECTURE.md` (all) | **lookup, not cover to cover** |
| `RESULTS.md` (all) | what was measured, and what the numbers do *not* establish |
| `EXPERIMENTS.md` | what to run next |

---

## Results

All three labs were first built on the USGS Geolex lexicon and then migrated
to Geoscience Australia's data; the Geolex runs and results are kept in each
lab's `runs/geolex-archive/`. Where both exist, both are shown — same code,
same hyperparameters, different corpus.

### Training labs: SFT and CPT

**SFT** (Qwen3-4B, QLoRA, 900 iterations):

| scored against | base | + SFT | Geolex version |
|---|---|---|---|
| rule labels (200 test rows) | 0.311 | 0.806 | 0.314 → 0.831 |
| **reviewed gold (150 rows)** | **0.320** | **0.807** | 0.393 → 0.705 |

**CPT** (Qwen3-1.7B, full fine-tune, TAPT then SFT, rule-label test set):

| arm | macro F1 |
|---|---|
| 1.7B + SFT | 0.778 |
| **1.7B + TAPT + SFT** | **0.801** |
| 4B + SFT | 0.806 |

Domain perplexity under TAPT: 29.24 → 13.27 (−55%).

### Three findings worth the compute

**1. TAPT helps, and it replicates across corpora.** +0.023 on ASUD; +0.019
at one seed and **+0.025 mean over three seeds** (95% CI [+0.012, +0.039]) on
Geolex. The gain lands on the vocabulary-heavy fields (chronostrat +0.052,
minerals +0.035). And 1.7B + TAPT nearly matches 4B: the task is not
capacity-limited.

**2. Better rules shrank the rule-label inflation — and a review that fixes
the rules contaminates the gold.** The Geolex rules were wrong on 83% of
reviewed rows and inflated the SFT score by ~40% (0.831 vs 0.705 on gold). The
ASUD rules were wrong on 57%, and rule and gold scores now agree (0.806 vs
0.807). But the gold review found five mechanical rule bugs, which were fixed
*before the final training labels were built*: the labels were improved using
what the test rows revealed. The gold score is therefore optimistic, and the
tuned model scores slightly *below* the fixed rules on every field — it
imitates its teacher imperfectly and shows no sign of exceeding it. A second
gold slice reviewed after the rules were frozen is the clean measurement
([geo-sft RESULTS §8](geo-sft/docs/RESULTS.md)).

**3. ASUD stores its structure in tables, and only the graph can read them.**

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| age, states | 0.00 | **1.00** | **1.00** |
| parent, members | 0.50–0.60 | **1.00** | **1.00** |
| relation | 0.38 | 0.88 | **1.00** |
| multi-hop filters | 0.00 | **0.88** | 0.06 |
| count | 0.00 | **1.00** | 0.00 |
| descriptive (prose only) | **0.88** | 0.25 | **0.88** |
| **overall** | 0.29 | **0.88** | 0.74 |
| *Geolex version* | *0.61* | *0.86* | *0.75* |

The graph systems replicate; RAG halves. Geolex summaries said *"Age is Late
Cretaceous ... in Texas"*; ASUD's reference notes give ages as "51.5–36.5 Ma"
and places as geological provinces, and keep the curated age and state in a
table. RAG's context recall on age is 0.12: the answer is not in the text. The
lesson of the lab — *where the facts live decides which system can answer* —
is sharper on the new corpus than on the one it was built for. A router
sending counts, filters and table facts to the KG and prose questions to
GraphRAG would score near 1.0 everywhere.

## The honest caveats

**The SFT gold score (0.807) is optimistic** — see finding 2. The review was a
*machine* review (claude-opus-5.5), not a geologist's.

**Seeds:** the ASUD TAPT result is one seed; the Geolex three-seed result is
what makes the effect credible. Everything else is one seed.

**The KG's 0.88 is partly circular.** Seven of eight benchmark categories take
gold answers from the curated metadata the graph was built from.
`descriptive` is the counterweight, and an independent, hand-written test set
is [geo-graphrag experiment #10](geo-graphrag/docs/EXPERIMENTS.md). Some
benchmark subjects are informal ASUD entries ("Balbirini Dolostone, 'lower'")
that the question filter should exclude. One run, 8 questions per category.

**The ASUD snapshot is dated.** GA rebuilds the state reports weekly; the
numbers here come from the 20 Sep 2026 snapshot recorded in
`geo-sft/data/raw/asud/manifest.json`.

**Bugs are documented rather than hidden**, in each lab's RESULTS and the
Geolex archives. Migrating to ASUD found seven more (a reversed participle
direction, a self-relation check broken by rank-bearing names, a record
stitcher that corrupted STRATNOs, `rstrip("s")` turning "Beds" into Beds, ...).
The common thread is unchanged: **every one produced a plausible dataset or
loss curve and no error message.**

## Where to go next

Ordered by return on effort. The first group closes out claims this repo has
already made; everything after it is new ground.

### 1. Finish what is started (hours, not days)

| | why | cost |
|---|---|---|
| **A second, clean gold slice** | the current gold informed the labeller fixes (finding 2). 150 fresh rows reviewed against the frozen rules measure the leakage. | an afternoon |
| **Three seeds on the ASUD TAPT arms** | +0.023 is one seed; the Geolex effect only became a claim at n=3. | ~7 h unattended |
| **Geologist spot-check of ~20 gold rows** | the review was a *machine* review. Twenty rows tells you how much to trust the other 130. | an hour of your time |
| **Run DAPT** | only TAPT ran (3.3M tokens). The 7.4M-token DAPT corpus (GA eCat abstracts + all of ASUD, held-out units removed) is built; `dapt_tapt` runs both in sequence. | ~4 h |
| **Experiment #1, prompt masking** | the most instructive hour in either lab: the unmasked run reaches *lower* training loss and *worse* task scores. | ~2 h |

### 2. The next technique: preference optimisation

SFT is **imitation** — it can never exceed the quality of your labels. That is
the ceiling this repo keeps hitting (the tuned model never beats its rules). Preference methods learn from *comparisons*, which are both
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

### 5. Retrieval: from comparison to system

- **Fix the benchmark's subject filter and re-run.** Exclude informal ASUD
  entries (quotes, commas, numbered units) from question subjects, and score
  ages in Ma as well as by name.
- **A router or an agent.** Route count and filter questions to the KG and the
  rest to GraphRAG, or give the MCP agent only the retrieval primitives and let
  it choose ([experiments #7 and #9](geo-graphrag/docs/EXPERIMENTS.md)).
- **The SFT model as the graph's extractor.** Serve the geo-sft adapter with
  `mlx_lm.server` and use it in place of the regex labeller. This closes the
  loop across all three labs, and the graph inherits whatever the fine-tune
  learned, including its errors.
- **Global GraphRAG.** Community detection plus LLM summaries, for "what are the
  main Permian groups of the Sydney Basin?": the synthesis questions none
  of the current three systems handles well.

### 6. Staying in geoscience

- **A harder, messier corpus.** Open-file exploration reports (WAMEX, state
  surveys) and well completion reports are PDFs, not clean API text. Ingest
  becomes the work, which is realistic.
- **Multimodal.** Core photographs, well logs, scanned maps. A large jump —
  vision encoders — but it is where the domain actually lives.
- **A different task on the same corpus.** Lithology-description
  classification, or summarisation, to see how much of the pipeline transfers
  when only the schema changes.

### If you only do one thing

**Review a second gold slice.** It is the only way to know how much of the
0.807 is real, and it is the lesson this repo keeps teaching one level
deeper each time: whoever builds the test set decides what "better" means —
including you, after you have looked at it.

If your interest is training scale rather than evaluation: **run DAPT.** The
corpus is built, and TAPT's replicated +0.023 is the baseline to beat.

If your interest is retrieval: **build the router** (geo-graphrag experiment
#9). The benchmark says exactly which questions to send where.

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
cd ../geo-graphrag && make setup && make up && make doctor && make ingest && make web
```

Budget ~40 GB of disk for model weights; `make clean-models` / `make clean-ckpt`
reclaim it.

geo-graphrag also needs **Docker** (for Neo4j, ~1 GB) and **LM Studio** serving
a chat model and an embedding model on `localhost:1234`. The defaults are
`google/gemma-4-12b` (~7.5 GB) and `text-embedding-nomic-embed-text-v1.5`. Load
both explicitly (`lms load ...`); with JIT loading they evict each other on
every question. The Neo4j browser is at http://localhost:7474 (`neo4j` /
`geograph-lab`).

## Licence

[MIT](LICENSE) © 2026 Daniel Bongiorno. Data sources carry their own licences —
Geoscience Australia's ASUD and eCat (CC BY 4.0, © Commonwealth of Australia
(Geoscience Australia)), Macrostrat (CC-BY 4.0), wikitext and ag_news (their
own terms). The reviewed gold set and benchmark questions committed here are
derived from ASUD and carry its attribution. The archived Geolex-era material
derives from USGS Geolex (public domain).
