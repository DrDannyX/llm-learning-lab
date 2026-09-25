# llm-learning-lab

Three hands-on labs for learning how to **adapt LLMs to a domain**, worked
end to end on a single Apple Silicon Mac, using geoscience as the domain.

The first two **change the model's weights**. Together they cover the two
training techniques that do most of the work in practice, and, just as
importantly, they show what each one *cannot* do. The third changes **what the
model is shown**: knowledge graphs, RAG and GraphRAG, compared side by side.

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

| | [geo-graphrag](geo-graphrag/) |
|---|---|
| Technique | **Retrieval**: knowledge graph vs vector RAG vs GraphRAG, side by side |
| Changes | the context the model is shown, not its weights |
| Data | 3,302 Geolex units, 8,757 passages → a 16k-node, 50k-relationship Neo4j graph |
| Stack | Neo4j (graph + vector index), Strands agents, MCP, local LLM via LM Studio |
| Signature failure | RAG: confidently counts only what it retrieved. KG: a query that silently returns nothing |
| The hard part | extraction and entity resolution (graph); recall over many passages (RAG) |
| Result | overall **0.61 / 0.86 / 0.75** (RAG / KG / GraphRAG); each wins the question types its design predicts |

## The task

The two training labs point at one problem: turning messy geological prose into a
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

The retrieval lab *does* answer questions, and it inherits that concern. Seven
of its eight benchmark categories have deterministic gold answers taken from
curated Geolex metadata ("How many Pennsylvanian units occur in Kansas?" → 41).
Only the prose-detail category needs an LLM judge, and the judge is calibrated
against the string matcher (98% agreement).

## How the labs fit together

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

### And where retrieval fits

Training puts knowledge *in* the weights, where it is compressed, hard to
update and impossible to cite. Retrieval leaves it *outside*, where it can be
queried, updated and shown as evidence. `geo-graphrag` imports geo-sft too: its
rule labeller extracts the knowledge graph's relations, so geo-sft's finding
that 83% of rule-labelled rows were wrong reappears as a graph-quality problem.
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
| 8 | — | `geocpt transfer`, compare against 0.831 |

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

### Training labs: SFT and CPT

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

### Retrieval lab: KG vs RAG vs GraphRAG

64 questions, 8 categories, `gemma-4-12b` as answerer and judge
([full report](geo-graphrag/docs/RESULTS.md)):

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| age, states, parent, relation | 0.75–0.88 | 0.88–1.00 | 0.95–1.00 |
| members (lists) | 0.58 | **1.00** | **1.00** |
| multi-hop filters | 0.07 | **1.00** | 0.04 |
| count | 0.00 | **1.00** | 0.00 |
| descriptive (prose only) | **1.00** | 0.00 | **1.00** |
| **overall** | 0.61 | **0.86** | 0.75 |

**1. Each system wins exactly where its design predicts.** The KG is the only
system that can count or filter the whole corpus, and it cannot answer anything
stated only in prose. RAG counts the 8 passages it retrieved and reports that
number confidently: "7 units" when the answer is 41.

**2. GraphRAG's gain over RAG is on entity-centred questions** (members
0.58 → 1.00), where the graph routes retrieval to the right unit. Its expansion
is local, so on counts and filters it is as blind as RAG.

**3. Context recall separates retrieval failures from generation failures.**
Did the retrieved context contain the answer at all? RAG's count failures are
retrieval failures (context recall 0.25); they cannot be fixed by a better
prompt.

The practical conclusion matches the GraphRAG literature: **the hybrid is a
routing problem**. A router sending count and filter questions to the KG and
everything else to GraphRAG would score ~1.0 on every category here.

## The honest caveats

**0.705 is the defensible SFT headline**, not 0.831. The gold review was a
*machine* review, not a geologist's, with one row flagged for expert eyes.

`states` fell hardest under review (0.946 → 0.664) because the model had
faithfully learned the labeller's blind spot — it never extracted postal
abbreviations, because it was never taught to. It scored 0.946 for reproducing
an error. Meanwhile the **base model improved** against gold (0.314 → 0.393):
it was being penalised for extracting things the rules had missed.

**Seeds:** the TAPT result is three seeds; everything else is one.

**The retrieval corpus is units A to C only.** geo-sft fetched the first 4,000
entries of Geolex's alphabetical index: 3,302 of 16,684 units, "A-L Peak" to
"Cross Creek". Famous units like the Eagle Ford are absent, and every count is a
count of A–C units. The comparison between systems is unaffected (all three see
the same slice), but no number there describes US geology as a whole.

**The KG's 0.86 is partly circular.** Seven of eight categories take gold answers
from the curated metadata the graph was built from, so the KG is reading back
its own contents. The prose-only category is the counterweight, and an
independent, hand-written test set is
[geo-graphrag experiment #10](geo-graphrag/docs/EXPERIMENTS.md). The retrieval
results are also one run with 8 questions per category: one question moves a
category by 0.125.

**Nine bugs are documented rather than hidden**, across
[SFT RESULTS §10](geo-sft/docs/RESULTS.md) and
[CPT RESULTS §6/§9](geo-cpt/docs/RESULTS.md). The common thread is worth more
than any result here: **every one produced a plausible loss curve and no error
message.** Three were caught only by running control arms, and one was masked
by a regression test that grepped source instead of executing the code — it
passed while the function was completely broken.

The retrieval lab added four more of the same kind, none of which raised an
error:
- a few-shot Cypher example pointed at a unit key that does not exist, silently
  teaching the model a query that returns nothing;
- usages joined by "in" ("Bloomfield limestone in Glenshaw Formation") were
  not split, silently turning two units into one name;
- the knowledge graph answered "I don't know" from correct rows, because the rows
  did not say what they were rows *of*;
- LM Studio's JIT loading swapped the chat and embedding models on every
  question, making the benchmark 5× slower.

The live test tier, which executes every few-shot example, caught the first.

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

### 5. Retrieval: from comparison to system

- **Fetch all of Geolex.** 16,684 units instead of the A–C slice. A config
  change in geo-sft and a longer fetch; it makes the retrieval lab's answers
  meaningful for real geology.
- **A router or an agent.** Route count and filter questions to the KG and the
  rest to GraphRAG, or give the MCP agent only the retrieval primitives and let
  it choose ([experiments #7 and #9](geo-graphrag/docs/EXPERIMENTS.md)).
- **The SFT model as the graph's extractor.** Serve the geo-sft adapter with
  `mlx_lm.server` and use it in place of the regex labeller. This closes the
  loop across all three labs, and the graph inherits whatever the fine-tune
  learned, including its errors.
- **Global GraphRAG.** Community detection plus LLM summaries, for "what are the
  main Pennsylvanian groups of the mid-continent?": the synthesis questions none
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

**Run DAPT.** It is already configured, it answers the question the CPT lab
was built for at 13× the scale, and you now have a three-seed baseline to
measure it against. If 20M tokens moves the needle further than 1.5M did, you
have located the scaling curve for yourself — which is worth more than any
single number in this repo.

If your interest is retrieval rather than training: **fetch all of Geolex and
rerun the benchmark.** It is the cheapest change that turns the retrieval lab
from a comparison on a 20% slice into a tool you could actually query.

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
USGS Geolex and Publications Warehouse (public domain), Macrostrat (CC-BY 4.0),
wikitext and ag_news (their own terms). None are redistributed here.
