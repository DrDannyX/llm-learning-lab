# Learning CPT with this repo

A guide to **continued pre-training** as a subject, using this project as the
worked example. It assumes you have been through
[geo-sft](../../geo-sft/docs/LEARNING.md) — CPT is best understood by contrast
with SFT, and this document leans on that contrast throughout.

**Contents**
1. [What CPT is, and where it sits](#1-what-cpt-is-and-where-it-sits)
2. [DAPT, TAPT and the terminology](#2-dapt-tapt-and-the-terminology)
3. [The objective: what changes from SFT](#3-the-objective-what-changes-from-sft)
4. [Packing, and why SFT-style batching would waste your compute](#4-packing)
5. [Catastrophic forgetting: the signature failure](#5-catastrophic-forgetting)
6. [Replay: the cheapest defence](#6-replay)
7. [Why full fine-tuning, not LoRA](#7-why-full-fine-tuning-not-lora)
8. [Corpus hygiene is the actual work](#8-corpus-hygiene-is-the-actual-work)
9. [Hyperparameters, and why they differ from SFT](#9-hyperparameters)
10. [Evaluating CPT: three questions](#10-evaluating-cpt-three-questions)
11. [The experiment grid](#11-the-experiment-grid)
12. [A guided first session](#12-a-guided-first-session)
13. [Failure modes](#13-failure-modes)
14. [Glossary](#14-glossary)

---

## 1. What CPT is, and where it sits

Continued pre-training is: **keep running the original pretraining objective,
on new text.** No labels, no new loss function, no change of task. You take a
model that has already been pretrained and you give it more of the same
treatment on different data.

That is the entire definition, and it is defined as much by what it does *not*
change as by what it does.

| | pretraining | **CPT** | SFT | DPO/RLHF |
|---|---|---|---|---|
| Data | web-scale raw text | **domain raw text** | labelled pairs | preference pairs |
| Labels | none | **none** | yes | ranked pairs |
| Objective | next token | **next token** | next token on answer only | preference |
| Teaches | language, world | **domain knowledge + vocabulary** | form, conventions | taste |
| Scale | 10¹²–10¹⁴ tokens | **10⁷–10¹⁰ tokens** | 10³–10⁶ examples | 10³–10⁵ pairs |

The geo-sft lab established that **SFT teaches form far better than facts**.
CPT is the technique for the other half. If your model needs to *know* more
geoscience rather than *format* its answers better, this is the tool.

> **Honest expectation for this project.** Our corpus is roughly 1.5M tokens
> for TAPT and ~15–20M for DAPT. Real domain adaptation uses billions. You
> should expect modest, measurable perplexity movement and quite possibly no
> downstream gain at all. That is not a failed experiment — knowing the scale
> at which a technique starts to pay is exactly what you are here to learn.

## 2. DAPT, TAPT and the terminology

These are not three techniques. They are one technique pointed at different
text.

- **CPT** — the general term. Continue pretraining on new data, whatever it is.
- **DAPT** (*domain-adaptive pretraining*) — CPT where the new data is a large
  corpus of unlabelled **domain** text. Geoscience broadly.
- **TAPT** (*task-adaptive pretraining*) — CPT on the unlabelled text **of your
  task dataset**. Much smaller, but exactly on-distribution.

DAPT ⊂ CPT. Both terms come from Gururangan et al., *"Don't Stop Pretraining"*
(ACL 2020), which showed they are **complementary**: DAPT → TAPT → fine-tune
beat any single stage.

There are two orderings and confusing them wastes weeks:

| | sequence | rationale |
|---|---|---|
| **Learning order** | TAPT first | an evening, on data you already have. If it moves nothing, the big corpus probably will not either. |
| **Pipeline order** | DAPT → TAPT → SFT | broad adaptation, then refocus on the task distribution, then teach the output format. |

In this repo it is one `stage` setting: `tapt`, `dapt`, or `dapt_tapt`. Same
trainer, same code, different corpus.

**One trap specific to TAPT.** Because the task text *is* your target
distribution, you should **not** quality-filter it. Our filter rejects 21.6% of
geo-sft's passages as `no_sentences` or `too_short` — and those are perfectly
valid lexicon fragments the downstream task must handle. Filtering them moves
pretraining *away* from the distribution you will be graded on, which is the
opposite of TAPT's purpose. Hence `quality_filter: false` in `configs/tapt.yaml`.

## 3. The objective: what changes from SFT

Here is the entire CPT loss, from `train/cpt.py`:

```python
inputs, targets = blocks[:, :-1], blocks[:, 1:]
logits = model(inputs)
loss = cross_entropy(logits, targets).mean()
```

Compare with the SFT lab, where the prompt was masked out and **72% of tokens
carried no loss**. Here there is no mask. Every token is a prediction target.

That difference is the whole distinction between pretraining and instruction
tuning, and it has consequences:

- **More signal per token.** Every token teaches something, so CPT extracts
  more gradient per unit of text than masked SFT.
- **No notion of "the answer".** The model is not learning to respond, it is
  learning the *distribution* of geological language.
- **You cannot overfit to a format**, because there is no format. You can and
  will overfit to the *corpus*, which is why deduplication matters so much.

This is also why CPT **must be followed by SFT** if you want an instruction
model. CPT alone will make the model better at continuing geological prose and
no better at answering your questions — it may actually be worse, because you
have shifted it away from its instruction tuning.

## 4. Packing

In SFT, each example was one prompt+answer pair, padded to the batch's longest
item. Reasonable: examples are similar lengths and each carries a label.

CPT documents vary enormously — a 40-token Geolex fragment beside a 3,000-token
USGS abstract. Padding each to `block_size` would waste **96% of the compute**
on the short one.

So CPT **packs**: concatenate everything into one token stream, slice into
equal blocks.

```
doc1 <eos> doc2 <eos> doc3 <eos> doc4 ...
|<-- block 0 -->|<-- block 1 -->|<-- block 2 -->|
```

Measured on our corpus, `geocpt pack` reports:

| mode | blocks | retention | efficiency |
|---|---|---|---|
| packed | fewer | 98.4% | **100%** |
| `no_cross_document` | more | 100% | **61.8%** |

Two different questions, and conflating them hides the tradeoff:

- **retention** — of the real tokens we had, how many made it into blocks?
  (Packing discards a partial tail; `no_cross_document` discards none.)
- **efficiency** — of the tokens we will actually *compute on*, how many are
  real rather than padding?

**The cost of packing: cross-document contamination.** A block can straddle a
boundary, so the start of doc3 attends to the tail of doc2 — unrelated text.
Essentially all pretraining tolerates this, because the waste from avoiding it
is worse and the model learns that `<eos>` means "what came before is
irrelevant". Serious implementations use block-diagonal attention masks; MLX's
trainer does not expose them, so this repo offers the blunt alternative and
makes you look at what it costs.

## 5. Catastrophic forgetting

**This is the signature failure of CPT, exactly as overfitting is the signature
failure of SFT.**

When you train on geoscience text and nothing else, the model's weights drift
toward the geoscience distribution. It gets better at geology and *worse at
everything else* — grammar, reasoning, other domains, and often the instruction
following that was carefully tuned into it.

The insidious part: **you cannot see it with one metric.** Domain perplexity
falls beautifully. The run looks like a success. Meanwhile the model has
quietly become worse at everything you did not measure.

So this repo **always evaluates two held-out sets**:

```
domain perplexity   did it LEARN?
general perplexity  what did it FORGET?
```

> ### What this project actually measured, and why the above is not enough
>
> Two things went wrong with this methodology in practice, and the second is
> more important than the first.
>
> **Mistake 1 — the probe overlapped the replay corpus.** The first run
> replayed wikitext into training and measured "forgetting" on *held-out
> wikitext*. Different documents, same distribution. General perplexity
> improved 40% and the run reported **−40.5% forgetting**. It was measuring
> "did we also learn wikitext". Fixed: the probe now comes from `ag_news`,
> never replayed, never trained on. `forgetting_dataset` is a separate config
> field and a test asserts it differs from `replay_dataset`.
>
> **Mistake 2 — perplexity was the wrong instrument entirely.** With the
> independent probe, forgetting still came out **−32.9%**: general perplexity
> *improved* by a third. Yet the same checkpoint scored **macro F1 0.002 and
> parse rate 0.015** on the downstream task with no SFT.
>
> Perplexity improved everywhere while the model could not follow an
> instruction. **Next-token perplexity on plain prose cannot see the loss of
> a behaviour.** If your starting model is instruction-tuned, the forgetting
> probe must be a *capability* evaluation, not a perplexity.
>
> (A control run later showed the 1.7B model never had that capability to
> begin with — it scored 0.001 before any CPT — so no forgetting had actually
> occurred. That does not rescue the methodology: perplexity would not have
> detected the loss either way, which is the point.)

Both are printed at every evaluation step, with the general-set drift coloured
red when it grows. A run that improves the first while wrecking the second is
a failed run.

Three levers control forgetting, in order of effect:

1. **Learning rate.** The single biggest factor. See §9.
2. **Replay fraction.** See §6.
3. **Number of steps.** More adaptation means more drift; the best checkpoint
   is often not the last.

## 6. Replay

The cheapest defence against forgetting is to **keep showing the model general
text while it learns the domain.**

`replay_fraction: 0.15` means 15% of training blocks are general English
(wikitext here; any broad corpus works). The model keeps being rewarded for
general fluency, so the weights cannot drift as far.

Two rules that are easy to get wrong:

1. **The general validation set must never be trained on.** `corpus/build.py`
   splits the replay pool: the first `val_docs` become held-out general
   validation, the remainder go into training. If you train on your forgetting
   probe, it stops measuring forgetting.
2. **Measure the baseline before training.** Forgetting is a *delta*. The
   trainer computes both perplexities at iteration 0, before a single weight
   moves, and everything afterwards is reported against that.

Try `replay_fraction: 0.0` once, deliberately. Watching general perplexity
climb is the most memorable half hour in this project.

## 7. Why full fine-tuning, not LoRA

The SFT lab used LoRA and trained 0.365% of parameters. Here the default is
`fine_tune_type: full` — every weight.

The reason is a genuine, measurable property: **LoRA learns less and forgets
less.** The low-rank constraint limits how far the model can move, which is
*ideal* for SFT (you want a small behavioural correction and no collateral
damage) and *limiting* for CPT (you are trying to move what the model knows).

Knowledge lives distributed across all the weights, especially the MLP blocks
that LoRA adapts only partially. A rank-16 update simply cannot store much new
knowledge.

The cost is real, and measured on an M4 Pro with Qwen3-1.7B at block 1024:

| batch | peak memory | time/step |
|---|---|---|
| 1 | 17.2 GB | 2.74 s |
| 2 | 24.1 GB | 5.23 s |

Those are a *bare* forward+backward+step. The real loop adds more, and the
measured behaviour is worth studying because it contradicts the usual reflex.

### Peak memory is dominated by the optimizer, not the batch

Logged from the actual TAPT run at `batch_size: 1`, `grad_accumulation_steps: 16`:

```
it  10   12.3 GB      <- before the first optimizer step
it  20   49.0 GB      <- after it
it  30   49.0 GB
```

The first optimizer step lands at iteration 16. **Everything below 12 GB is
activations and gradients; the other 37 GB appears the instant AdamW allocates
its state.** For a 1.7B model in bf16:

| what | size | scales with |
|---|---|---|
| parameters | 3.4 GB | model |
| gradients | 3.4 GB | model |
| **accumulator** (an extra gradient tree) | **3.4 GB** | model |
| **AdamW moments** (m and v) | **~13.6 GB** | model |
| activations | small at batch 1 | **batch × sequence** |

The consequence is counterintuitive: **dropping batch size from 2 to 1 barely
moved peak memory** (48.7 → 49.0 GB) because peak is set by the optimizer
step, not the forward pass.

It was still worth doing. Steady-state pressure fell, so the machine stopped
swapping and throughput improved **2.7×: 9.1 s/iter → 3.4 s/iter.**

So when a full fine-tune will not fit, reaching for a smaller batch is usually
the wrong lever. The levers that actually move peak memory are a smaller
model, a cheaper optimizer state (SGD/Adafactor instead of AdamW), or dropping
the accumulator by stepping every iteration. In LoRA none of this matters,
because the gradient tree and optimizer state are proportional to the *adapter*
— which is exactly why SFT fits a 4B model in 8.5 GB and CPT needs 49 GB for
1.7B.

Compare the SFT lab: 8.5 GB for a **4B** model, because only the adapter had
gradients and optimizer state. Full FT needs weights + gradients + Adam
moments, roughly 8× the parameter memory. That is why this lab defaults to
1.7B where the SFT lab used 4B.

`fine_tune_type: lora` is offered so you can measure the difference yourself —
that is [experiment #4](#11-the-experiment-grid).

## 8. Corpus hygiene is the actual work

In SFT, label quality was the work. In CPT there are no labels, and the work
moves to **corpus hygiene**. Three stages, each reported so it cannot silently
eat your data.

### Deduplication

**This matters far more for CPT than SFT.** With no labels to anchor it, the
model predicting every token of duplicated text simply memorises it, and every
duplicate token is budget spent teaching recitation.

Our corpus is duplicate-heavy by nature: Geolex publishes several reference
summaries per unit, often quoting each other nearly verbatim.

Exact hashing only catches byte-identical text. Two abstracts differing by one
year are different strings, same content. So we compare **sets of word
shingles** by Jaccard similarity:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

All-pairs is O(n²) — 100k documents is 5 billion comparisons. **MinHash**
reduces each document to a signature whose collision probability *equals* the
Jaccard similarity; **LSH** buckets signatures so only plausible pairs are ever
compared. O(n²) becomes roughly O(n).

### Quality filtering

Classic cheap heuristics: too short, low alphabetic ratio (tables, coordinate
dumps), digit-heavy (assay appendices), repetitive, no sentence structure.

**Every rejection is counted and reported by reason, with an example.** A
filter that silently eats 40% of your corpus is worse than no filter, because
you will never know. Read the histogram and sample the rejects before trusting
it — that is how we discovered the filter was inappropriate for TAPT (§2).

### Mixing

Blend in replay text, hold out clean validation for both domains, shuffle.

## 9. Hyperparameters

| knob | SFT lab | **this lab** | why |
|---|---|---|---|
| learning rate | 1e-4 | **1e-5** | the model is already at a good minimum; CPT nudges, it does not reshape. Too high is the fastest route to forgetting. |
| method | LoRA r=16 | **full** | knowledge is distributed; low-rank cannot hold it |
| masking | prompt masked | **none** | every token is a target |
| batching | padded pairs | **packed blocks** | documents vary hugely in length |
| grad clipping | not needed | **1.0** | one bad batch can wreck every weight in full FT |
| model size | 4B | **1.7B** | full FT needs ~8× the parameter memory |

The learning rate deserves emphasis. **1e-5 is not a typo, and 1e-4 will
visibly damage the model.** In SFT the LoRA adapter starts at zero and has to
be driven somewhere; in CPT every weight already encodes something you want to
keep.

Gradient accumulation still applies, and **so does the trap from the SFT lab**:
MLX indexes LR schedules by *optimizer steps*, not iterations. `build_schedule`
converts explicitly and prints the resolved numbers before training starts.

## 10. Evaluating CPT: three questions

CPT has no task, so "accuracy" is undefined. Three questions in increasing
order of what they actually prove:

### Q1 — did it learn? (domain perplexity)

`geocpt eval`. Should fall. Easy to move, and **weak evidence**: perplexity
improves when the model learns the *style* of USGS prose — the hedging, the
abbreviations, the rhythm — which is real but is not knowledge.

### Q2 — what did it forget? (general perplexity)

Same command, second row. Should stay roughly flat. A large rise means you
traded general capability for domain fluency.

**Necessary but not sufficient**, and this project proved it. Two rules:

1. The probe corpus must never appear in replay, or you measure whether you
   learned the replay distribution (`forgetting_dataset` ≠ `replay_dataset`).
2. Perplexity only detects forgetting of *language modelling*. It is blind to
   the loss of *behaviours* such as instruction-following. Measured here:
   general perplexity **improved 33%** on an independent corpus while the
   model scored **0.002 macro F1** on the task. For an instruction-tuned
   starting model, add a capability check — run `geosft eval` on the CPT'd
   model with no adapter.

### Q3 — does it know more? (cloze probes)

`geocpt probe`. Blank a domain term in held-out text and check whether the
model ranks the true term above distractors from the same class:

```
"The Austin Chalk is of Late ______ age."   →  Cretaceous
                                     vs Jurassic / Devonian / Permian
```

Chance is 0.25 with three distractors. This separates *knowledge* from *style*
in a way perplexity cannot. It is still a weak proxy — it measures whether the
right token is more likely, not whether the model can reason.

### Q4 — the one that actually matters: downstream transfer

`geocpt transfer`. Quantise the CPT'd model, run **geo-sft's own trainer and
evaluator** on it — same data, same hyperparameters, same metrics — and compare
macro F1 against the SFT-only baseline of **0.831**.

This is the only question with a business answer. Q1–Q3 are diagnostics.

## 11. The experiment grid

Because geo-sft's baseline exists, this project is a controlled experiment:

```
            ┌─ none      ──┐
starting    ├─ TAPT      ──┤
model  ─────┼─ DAPT      ──┼──► SFT ──► macro F1  vs  0.831
            └─ DAPT→TAPT ──┘
```

| # | experiment | what it tests |
|---|---|---|
| 1 | TAPT vs none | does cheap, on-distribution CPT help at all? |
| 2 | DAPT vs TAPT | does *more, broader* domain text help further? |
| 3 | replay 0.0 / 0.15 / 0.30 | the forgetting/learning tradeoff, directly |
| 4 | full vs LoRA CPT | "LoRA learns less and forgets less", measured |
| 5 | LR 1e-6 / 1e-5 / 1e-4 | find the forgetting cliff yourself |
| 6 | packed vs `no_cross_document` | is contamination worth 38% of your compute? |
| 7 | dedup threshold 0.6 / 0.8 / off | how much duplication actually hurts |
| 8 | **vocabulary extension + CPT** | **the open question geo-sft left** |

Experiment 8 closes the loop. geo-sft grafted 512 geoscience tokens and got a
4.89% context saving but could not train the new embedding rows — MLX LoRA
never touches embeddings. **CPT does.** Full fine-tuning trains the embedding
matrix, so this is where vocabulary extension can finally pay off. Start the
CPT run from geo-sft's `artifacts/models/*-geo-ext` and compare.

## 12. A guided first session

**1. Check the environment (1 min).**
```bash
make doctor
```
Confirms MLX, disk, and that geo-sft's task data exists next door.

**2. Build the TAPT corpus (2 min).**
```bash
geocpt corpus -c configs/tapt.yaml
```
**Read the output.** The quality-rejection histogram and the dedup counts are
the point of the exercise, not a progress bar.

**3. Pack, and read the efficiency numbers (1 min).**
```bash
geocpt pack -c configs/tapt.yaml
```
Then re-run with `no_cross_document: true` and compare. Seeing efficiency drop
from 100% to ~62% makes the tradeoff concrete.

**4. Train (~1 h).**
```bash
geocpt train -c configs/tapt.yaml --name tapt-v1
```
Watch the two perplexities at each eval. Domain should fall; general should
barely move. That pair of numbers is the CPT equivalent of a loss curve.

**5. Evaluate.**
```bash
geocpt eval  --checkpoint runs/tapt-v1/checkpoint-000600
geocpt probe --checkpoint runs/tapt-v1/checkpoint-000600
```

**6. The payoff.**
```bash
geocpt transfer --checkpoint runs/tapt-v1/checkpoint-000600
```
Compare against 0.831. **Be prepared for no improvement** — at 1.5M tokens
that is the likely outcome, and it is a real result about scale.

**7. Then run experiment 3** (replay 0.0). Watching general perplexity climb
teaches forgetting better than any explanation.

## 13. Failure modes

| symptom | cause | check |
|---|---|---|
| General ppl rises sharply | catastrophic forgetting | lower LR, raise `replay_fraction` |
| Domain ppl falls, downstream unchanged | learned style, not knowledge | run `geocpt probe` — did Q3 move? |
| Loss spikes then NaN | LR too high for full FT | clip harder, drop to 1e-6 |
| Loss drops implausibly fast | duplicate-heavy corpus being memorised | check the dedup report |
| Both perplexities rise | LR far too high, or corrupted packing | decode a block and read it |
| Out of memory | full FT needs ~8× parameter memory | smaller model, `grad_checkpoint`, batch 1 |
| Disk vanishes | checkpoints are full models (~3.4 GB) | lower `keep_last_n` |
| Downstream *worse* after CPT | drifted away from instruction tuning | expected — CPT must be followed by SFT |

## 14. Glossary

**Block** — a fixed-length token sequence, the unit of CPT training.

**Catastrophic forgetting** — losing prior capability while acquiring new.

**CPT** — continued pre-training.

**DAPT** — domain-adaptive pretraining: CPT on broad domain text.

**Jaccard similarity** — |A∩B|/|A∪B| over shingle sets.

**LSH** — locality-sensitive hashing; buckets similar signatures.

**MinHash** — a compact signature whose collision probability equals Jaccard.

**Packing** — concatenating documents and slicing into equal blocks.

**Perplexity** — exp(mean cross-entropy).

**Replay** — mixing general text into domain training to limit forgetting.

**Shingle** — an overlapping word n-gram.

**TAPT** — task-adaptive pretraining: CPT on the task's unlabelled text.
