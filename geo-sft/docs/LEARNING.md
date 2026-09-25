# Learning SFT with this repo

A guide to supervised fine-tuning as a *subject*, using this project as the
worked example. Everything here is grounded in numbers this repo actually
produced, so you can reproduce each claim.

Read this first — sections 1–5 before you run anything. The README has a
[four-session learning path](../README.md#how-to-learn-from-this-repo) that
interleaves these sections with actually running the pipeline, which is the
recommended way through. [ARCHITECTURE.md](ARCHITECTURE.md) is the code tour,
[RESULTS.md](RESULTS.md) the outcomes, [EXPERIMENTS.md](EXPERIMENTS.md) what to
run next.

**Contents**
1. [What SFT is, and what it is not](#1-what-sft-is-and-what-it-is-not)
2. [What actually changes inside the model](#2-what-actually-changes-inside-the-model)
3. [LoRA and QLoRA](#3-lora-and-qlora)
4. [The anatomy of one training example](#4-the-anatomy-of-one-training-example)
5. [Loss: what it measures, and why it is not your metric](#5-loss-what-it-measures-and-why-it-is-not-your-metric)
6. [Steps, iterations, epochs, optimizer steps](#6-steps-iterations-epochs-optimizer-steps)
7. [The hyperparameters, physically](#7-the-hyperparameters-physically)
8. [Reading a training run](#8-reading-a-training-run)
9. [Evaluating honestly](#9-evaluating-honestly)
10. [Where the data comes from](#10-where-the-data-comes-from)
11. [Tokenizers: the part everyone gets wrong](#11-tokenizers-the-part-everyone-gets-wrong)
12. [A guided first session](#12-a-guided-first-session)
13. [Failure modes you should be able to name](#13-failure-modes-you-should-be-able-to-name)
14. [Glossary](#14-glossary)

---

## 1. What SFT is, and what it is not

Supervised fine-tuning is: **show the model input→output pairs, and adjust its
weights so it produces those outputs.** That is the whole idea. Everything
else is engineering around that sentence.

It sits in a stack of techniques that are often confused:

| technique | changes weights? | teaches | costs |
|---|---|---|---|
| **Prompting** | no | nothing — you steer what is already there | pennies |
| **RAG** | no | nothing — you supply facts at inference | retrieval infra |
| **SFT** (this repo) | yes | behaviour, format, conventions, domain patterns | hours on a laptop |
| **RLHF / DPO** | yes | preferences between outputs | SFT first, plus preference data |
| **Continued pretraining** | yes | genuinely new knowledge and vocabulary | billions of tokens |

The single most useful thing to internalise: **SFT is far better at teaching
*form* than at teaching *facts*.** You are reshaping how the model responds,
not stuffing new knowledge into it.

This repo demonstrates that split cleanly. Two fields from the same run:

- `unit_name` went **0.101 → 0.946**. Huge — but it is almost entirely
  convention. The base model answers `"Amsden Group"`; our schema wants
  `"Amsden"` with the rank in a separate field. The base *cannot* know that.
  We taught it a house style.
- `lithologies` went **0.332 → 0.934**. This is closer to real learning: the
  base misses rock types buried in archaic adjectival phrasing
  (`argillaceous`, `fossiliferous`) and fails to normalise them to a fixed
  vocabulary. The tuned model does both.

If your task is "the model needs to know things it doesn't know", SFT is
probably the wrong tool and you want RAG or continued pretraining. If your
task is "the model knows roughly enough but won't reliably produce what I
need", SFT is exactly right.

## 2. What actually changes inside the model

A transformer is a stack of layers; each contains attention projections
(`q_proj`, `k_proj`, `v_proj`, `o_proj`) and an MLP (`gate_proj`, `up_proj`,
`down_proj`). Each is a big matrix multiply. "Training" means nudging the
numbers in those matrices.

Full fine-tuning updates all of them. For a 4B model in bf16 that means 8 GB
of weights, plus gradients (8 GB), plus Adam optimizer state (16 GB+) — over
32 GB before activations. Painful on a laptop, and you would be moving 4
billion numbers to teach a JSON convention.

The insight behind LoRA: **the update you need is low-rank.** You are not
rebuilding the model, you are applying a small correction, and small
corrections live in a much smaller space than the full weight matrix.

## 3. LoRA and QLoRA

### LoRA

For a weight matrix `W` of shape `(d, k)`, instead of learning a full update
`ΔW` of the same shape, learn two thin matrices:

```
W' = W + (α/r)·B·A          A: (r, k)   B: (d, r)   r ≪ min(d, k)
```

`A` is initialised random, `B` is initialised **zeros** — so `B·A = 0` and at
step 0 the model is *exactly* the base model. Training is undisturbed at the
start, and the adapter grows from nothing.

Two things follow:

- **`r` (rank) is capacity.** It caps how expressive the correction can be.
  For format and extraction tasks, 8–16 is plenty. Raise it only if *training*
  loss plateaus high — that is the signal of insufficient capacity.
- **`α` (alpha) is a scale.** The update is multiplied by `α/r`. Keeping
  `α = 2r` is a common convention (this repo uses r=16, α=32, so scale 2.0),
  which means changing `r` doesn't silently change your effective learning
  rate.

In this repo, adapting 16 of Qwen3-4B's 36 layers at rank 16:

```
Trainable parameters: 0.365% (14.680M / 4022.468M)
```

**14.7 million trainable parameters instead of 4 billion.** That is the whole
trick. Gradients and optimizer state scale with the adapter, not the model.

Because `B` starts at zero, LoRA is also *composable and reversible*: the
adapter is a separate 58 MB file, and the base model on disk is untouched. You
can train ten adapters for ten tasks against one base.

### QLoRA — the "Q"

QLoRA adds: **quantize the frozen base to 4 bits.**

The base weights never receive gradients, so they do not need full precision —
they only need to be accurate enough for the forward pass and for gradients to
flow *through* them to the adapter. Storing them at 4 bits instead of 16
quarters the memory.

In this project the base is loaded already quantised (MLX uses affine
quantization, group size 64; the extended model measured **4.501 bits per
weight** including the scale/zero-point overhead). Peak memory for the whole
900-iteration run was **8.5 GB** — comfortable inside an M4 Pro's 48 GB, and
the reason a 4-bit base is the *point* here rather than a compromise.

> **On a Mac, MLX is the real QLoRA path.** bitsandbytes (the usual 4-bit
> library) has no working MPS backend, so the HuggingFace path in this repo
> runs bf16 LoRA on a full-precision base. It is kept because PEFT/TRL is the
> API you will meet everywhere else, and because it is the only path that can
> train grafted embedding rows. The CUDA QLoRA block is there, commented.

## 4. The anatomy of one training example

This is where most silent failures live, so it is worth being very concrete.

Start with the chat messages (`schema.py:build_messages`):

```
system:    You are a geoscience information-extraction engine... [schema]
user:      Passage: Wilton Formation. Overlies Woonona Coal. Overlain by Appin...
assistant: {"unit_name": "Wilton Formation", "rank": "Formation", ...}
```

The chat template renders that into a flat token sequence:

```
<|im_start|>system\n...<|im_end|>\n<|im_start|>user\nPassage:...<|im_end|>\n<|im_start|>assistant\n{"unit_name":...}<|im_end|>\n
└──────────────────── PROMPT (masked) ────────────────────────────────────┘└──── COMPLETION (loss) ────┘
```

The model is trained to predict the next token at every position. But you do
**not** want loss on the prompt. Predicting the passage back is not the task,
and in this project the prompt is ~4× longer than the answer — so without
masking, ~80% of your gradient signal teaches the model to recite geological
prose.

That is `mask_prompt: true`, and it is the **single biggest quality lever**
here. Measured on a real example (`geosft inspect --index 3`): **388 total tokens,
278 masked (72%), 110 carrying loss.** Run that command on your own data
before every new run — it takes a second and catches a whole family of bugs.

Two details that matter more than they look:

**The EOS token is inside the loss region.** `...}<|im_end|>` — the model must
learn to *stop*. Omit it and your model produces valid JSON followed by an
endless ramble.

**The prompt at training must be byte-identical to the prompt at inference.**
This is the property everything depends on, and this repo has one function
(`build_messages`) that both paths call, plus a test asserting it.

> ### A real bug worth studying
>
> Qwen3's chat template renders a *full conversation* with
> `<think>\n\n</think>\n\n` injected before the assistant content — but
> `add_generation_prompt=True`, which is what inference actually supplies,
> stops at `<|im_start|>assistant\n`. mlx-lm tokenises the full form and
> derives the mask boundary from the generation form, so that scaffolding
> landed **inside the loss region**.
>
> The model learned to emit it. Every single generation came out as
> `<think>\n\n</think>\n\n{...}`. Symptom: **strict JSON rate went 1.000 on the
> base to 0.000 on the tuned model, while macro F1 rose by 0.5.**
>
> And note why masking it is *not* the fix: masked tokens are removed from the
> loss but stay in the *input*. The model would then be trained to emit JSON
> conditioned on a prefix that does not exist at inference — a context
> mismatch, which is worse than the original symptom. The fix
> (`PromptCompletionDataset`) builds `prompt + answer + EOS` directly.

## 5. Loss: what it measures, and why it is not your metric

Training loss is **cross-entropy**: the negative log-probability the model
assigns to the correct next token, averaged over unmasked positions.

- loss 0.69 → the right token got ~50% probability
- loss 0.10 → ~90%
- loss 0.07 → ~93% (roughly where this project's run landed)

**Perplexity** is just `exp(loss)` — "how many options is the model effectively
choosing between". Test perplexity 1.073 means it is almost never surprised.

Loss is a wonderful *diagnostic* and a poor *objective*. Three reasons:

1. **It measures token agreement, not task success.** A model that outputs
   `["marl","chalk"]` when gold says `["chalk","marl"]` is semantically perfect
   and loss-wise wrong. (This is exactly why this repo sorts every list — see
   §10.)
2. **It is not comparable across configurations.** When the template-prefix bug
   was fixed, initial validation loss changed from 2.266 to 1.766 — not because
   anything improved, but because the loss is now averaged over a *different
   token set*. Comparing those two numbers would be meaningless.
3. **Lower loss can mean worse task performance.** Predicted outcome of
   [experiment #1](EXPERIMENTS.md): turning `mask_prompt` off will *lower*
   training loss (prose is easier to predict than exact JSON) while *lowering*
   task scores. Run it — it is the most convincing hour in the curriculum.

**Rule: use loss to diagnose the run, use task metrics to judge the model.**

## 6. Steps, iterations, epochs, optimizer steps

This vocabulary is a genuine trap, and it cost this project a wasted run.

| term | meaning | this repo |
|---|---|---|
| **example** | one training pair | 14,691 in train |
| **micro-batch / iteration** | `batch_size` examples processed in one forward+backward | batch 4 |
| **optimizer step** | one actual weight update | every 4 iterations |
| **effective batch** | examples per weight update | 4 × 4 = **16** |
| **epoch** | one full pass over the data | 14,691 / 4 ≈ 3,673 iterations |

**Gradient accumulation** means running several micro-batches, summing their
gradients, and only *then* updating. It buys a large effective batch without
the memory of a large batch — it costs time, not RAM. Prefer raising
accumulation over raising `batch_size` when you run out of memory.

> ### The trap
>
> MLX indexes the learning-rate schedule by **optimizer steps**, not
> iterations. Pass an iteration count into it and the schedule stretches by
> the accumulation factor.
>
> This repo originally had `warmup_steps: 30` with `grad_accumulation_steps:
> 8`. That warmed up over **240 iterations** — 40% of a 600-iteration run
> spent at effectively zero learning rate. Observed:
>
> ```
> it   10 train 2.2559 lr 6.67e-06     ← flat
> it   20 train 2.2608 lr 3.33e-06     ← flat
> it   30 train 2.2579 lr 6.67e-06     ← flat
> ```
>
> After converting config values (in iterations) to optimizer steps internally:
>
> ```
> it   10 train 2.2559 lr 6.67e-06
> it   30 train 1.1983 lr 4.00e-05
> it   50 train 0.5204 lr 7.33e-05     ← learning
> ```
>
> Nothing errored. The run completed. It just did not learn. **If your loss is
> flat, print the learning rate before you touch anything else.**

The 900-iteration run here saw 3,600 examples of 6,635 — about **0.54 epochs**.
It never saw half the data and still reached macro F1 0.831. Format-and-
extraction tasks need less data than people assume; [experiment
#3](EXPERIMENTS.md) measures where that curve flattens.

## 7. The hyperparameters, physically

Roughly in order of how much they matter for a task like this.

| knob | what it physically does | how to choose |
|---|---|---|
| `mask_prompt` | excludes prompt tokens from the loss | always `true` for instruction data |
| `num_layers` | how many transformer blocks (from the top) get adapters | 16 is a good default; `-1` = all, costs more memory |
| `lora_rank` | adapter capacity (the `r` in §3) | 8–16 for format tasks; raise only if *train* loss plateaus high |
| `learning_rate` | step size | `1e-4` is the LoRA default. `1e-3` often diverges, `1e-5` looks like nothing happening |
| `batch_size × grad_accumulation_steps` | examples per weight update | 16–32 is a sane effective batch |
| `warmup_steps` | ramps LR from 0 | ~5–10% of optimizer steps. Matters more for LoRA than people expect, because `B` starts at zero so early updates are large and badly scaled |
| `lora_alpha` | scales the adapter output by `α/r` | keep at `2r` so changing rank doesn't change effective LR |
| `max_seq_length` | truncation point | long enough for your longest example; costs memory quadratically in attention |
| `grad_checkpoint` | recompute activations instead of storing them | `true` when memory-bound; ~30% slower |

Why adapt *top* layers rather than all? Early layers encode general
lexical/syntactic structure that transfers fine; later layers encode
task-and-output-shaped behaviour. For teaching an output convention, the top
of the stack is where the leverage is.

## 8. Reading a training run

Here is the actual validation curve from `geo-sft-v1`, and how to read it:

```
iter     0   2.2662     ← untrained baseline on this task
iter    74   0.2940     ← steep phase: learning the FORMAT
iter   149   0.1539
iter   299   0.1157     ← gradual phase: learning the CONTENT
iter   449   0.0931
iter   599   0.0790
iter   749   0.0703     ← BEST
iter   824   0.0707     ← flat/rising: nothing more to gain
iter   899   0.0717
```

Three phases, and they are typical:

1. **The cliff (0→74).** Loss falls ~87% almost immediately. This is the model
   learning the *shape* of the answer — emit JSON, these keys, this order. It
   is nearly free.
2. **The grind (74→749).** Slow, real improvement as the model learns which
   content goes in which field.
3. **The plateau (749→900).** Validation bottoms and drifts *up*. The last 150
   iterations bought nothing. **This is why `save_every` exists** — take the
   best checkpoint, not the last one.

Also check the **train↔val gap**. Here: final train 0.0756, final val 0.0717.
Essentially zero gap (val is even marginally lower, which just means the
validation split is slightly easier) — the model is generalising, not
memorising. A widening gap, with train falling while val rises, is the classic
overfitting picture, and the monitor raises an alarm for it.

The live panel (`monitor/tracker.py`) shows all of this while running, plus
tokens/sec, peak memory and ETA, and raises four alarms — NaN loss, flat loss,
no val improvement for N evals, and a widening train↔val gap.

## 9. Evaluating honestly

This is the part most tutorials skip, and it is most of the value.

**Rule 1: always score the untuned base on the same prompts.** A macro F1 of
0.831 means nothing on its own. Against a base of 0.314 it is a real win;
against a base of 0.80 you wasted an afternoon. `geosft eval` always runs both.

**Rule 2: separate format compliance from content accuracy.** They fail
differently and want different fixes. This repo reports:

- `parse_rate` — did it emit parseable JSON at all?
- `schema_valid_rate` — does the JSON satisfy the schema?
- then per-field accuracy/F1

A large part of what SFT buys you is the first two. Keeping them separate
stops you crediting the fine-tune with domain skill it did not acquire.

**Rule 3: score an unparseable answer as wrong, not as skipped.** If you drop
unparseable outputs you reward a model for refusing to answer. Here, every
gold item in an unparseable response becomes a false negative.

**Rule 4: hold out by *group*, not by row.** ASUD files up to a dozen
overlapping notes under the same unit. A random split puts near-duplicates on both
sides and inflates your score — you would be testing on paraphrases of the
training set. Split on `unit_id`.

**Rule 5: know your label ceiling.** This is the big one here. The labels are
generated by regexes and gazetteers, so **a model trained on them learns to
imitate the regexes.** Scoring against those same rules will approach 100%
while proving nothing.

Two defences: the rules are deliberately high-precision/low-recall (so the
model must *generalise* past them to score well), and there is a 150-row gold
set to correct by hand:

```bash
geosft review --n 20        # print rows for checking
# fix data/gold/gold.jsonl, set "reviewed": true
geosft eval --adapter runs/geo-sft-v1 --gold
```

The interesting question the gold set answers: **does the tuned model find
things the rules missed?** If yes, it generalised past the gazetteer and you
have a domain model. If never, your labels are too narrow and the fix is
better data, not more training.

You can already see the ceiling in the results: `minerals` (0.680) and
`relations` (0.659) are the weakest fields, and both are the ones where the
labeller is weakest — sparse mineral mentions, and relation regexes that miss
unusual sentence forms. The model's scores track the labels' quality, exactly
as theory predicts.

## 10. Where the data comes from

Most SFT effort is data work, and this repo is honest about that: the data
pipeline is bigger than the training code.

**Weak supervision** is the technique here: rather than paying humans to label
8,000 passages, write rules that produce labels programmatically, then
distil those rules into a model. The model generalises past the rules because
it sees context the rules cannot.

Three principles worth stealing:

1. **Only label what is in the input.** ASUD gives us curated ages, states,
   thickness and even stratigraphic relations for each unit, and using them
   would be tempting. It would also teach
   the model to assert facts it cannot see — i.e. to hallucinate confidently.
   The metadata is used *only to audit* the labeller.
2. **Audit with held-out signal.** Because the labeller never reads the curated
   metadata, that metadata is independent ground truth. Name-derived ranks
   agree with ASUD's **98.3%** of the time, extracted thicknesses fall inside
   the curated range **89.2%** of the time, and **59.7%** of extracted
   relations are ones ASUD also curates. If you change a rule and one of those
   drops, you broke something.
3. **Make targets canonical.** Lists are sorted and de-duplicated. If the same
   facts could serialise two ways, you are training the model to predict a coin
   flip — pure noise in the gradient, and unnecessary loss you can never drive
   out.

## 11. Tokenizers: the part everyone gets wrong

> **You cannot swap a pretrained model's tokenizer and then fine-tune.**

Token IDs are row indices into the embedding matrix. Token 5432 means
"whatever vector sits in row 5432". Install a new tokenizer and every index
points at an unrelated vector — the model is destroyed. Recovering needs
continued *pretraining* on billions of tokens, not SFT on a few thousand
examples.

So what *is* a domain tokenizer good for?

**Measurement.** Fertility (tokens per word) tells you how badly the base
tokenizer handles your domain. Measured here:

| corpus | base (Qwen3) | geo tokenizer |
|---|---|---|
| geoscience | 1.701 | **1.527** (−10%) |
| general English | **1.123** | 1.938 (+73%) |

The second row is the lesson. A domain tokenizer wins modestly on domain text
and loses catastrophically on everything else. That is *why* nobody swaps them.

**Vocabulary extension** is the technique that legitimately feeds training:
keep the tokenizer, *add* a few hundred domain tokens, and initialise each new
embedding row as the **mean of the sub-word embeddings it replaces** (a
sensible starting point on the model's existing manifold, unlike random init).

```
base    (18 tokens): Penn|s|ylv|anian| s|ilt|stone| uncon|form|ably| ...
extended (7 tokens): Pennsylvanian| siltstone| unconformably| ...
```

Worth **4.89%** fewer tokens across held-out passages in the Geolex-era
version of this lab (the extension experiment has not been re-run on ASUD).

**But be clear about what that is.** It is a *cost* win — shorter sequences,
faster steps, more room in the context window. It is **not** evidence of a
quality win, and you should expect a null result on quality: vocabulary
extension is a pretraining-scale technique, and with a few thousand SFT
examples the new rows get little gradient. On the MLX path they get *none*
(LoRA adapts attention and MLP, not embeddings), so they keep their mean-init
values forever. Only the HF path can train them, via
`modules_to_save=["embed_tokens"]`.

[Experiment #5](EXPERIMENTS.md) is the A/B. **A null result is a real result** —
and learning to tell "cheaper" from "better" is the point.

## 12. A guided first session

About two hours, most of it unattended. Do these in order.

> This is the **command-by-command** version of the reading path in
> [the README](../README.md#how-to-learn-from-this-repo). That path tells you
> which sections to read alongside each step; this one is just the doing. Use
> them together rather than picking one.

**1. Check the machine (1 min).**
```bash
make doctor
```
Confirms MLX sees your GPU, reports the working set, and warns on low disk.

**2. Prove the wiring (5 min).**
```bash
make smoke
```
Runs every stage on Qwen3-1.7B with 300 units and 60 iterations. If something
is broken, you find out now instead of in an hour.

**3. Look at the data before you train on it.** This is the habit that
separates people who debug fine-tunes quickly from people who don't.
```bash
head -1 data/processed/train.meta.jsonl | python3 -m json.tool | head -40
cat data/processed/dataset_card.json
```
Read a few passages and their targets. Ask: *would I produce this label?* You
will find labeller mistakes. That is the point.

**4. Check what the model is actually trained on (1 min).** This is the check
that catches the entire family of silent prompt/masking bugs.
```bash
geosft inspect --index 3
```
It prints the prompt (masked, grey) and the loss region (trained, green), then
asserts the two invariants that matter:
```
 total tokens                      388
 masked (prompt)                   278  (72%)
 trained (answer)                  110  (28%)
 ends with EOS                     yes
 train prompt == inference prompt  yes
```
If `ends with EOS` is no, your model will never stop generating. If `train
prompt == inference prompt` is no, you have the bug from §4 and no amount of
training will fix it.

**5. Train, and watch the panel (~2 h).**
```bash
make train
```
Watch train and val loss, the gap between them, tokens/sec, peak memory. Note
when the steep phase ends — that is format being learned.

**6. Evaluate, and read the delta column.**
```bash
geosft eval --adapter runs/<name>
```
Not the tuned column. The **delta** column.

**7. Read what the model actually wrote.**
```bash
python3 -c "
import json
rows=[json.loads(l) for l in open('runs/<name>/predictions_tuned.jsonl')]
for r in rows[:5]:
    print(r['passage'][:200]); print('GOLD:', r['gold']); print('PRED:', r['parsed']); print()
"
```
Find a case where the model and the gold label disagree, and decide which is
right. Sometimes the model is. That is generalisation past the rules, and it
is the most encouraging thing you will see.

**8. Then pick an experiment.** [EXPERIMENTS.md](EXPERIMENTS.md) #1
(prompt masking) is the most instructive.

## 13. Failure modes you should be able to name

By the end of this project you should diagnose each of these in under a minute.

| symptom | likely cause | check |
|---|---|---|
| Loss is `nan` | LR too high | drop to 1e-5 and restart |
| Loss flat from step 1 | LR ≈ 0 (warmup/accumulation trap), or no trainable params | **print the LR**; check "Trainable parameters" > 0 |
| Train falls, val rises | overfitting | use the best checkpoint; fewer iters, more data |
| Both plateau high | insufficient capacity | raise `lora_rank` or `num_layers` |
| Great loss, bad outputs | train/inference prompt mismatch | print both prompts and diff them |
| Model never stops | EOS not in the loss region | check the completion includes EOS |
| Output has an odd fixed prefix | template scaffolding inside the loss region | decode the trained region (§4) |
| Base scores suspiciously well | task too easy, or leakage | check your split is group-wise |
| Tuned scores suspiciously well | leakage, or you are scoring rules against rules | use the gold set |

The theme of the four bugs found building this repo: **every one produced a
beautiful loss curve and no error message.** Silent failure is the normal
failure mode in this field. Your defences are printing intermediate state,
holding out data properly, and always comparing against the base.

## 14. Glossary

**Adapter** — the small trained LoRA matrices, saved separately from the base.

**Alpha (α)** — LoRA scaling; update multiplied by `α/r`.

**Base model** — the pretrained model you start from, frozen in LoRA training.

**Catastrophic forgetting** — losing general ability while learning a narrow
task. LoRA mitigates it: base weights are frozen, and the adapter is removable.

**Checkpoint** — adapter weights saved mid-run.

**Completion-only loss / prompt masking** — computing loss only on the answer.

**Cross-entropy** — the loss function; negative log-probability of the correct
token.

**Effective batch** — `batch_size × grad_accumulation_steps`.

**Epoch** — one full pass over the training set.

**Fertility** — tokens per word; lower means more efficient tokenization.

**Gradient accumulation** — summing gradients over several micro-batches before
updating, for a large effective batch at small-batch memory cost.

**Gradient checkpointing** — recomputing activations in the backward pass to
save memory, at ~30% time cost.

**LoRA** — Low-Rank Adaptation; train `B·A` instead of the full `ΔW`.

**Perplexity** — `exp(loss)`; effective number of choices.

**QLoRA** — LoRA on a 4-bit quantized frozen base.

**Quantization** — storing weights at lower precision (here 4-bit affine,
group size 64, measured 4.501 bits/weight with overhead).

**Rank (r)** — the inner dimension of the LoRA matrices; adapter capacity.

**Warmup** — ramping LR from 0 over the first steps.

**Weak supervision** — generating labels programmatically instead of by hand.
