# Results

Everything measured on this project, with the caveats that make the numbers
mean something. The data is Geoscience Australia's stratigraphic lexicon
(ASUD); the earlier USGS Geolex version of this lab, its runs and its results
are kept in [`runs/geolex-archive/`](../runs/geolex-archive/RESULTS-geolex.md)
and compared against where it helps.

---

## 1. Setup

| | |
|---|---|
| Hardware | Apple M4 Pro, 48 GB unified memory |
| Base model | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| Method | QLoRA — frozen 4-bit base, rank-16 LoRA on the top 16 layers |
| Trainable | 14.68 M of 4,022 M parameters (**0.365%**) |
| Effective batch | 16 (batch 4 × accumulation 4) |
| LR | 1e-4 peak, linear warmup then cosine to 1e-5 |
| Loss | completion-only (`mask_prompt`), on the JSON answer alone |

Identical to the Geolex run: only the data changed.

## 2. Dataset

| stage | count |
|---|---|
| ASUD units (WFS index joined to state reports) | 18,387 |
| Units with prose | 8,094 |
| Passages (2,307 definition cards + 20,004 reference notes) | **22,311** |
| Dropped as duplicates | 57 |
| Dropped as low-signal (<2 filled fields) | 3,679 |
| **Kept** | **18,575** across ~7,300 units |
| train / valid / test | 14,691 / 1,863 / 2,021 |
| gold (reviewed, frozen, held out of everything) | 150 |

Against Geolex: 2.5× the passages and **2.2× the training pairs** (6,635 →
14,691). The training budget was kept at 900 iterations, so the run sees
3,600 examples — 0.24 of an epoch rather than 0.54. More data here buys
diversity, not more compute.

Source snapshot: `data/raw/asud/manifest.json` (report ZIPs dated
20 Sep 2026, SHA-256 per file).

### Labeller audit

The labeller never reads ASUD's curated metadata, which leaves it free as
independent ground truth. ASUD curates far more than Geolex did, so almost
every field has a tripwire (full corpus, 22,311 passages):

| check | ASUD | Geolex |
|---|---|---|
| unit_name retained | 99.9% | 85.5% |
| rank agrees with curated rank | 98.3% | — |
| thickness inside curated range | 89.2% | — |
| relations also curated by ASUD (precision floor) | 59.7% | — |
| chronostrat token overlap with curated ages | 51.9% | 89.5% |
| states precision / recall | 0.841 / 0.041 | 0.581 / 0.233 |

Chronostrat consistency *fell*, and not because the rules got worse: ASUD
curates fine-grained ICS ages (Statherian, Cisuralian) while reference notes
cite stages (Sakmarian) or ranges in Ma, and a token-overlap check cannot see
that Sakmarian lies within Cisuralian. States recall is tiny because a
reference note almost never names a state — ASUD records the jurisdiction in
the table, not the prose.

## 3. Gold review

All 150 gold rows were reviewed against the passage (a **machine review by
claude-opus-5.5, not a geologist's** — the same standard as the Geolex
review). **86 of 150 (57%) had at least one wrong label**, against 125 of 150
(83%) for Geolex. Notes per row are in `data/gold/corrections.json`.

The most common rule errors, in order:

1. **Coordination.** "Overlain by Pargee Sandstone, Muriel Range Sandstone,
   Lucas Formation and Pedestal beds" yields one relation, not four.
2. **Facts belonging to a neighbour.** "Unconformable on Precambrian rocks" is
   the *underlying* unit's age; "overlies basalt ... of the Eastern Creek
   Volcanics" is the underlying unit's rock.
3. **Metamorphic facies read as rocks.** "amphibolite facies" is a grade, not
   an amphibolite.
4. **ASUD house style** the rules did not cover: `Sltst`, `C.M.`, `BIF`,
   "Overlies: X", "Over X; under Y", "overlain conformably by".
5. **Superseded statements.** "Was previously considered equivalent to X"
   is labelled as an equivalence.

### Review-driven rule fixes — and the leakage they create

The review found five *mechanical* rule bugs (a reversed participle
direction, a self-relation check broken by ASUD's rank-bearing names,
hyphenated tokens hidden from the gazetteer, author citations "(Flint et al."
read as rocks, a name mask that swallowed full stops). All were fixed, with
regression tests, **after** the gold review, and the training labels were
rebuilt with the fixed rules.

That makes the gold score below **optimistic**: the labels the model trained
on were improved using what the gold rows revealed. Measured directly, the
final rules score **macro F1 0.855 against the reviewed gold** — a number that
is circular for the same reason. A clean measurement needs a second gold slice
drawn and reviewed *after* the rules were frozen (§8).

The reviewed gold set is **frozen**: `build` keeps it verbatim and forces its
units into test, so no later labeller change can silently reshuffle it.

## 4. Training run — `geo-sft-v1`

900 iterations, **167 minutes** (the GPU was shared with the graph lab's
ingest and benchmark; the Geolex run took 117), 13.5 GB peak.

| iter | val loss |
|---|---|
| 0 | 1.4031 |
| 74 | 0.1342 |
| 224 | 0.0751 |
| 449 | 0.0594 |
| 674 | 0.0596 |
| **824** | **0.0477** ← best |
| 899 | 0.0618 |

Test loss 0.0497 (perplexity 1.05). The starting loss is lower than Geolex's
(1.40 vs 2.27): the ASUD target is easier to *format* because every passage
leads with its unit name.

**A false overfitting alarm.** At iteration 674 the monitor fired "val loss
has not improved for 3 evals — stop here". It then improved to its best at
824. Validation is 20 batches, noisy enough to plateau for three evals by
chance. Patience rules on noisy validation stop runs early; that is worth
seeing once on your own curve.

![loss curve](../runs/geo-sft-v1/loss_curve.png)

## 5. Evaluation — base vs tuned

Greedy decoding, identical prompts for both.

| metric | base (test) | tuned (test) | base (**gold**) | tuned (**gold**) |
|---|---|---|---|---|
| schema valid rate | 0.910 | 1.000 | 0.867 | **0.993** |
| unit_name accuracy | 0.990 | 0.995 | 0.993 | 0.993 |
| rank accuracy | 0.890 | 0.970 | 0.887 | **0.973** |
| thickness accuracy | 0.685 | 0.955 | 0.687 | **0.947** |
| lithologies F1 | 0.307 | 0.846 | 0.359 | **0.825** |
| chronostrat F1 | 0.164 | 0.837 | 0.120 | **0.667** |
| minerals F1 | 0.590 | 0.837 | 0.554 | **0.854** |
| states F1 | 0.174 | 0.880 | 0.126 | **1.000** |
| relations F1 | 0.322 | 0.632 | 0.438 | **0.691** |
| **macro F1** | **0.311** | **0.806** | **0.320** | **0.807** |

Test = 200 rows scored against rule labels. Gold = 150 reviewed rows.

### Reading the table

**The SFT gain replicates on new data.** +0.495 (test) and +0.488 (gold)
here; +0.516 and +0.312 on Geolex. Base models land at 0.31–0.32 on both
corpora — a general 4B instruct model does about a third of this task
unaided.

**The ~40% rule-label inflation of the Geolex run has gone** (0.831 → 0.705
there; 0.806 → 0.807 here). Partly real — the ASUD rules are better (57% of
rows needed correction, not 83%) — and partly the leakage in §3. Do not read
it as "rule labels are fine now".

**The model is a slightly worse copy of its teacher.** Against gold, per
field:

| field | rules | tuned model |
|---|---|---|
| lithologies | 0.864 | 0.825 |
| chronostrat | 0.727 | 0.667 |
| minerals | 0.910 | 0.854 |
| relations | 0.772 | 0.691 |

It does not generalise past the rules; it imitates them imperfectly, losing
most on **relations recall** (0.58 vs the rules' 0.68). That is what
weak supervision predicts when the rules are high-precision and the budget is
a quarter of an epoch. The flattering comparison for the rules is itself
circular (§3), so the honest reading is "no evidence the model beats its
labels", not "the model is worse than regexes".

**Chronostrat is the base model's worst field** (0.12 on gold): an instruct
model tags every age it sees, including the neighbours' and the orogenies'.
The tuned model learned the house convention; its remaining error is still
precision (0.55) — the same neighbour-age problem the gold review corrected
most often.

**States F1 1.000 on gold rests on 7 items.** Few passages name a state.

## 6. Tokenizer analysis

An 8,000-token BPE tokenizer trained on the train split
(`artifacts/tokenizer_report.json`):

| corpus | base (Qwen3) | geo BPE |
|---|---|---|
| geoscience passages | 1.921 | 1.610 |
| general English | 1.123 | 1.877 |

ASUD prose fragments *more* than Geolex did (1.92 vs 1.70 tokens per word
for the base tokenizer): Australian lexicon prose is dense with igneous rock
names. The costliest terms: siltstone (3 tokens × 2,951 uses), granodiorite
(4 × 1,858), Biotite (3 × 1,406), rhyolite (4 × 1,135), monzogranite (5 × 538).

The vocabulary-extension experiment (graft domain tokens, retrain) has **not
been re-run on ASUD**; its Geolex result (context saving 4.89%, downstream
−0.084 macro F1 at 1.7B) is in the archive.

## 7. What these numbers do *not* establish

- **The gold score is optimistic** (§3): labeller fixes were informed by the
  gold review.
- **One run, one seed, one configuration.** No sweep of rank, depth, data size
  or model size. The Geolex version's three-seed TAPT result has not been
  repeated here.
- **Machine review, not a geologist's.** The conventions (no rock words from
  proper names, no facies as rocks, neighbours' facts excluded) are defensible
  but not authoritative.

## 8. Next experiment worth running

**A second, clean gold slice.** Delete nothing: draw 150 *more* test rows
with `build` (a new file), review them against the frozen rules, and score
both the rules and the model. The difference between that score and §5's is
the measured size of the leakage — the same lesson as the Geolex review, one
level up: *whoever builds the test set decides what "better" means, including
you, after you have looked at it.*

## 9. Bugs found while building

The Geolex-era bugs (the `re.IGNORECASE` name swallow, the
gradient-accumulation LR trap, lowercase vocabulary grafting, the
template-prefix trap, the gold/test report overwrite, the grepping regression
test) are in the [archived results](../runs/geolex-archive/RESULTS-geolex.md)
and still guarded by tests. New with ASUD:

| bug | symptom | why it is nasty |
|---|---|---|
| rank from the word *after* the name | rank filled on 1.1% of passages | ASUD names carry their rank; Geolex names did not |
| `rstrip("s")` on "Beds", "Volcanics" | "X Beds" labelled Bed; "Volcanics" matched nothing | a one-character normalisation, two opposite errors |
| state-report record stitching | a newline in the last field glued the fragment onto the **next** record's STRATNO | corrupts identity silently; row counts barely move |
| participle direction | "an interval overlying X" labelled as *underlies* X | the article ("the overlying X") is the whole difference |
| self-relation check | "Breakfast Sandstone overlies Breakfast Sandstone"; then, once fixed, Murchison Granite discarded as Murchison Volcanics | the fix for one broke the other |
| name mask swallowed full stops | the next sentence's "Sandstone," read as mid-sentence and dropped | a trailing `\.?` in a regex |

The common thread, again: **none of them raised an error.** Every one
produced a plausible dataset and a plausible loss curve.
