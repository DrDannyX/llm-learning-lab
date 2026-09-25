# Results

One run: `runs/bench-v1/` (64 questions × 3 systems, `google/gemma-4-12b` with
thinking off, as both answerer and judge; nomic-embed-text-v1.5; the whole
ASUD lexicon). **One run, one seed, 8 questions per category.** A single
question moves a category score by 0.125. Read the shape, not the second
decimal.

The earlier USGS Geolex version of this lab is archived in
[`runs/geolex-archive/`](../runs/geolex-archive/RESULTS-geolex.md) and
compared against below: same code, same models, same benchmark design,
different corpus.

## The corpus

| | ASUD (now) | Geolex (archived) |
|---|---|---|
| units in the graph | **18,387** (the whole lexicon) | 3,302 (an A–C slice) |
| units with passages | 8,094 | 3,302 |
| passages | 22,311 (median 273 chars) | 8,757 (median 545 chars) |
| relationships | ~180,000 | ~50,000 |
| curated relations | **yes** (overlies, intrudes, equivalent to, ...) | no |
| placeholder units | 453 | 3,211 |

## Headline

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| age | 0.00 | **1.00** | **1.00** |
| states | 0.00 | **1.00** | **1.00** |
| parent | 0.50 | **1.00** | **1.00** |
| members | 0.60 | **1.00** | **1.00** |
| relation | 0.38 | 0.88 | **1.00** |
| multi_hop | 0.00 | **0.88** | 0.06 |
| count | 0.00 | **1.00** | 0.00 |
| descriptive | **0.88** | 0.25 | **0.88** |
| **ALL** | 0.29 | **0.88** | 0.74 |
| *Geolex version* | *0.61* | *0.86* | *0.75* |

Context recall (did the retrieved context contain the answer?):

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| age | 0.12 | 1.00 | 1.00 |
| states | 0.25 | 1.00 | 1.00 |
| relation | 0.50 | 0.88 | 1.00 |
| multi_hop | 0.06 | 0.88 | 0.06 |
| count | 0.00 | 1.00 | 0.00 |
| descriptive | 0.88 | 0.00 | 0.75 |
| **ALL** | 0.38 | 0.84 | 0.73 |

| | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| abstain rate | **0.50** | 0.08 | 0.22 |
| median seconds / question | 14.5 | 23.4 | 16.3 |
| median context chars | 3,999 | 320 | 4,457 |

Timings are inflated: the benchmark shared the GPU with geo-sft's training
run. The judge agreed with the string matcher on **97%** of structured
questions (LLM-judged ALL: RAG 0.36, KG 0.88, GraphRAG 0.75). No Cypher query
was still erroring after the repair round.

## What the numbers say

**1. The KG and GraphRAG results replicate; RAG's halves — because of the
text, not the retriever.** RAG scores 0.00 on age and states, against 0.76 and
0.88 on Geolex. Its context recall there is 0.12 and 0.25: the answer is
simply not in ASUD prose. Geolex summaries said *"Age is Late Cretaceous ...
in Texas"*; ASUD reference notes give ages as numbers and places as
geological provinces:

> *What is the geologic age of the Plantagenet Group?* — gold **Eocene**
> RAG: *"The age of the Plantagenet Group is 51.5-36.5 Ma."*

That answer is right in substance and fails the string match (the judge gives
RAG 0.38 on age). The states questions fail the same way: *"located in the
Pilbara Craton"* never says Western Australia. ASUD keeps those facts in its
**tables**, which is exactly what a knowledge graph reads and a text index
does not. The lesson of the lab — *where the facts live decides which system
can answer* — is sharper on this corpus than on the one it was built for.

**2. RAG now abstains instead of confabulating.** On Geolex, RAG counted the
8 passages it retrieved and reported the number confidently ("7 units" when
the answer was 41). Here it mostly says "I don't know based on the retrieved
context" (abstain rate 0.50 vs 0.11). Same model, same prompt: short,
number-heavy reference notes give it less to extrapolate from. It still
miscounts when it has *some* matching passages — *"There are 3 Ordovician
units ... in Tasmania"* (gold: 58).

**3. Curated relations made `relation` a graph category.** Relation gold is
now curated by ASUD, not produced by geo-sft's rule labeller, so it is no
longer the noisy category. RAG drops to 0.38 (context recall 0.50): a unit's
overlying units are usually recorded on *other* units' notes, if at all.
GraphRAG is perfect because its unit card lists them.

**4. GraphRAG still cannot count or filter** (count 0.00, multi_hop 0.06):
its expansion is local. Its one multi-hop success is instructive — it found
the Rocky Cape Group through a passage and read its age and lithology from the
card, but missed the other unit.

**5. The KG's weakness is still prose** (descriptive 0.25; the two it "got"
were lithology and mineral lists the graph happens to hold). And its misses
elsewhere are linking misses: *"Yartoo volcaniclastics"* (with stray quote
marks from ASUD's informal name) and *"quartz arenite"* (not a lithology the
linker knows) returned no rows.

## What the numbers do *not* establish

- **The KG's 0.88 is still partly circular.** Seven categories take gold from
  the curated metadata the graph was built from. `descriptive` is the
  counterweight; an independent test set is experiment #10.
- **Some question subjects are informal ASUD units** — *"Balbirini
  Dolostone, 'lower'"*, *"Yungkulungu Formation - volcanic lithofacies"*, and
  multi-hop gold lists include numbered informal units (*"Mesoproterozoic
  granites 76632"*). They are real ASUD entries, but no geologist would ask
  about them. The subject filter (`bench/build.py: Index.clean`) should
  exclude names with commas, quotes, hyphenated qualifiers or numbers; that
  was found after the run and is not yet fixed.
- **The string matcher is strict about vocabulary.** An answer in Ma, or a
  province for a state, scores 0. The judge column is the fairer read for
  RAG on age.
- **One run with 8 questions per category.**

## The takeaway

The routing conclusion holds, and more strongly: **counts, filters, ages,
states and relations go to the graph; prose details go to text; GraphRAG is
the best single system when you cannot route.** A router (experiment #9)
would score ~1.0 on every category except the few linking misses. And the
corpus itself taught the second lesson: *before choosing a retrieval method,
look at where your facts are stored.* ASUD puts its structure in tables and
its nuance in prose, and each system can only reach one of them.

## Reproducing

```bash
make ingest && make bench-build && make bench     # ingest ~25 min (embedding 22k passages), bench ~70 min
georag bench report runs/<dir>                    # re-render a report
```

Question generation for `descriptive` uses the LLM, so a regenerated question
set will differ. `data/bench/bench-v1.jsonl` is the set these numbers came from.
