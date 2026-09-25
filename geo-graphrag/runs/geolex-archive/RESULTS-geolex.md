# Results

One run: `runs/bench-v1/` (64 questions × 3 systems, `google/gemma-4-12b` with
thinking off, as both answerer and judge; nomic-embed-text-v1.5; full corpus).
**One run, one seed, 8 questions per category.** A single question moves a
category score by 0.125. Read the shape, not the second decimal.

## Headline

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| age | 0.76 | **1.00** | 0.95 |
| states | 0.88 | **1.00** | **1.00** |
| parent | 0.75 | 0.88 | **1.00** |
| members | 0.58 | **1.00** | **1.00** |
| relation | 0.88 | **1.00** | **1.00** |
| multi_hop | 0.07 | **1.00** | 0.04 |
| count | 0.00 | **1.00** | 0.00 |
| descriptive | **1.00** | 0.00 | **1.00** |
| **ALL** | 0.61 | **0.86** | 0.75 |

Context recall (did the retrieved context contain the answer?):

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| multi_hop | 0.13 | 1.00 | 0.16 |
| count | 0.25 | 1.00 | 0.12 |
| members | 0.61 | 1.00 | 1.00 |
| descriptive | 1.00 | 0.00 | 1.00 |
| **ALL** | 0.66 | 0.88 | 0.79 |

| | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| abstain rate | 0.11 | 0.11 | 0.08 |
| median seconds / question | 8.0 | 4.6 | 6.4 |
| median context chars | 5,489 | 320 | 5,074 |

The judge agreed with the string matcher on **98%** of structured questions.
Text-to-Cypher needed the repair round on 4 of 64 questions, and no query was
still erroring after repair.

## What the numbers say

**1. Each system wins exactly where its design predicts.** The KG is perfect on
everything the schema can express and scores 0 on `descriptive`, where it
abstained 7 of 8 times because it has no prose. RAG and GraphRAG are perfect on
`descriptive` and near zero on `count` and `multi_hop`. No system is best
overall in any sense that survives a change of question mix.

**2. RAG's aggregate failures are retrieval failures, and they are confident.**
On `count`, context recall is 0.25 and score is 0. Only 8 passages come back,
and the model counts *those*: "There are 7 Pennsylvanian units" (true answer
41), "There are 3 Tertiary units ... in Arizona" (40). It rarely says it cannot
count. A bigger k does not fix a question whose answer is spread over 100
passages (experiment #5).

**3. GraphRAG's gain over RAG is on entity-centred questions:** `members` 0.58 →
1.00, `parent` 0.75 → 1.00, `age` 0.76 → 0.95. That is the unit card and the
graph-routed passages at work. Its expansion is *local*, so on `multi_hop` and
`count` it is as blind as RAG (0.04 and 0.00). Fixed Cypher around seed units
cannot answer a question about all units.

**4. The KG is the cheapest system.** 320 characters of context against ~5,000,
and the fastest median time despite two LLM calls. Exact retrieval is compact
retrieval.

**5. A string matcher caught what the judge missed.** The one KG `parent` miss:
the graph returned "Schnebly Hill Formation", and the answer model wrote
"Schnebley Hill Formation", "correcting" the spelling from its own prior. The
judge called it correct. It *is* a small faithfulness failure (the answer does
not match its evidence), and only the deterministic check saw it.

## What the numbers do *not* establish

- **Anything about US geology as a whole.** geo-sft fetched the first 4,000
  entries of the Geolex index, which is alphabetical, so the corpus holds only
  units named A to C ("A-L Peak" to "Cross Creek"): 3,302 of 16,684 units, or
  20%. "41 Pennsylvanian units in Kansas" means 41 *A–C* units. The comparison
  between systems is unaffected, because all three see the same slice.

- **That the KG is "86% accurate".** Seven of eight categories take gold answers
  from the curated metadata the graph was built from; there, the KG reads back
  its own contents. `multi_hop` and `count` at 1.00 mean "text-to-Cypher reliably
  expresses these templates", not "the graph is complete". The KG *missed* Gober
  Chalk and Ector Chalk on "Cretaceous chalk units in Texas": they are not Geolex
  units in this corpus, only placeholder nodes (`geolex: false`) with no age or
  lithology edges. GraphRAG found them in prose. The benchmark cannot see
  that, because its gold came from the same graph. Experiment #10 fixes this.
- **That RAG is good at descriptive questions in general.** Every descriptive
  question names its unit and was generated from one passage, so retrieving
  that passage is easy (context recall 1.00). Questions phrased the way a
  geologist would ask, without the unit's name, would be much harder.
- **Anything about the judge on its own.** 98% agreement is measured on
  structured questions. On `descriptive` it is the only grader, and it is the
  same model that answered.
- **Robustness to question wording.** Structured questions come from eight
  templates, and the text-to-Cypher prompt carries examples close to them. On
  free-form questions, expect more repair rounds and more empty results.

## The takeaway

The main practical conclusion from GraphRAG work in the field holds up here:
**the hybrid is a routing problem.** A router that sends count/list/filter
questions to the KG and everything else to GraphRAG would score ~1.0 on every
category in this benchmark (experiment #9). Building that router, or letting an
agent do the routing over MCP (#7), is the natural next step.

## Reproducing

```bash
make ingest && make bench-build && make bench     # ~40 min with both models pinned
georag bench report runs/<dir>                    # re-render a report
```

Question generation for `descriptive` uses the LLM, so a regenerated question
set will differ. `data/bench/bench-v1.jsonl` is the set these numbers came from.
