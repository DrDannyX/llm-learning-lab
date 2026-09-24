# Knowledge graphs, RAG and GraphRAG — the concepts

Read §1–3 before your first `make ingest`, §4–6 before your first `georag ask`,
and §7–9 before you trust a benchmark number. Everything else is reference.

---

## 1. Three ways to give a model knowledge it was not trained on

All three systems answer questions from the same USGS Geolex lexicon (3,302
geologic units, 8,757 passages) with the same local model. That is the
alphabetical slice A–C of Geolex's 16,684 units, which is why examples use the
Aarde Shale Member and the Chickamauga Group rather than famous units like the
Eagle Ford. None of them changes
the model's weights — that was geo-sft and geo-cpt. They change **what the model
is shown** at question time.

```
                        ┌───────────────────── Neo4j ─────────────────────┐
question ─┬─ embed ─────┤ vector index ──► top-k Passages                  │──► context ─┐
          │             │                                                  │             │
 Vector   │             │                                                  │             │
 RAG      │             │                                                  │             │
          │             │                                                  │             │
 KG     ──┼─ link ──► LLM writes Cypher ──► (:Unit)-[:OVERLIES]->(:Unit)  │──► rows ────┤──► same LLM,
          │             │                                                  │             │    same prompt
 GraphRAG ┴─ embed + link ─► Passages ──DESCRIBES──► Units ──► neighbours  │──► both ────┘    ──► answer
                        └──────────────────────────────────────────────────┘
```

| | Vector RAG | Knowledge graph | GraphRAG (local) |
|---|---|---|---|
| stores | text chunks + vectors | entities + typed relations | both, linked |
| retrieval | nearest neighbours | a query language (Cypher) | vectors, then graph expansion |
| who writes the query | nobody — it is a similarity | the LLM | a human, once |
| sees prose | yes | **no** | yes |
| can count / filter / traverse | **no** | yes | partly (fixed neighbourhood) |
| signature failure | retrieves the wrong 8 passages | a query that silently returns nothing | expands around the wrong unit |
| build cost | embed everything | **extraction + entity resolution** | both |

The one-sentence version: **RAG retrieves what *sounds like* the question; a KG
retrieves what the question *means*, if you can say it in the schema; GraphRAG
uses the first to find a way into the second.**

---

## 2. What a knowledge graph is

A **property graph** has nodes (with labels and properties) and directed,
typed relationships (which can also carry properties):

```
(:Unit {name: "Aarde", rank: "Member"}) -[:PART_OF]-> (:Unit {name: "Howard", rank: "Formation"})
(:Unit {name: "Church"}) -[:OVERLIES]-> (:Unit {name: "Aarde"})
(:Unit {name: "Aarde"}) -[:HAS_AGE]-> (:Interval {name: "Virgilian"}) -[:WITHIN]-> (:Interval {name: "Pennsylvanian"})
```

(The alternative, RDF triples with ontologies and SPARQL, is more formal and
common in linked-data and government geoscience, e.g. GeoSciML. Property graphs
are simpler and are what most GraphRAG tooling uses.)

The graph's value is in the **edges**. A table can hold "Aarde's age is
Virgilian". Only a graph makes "every unit whose age is *anywhere inside* the
Pennsylvanian, in Kansas, that sits above a limestone" a single query.

### This lab's schema

```
(:Passage)-[:DESCRIBES]->(:Unit)<-[:PART_OF]-(:Unit)
(:Passage)-[:MENTIONS]->(:Unit)
(:Unit)-[:OVERLIES]->(:Unit)           (:Unit)-[:EQUIVALENT_TO|GRADES_INTO|INTERTONGUES_WITH]-(:Unit)
(:Unit)-[:HAS_AGE]->(:Interval)-[:WITHIN]->(:Interval)
(:Unit)-[:HAS_LITHOLOGY]->(:Lithology)     (:Unit)-[:HAS_MINERAL]->(:Mineral)
(:Unit)-[:OCCURS_IN]->(:State)             (:Unit)-[:IN_PROVINCE]->(:Province)
```

Open http://localhost:7474 after ingest and run
`MATCH p=(:Unit {name:'Aarde'})-[*1..2]-() RETURN p LIMIT 60` to see it.

---

## 3. Building a knowledge graph — where the work actually is

Retrieval gets the attention, but **a KG is only as good as its extraction**, and
extraction is three separate problems. `src/georag/ingest/extract.py` does all
three; read it alongside this section.

### 3.1 Extraction: where do the facts come from?

| source | used for | trust |
|---|---|---|
| **curated metadata** (Geolex usages, ages, states, provinces) | hierarchy, ages, states | high — a person entered it |
| **rules** (geo-sft's labeller over passage text) | relations, lithology, minerals, thickness | medium — geo-sft found errors in 83% of reviewed rows |
| **an LLM** reading each passage | *not used here* — experiment #1 | variable, and slow: ~9k calls |

Every edge records its `sources`. That provenance is not decoration: it is how
you later decide which facts a query may rely on. **Ages come only from curated
metadata**, because the rule labeller tags every interval a passage mentions,
including the ages of *neighbouring* units, and that noise would poison exactly
the "which units are Cretaceous?" queries a graph is for.

### 3.2 Entity resolution: are these the same thing?

Prose says "Eagle Ford Clay", "Eagle Ford shales" and "the Eagle Ford". Geolex
calls the unit "Eagle Ford". If those become four nodes, the graph is four
disconnected fragments and every traversal is wrong.

This lab reduces every name to a **core** (`core_name`: strip trailing rank and
lithology words) and matches on that. Then it hits **homonyms**: 289 names in
this corpus belong to more than one unit (two Adas, two Aberdeens). Ties are
broken by shared states. The ingest reports the outcome; look at it:

```
resolved              1,504    exact unique match to a Geolex unit
ambiguous_by_state      216    homonym, state overlap picked one
ambiguous_guess          89    homonym, no signal -- a coin flip, recorded as such
placeholder_created   3,211    no Geolex unit in this corpus: new Unit {geolex: false}
```

Placeholders matter. Most relation targets ("overlies the Wauneta Limestone")
name units outside our 3,302. Dropping them would drop the relation; keeping
them as `geolex: false` nodes keeps it and marks it as less-known.

### 3.3 Canonicalisation: one fact, one shape

"A underlies B" and "B overlies A" are the same fact. The graph stores only
`(B)-[:OVERLIES]->(A)`. **There is no UNDERLIES relationship.** One shape per
fact means one way to query it, which matters enormously when a 12B model is
writing the query. The same logic applies to the computed timescale: intervals
carry numeric ages, so Virgilian is linked `-[:WITHIN]->` Pennsylvanian by
containment, and `HAS_AGE/WITHIN*0..6` finds a unit tagged only "Cenomanian"
when you ask for "Cretaceous".

> **The key idea.** In RAG, most of the intelligence is at query time. In a KG,
> most of it is at **build** time: every decision above is a question you
> answered once, for every future query. It is why KGs are expensive to build
> and cheap and exact to query.

---

## 4. Querying a knowledge graph: Cypher and text-to-Cypher

Cypher is ASCII-art pattern matching:

```cypher
// what overlies the Aarde Shale Member?
MATCH (above:Unit)-[:OVERLIES]->(u:Unit {key: 'geolex:6304'})
RETURN above.full_name

// how many Pennsylvanian units in Kansas? -- no vector system can answer this
MATCH (u:Unit)-[:HAS_AGE]->(:Interval)-[:WITHIN*0..6]->(:Interval {name: 'Pennsylvanian'})
MATCH (u)-[:OCCURS_IN]->(:State {code: 'KS'})
RETURN count(DISTINCT u)
```

The KG system has the LLM write that query (`retrievers/kg.py`). Three things
make it work with a small local model, and each is a general lesson:

1. **A hand-written schema, not an auto-dump.** Neo4j can introspect its own
   schema, but that dump includes every property and no guidance. The prompt
   carries a compact schema plus the rules that bite ("there is no UNDERLIES";
   "use WITHIN*0..6 for ages") and five worked examples.
2. **Entity linking first.** The model is never asked to guess that "the Eagle
   Ford Shale" is `{name: 'Eagle Ford'}`, or which Ada you meant. `link.py`
   resolves names to keys deterministically, and the prompt says *use these keys*.
3. **One repair round.** On an error, *or an empty result*, the model sees what
   happened and tries again. Empty results are the dangerous case: a wrong
   query and a true "none" look identical.

And one that is easy to miss: **show the answer model the query.** Rows alone
(`{"overlying_unit": "Church Member"}`) do not say *what* they overlie; the
first version of this lab answered "I don't know" from correct rows because
of it (`kg.format_rows`).

**Safety:** model-written queries run in a read-only transaction with a timeout
(`db.read_untrusted`). The regex blocklist is belt-and-braces; the read-only
session is the actual guarantee.

---

## 5. RAG: chunks, embeddings, nearest neighbours

`retrievers/rag.py` is 40 lines, which is the appeal:

1. **Chunk.** Geolex passages are already paragraph-sized (median 545 chars), so
   each passage is one chunk. On long documents, chunking strategy is most of
   the RAG work; here it is a non-issue, which keeps the comparison clean.
2. **Embed.** `nomic-embed-text` via LM Studio, 768-d. Two details that silently
   cost quality if missed:
   - **Task prefixes.** nomic was trained with `search_document:` on passages and
     `search_query:` on queries. Omit them and retrieval degrades with no error
     (experiment #2).
   - **Contextual headers.** Many passages never name their unit ("Consists of
     erratic development of sandstones..."). Each is embedded as
     `"<unit name>. <passage>"`, so the vector knows what the passage is about.
3. **Index.** A Neo4j vector index (HNSW, cosine) on `Passage.embedding`. A
   vector index is just another index; it does not need its own database.
4. **Retrieve k=8, stuff into the prompt.**

**What RAG cannot do, structurally:** anything whose answer is spread over more
passages than k. "How many Pennsylvanian units are in Kansas?" has 41 answers
across ~100 passages. RAG retrieves 8, and the model then *counts the 8* and
reports it confidently. That is not a bug to be fixed with a bigger k; it is
the shape of the method.

---

## 6. GraphRAG: a family, not a technique

"GraphRAG" names at least three different designs:

| variant | idea | here |
|---|---|---|
| **local search / graph-expanded retrieval** | find entry points by vector search, expand around them in the graph | **implemented** (`retrievers/graphrag.py`) |
| **global search / community summaries** (Microsoft GraphRAG) | cluster the graph into communities, LLM-summarise each, answer "what are the main themes?" from the summaries | experiment #6 |
| **agentic / hybrid** | an agent chooses per question: vector search, Cypher, or both | the MCP agent with primitive tools, experiment #7 |

The local-search pipeline here:

1. vector search, top-5 entry passages;
2. entity linking on the question;
3. **seed units** = linked units first, then the units the entry passages describe;
4. a fixed, human-written Cypher **unit card** for each seed: ages *and their
   parent intervals*, lithologies, states, hierarchy up and down, neighbours above
   and below;
5. **graph-routed text:** for each linked unit that vector search missed, fetch
   its best passages *through the graph* (`DESCRIBES`), ranked by similarity.

Step 5 is what plain RAG cannot do: the graph routes text retrieval to the right
entity even when the question's wording is far from the passage's. Step 4's
Cypher is fixed, so it never fails to parse, but it also cannot count or filter
across the whole graph. **That trade-off is the most useful thing to watch in
the benchmark.**

---

## 7. A controlled comparison

`retrievers/pipeline.py` sends every system's context to the **same model with
the same prompt** (`ANSWER_SYSTEM`). There is a test that checks this. If you
tune the prompt for one system, the comparison stops meaning anything. The
same principle ran through both earlier labs: geo-cpt reused geo-sft's trainer
and scorer so that "did CPT help?" was measured through identical machinery.

Change one thing at a time: retrieval (k, prefixes, headers), graph (schema,
extraction source), or the shared answer model, never two at once.

---

## 8. Measuring it

There is no single standard metric for "which retrieval system is better", so
the benchmark (`bench/`) reports several, per category.

**Eight categories**, chosen to pull the systems apart (see
`bench/build.py` for the table). Averaging them into one number hides exactly
the intuition you are here to build. Read the per-category rows.

**Four numbers per answer:**

| metric | how | answers |
|---|---|---|
| `score` | string-match against gold (structured), LLM judge (descriptive) | was the answer right? |
| `context_recall` | the same matcher, run on the retrieved **context** | could it have been right? |
| `judge` | LLM grader on every question | cross-check on the string matcher |
| `abstained` | "I don't know..." | did it know that it didn't know? |

**`context_recall` is the most useful diagnostic here.** Low score with low
context recall is a *retrieval* failure (the answer model never saw it). Low
score with high context recall is a *generation* failure (it was there and was
missed). The two have completely different fixes. This is the
retrieval/generation split that frameworks like RAGAS formalise as "context
recall" and "faithfulness".

**Abstention is a feature.** A system that says "I don't know" when the context
lacks the answer is more useful than one that confabulates from the 8 passages
it has. Compare abstain rates against scores.

### The circularity caveat: read before quoting any number

Seven of eight categories take their gold answers **from the same curated
metadata the knowledge graph was built from.** On those, the KG is reading back
its own contents. It *will* look better than it would against an independent
test. `descriptive` (answers exist only in prose) is the counterweight, and
`relation` gold comes from the rule labeller geo-sft found to be noisy.

This is geo-sft's gold-review lesson from the other side: **whoever builds the
test set decides what "better" means.** A fair headline is "the KG wins on
questions the schema can express, and cannot answer the rest", not "the KG
scores 0.8".

---

## 9. LLM-as-judge

`descriptive` answers have no string gold, so an LLM grades them against a
one-passage reference. Three caveats:

- **Same model, grading itself.** By default the judge is the answering model.
  Self-preference bias is well documented. `llm.judge_model` lets you use a
  different model (experiment #4).
- **Calibrate it.** The report prints how often the judge agrees with the
  deterministic string matcher on structured questions. That agreement is your
  best estimate of how far to trust it on the category where it is the only grader.
- **One reference, many truths.** A passage says "named for Butte Falls"; another
  passage about the same unit may say something else true. That is why the
  generator is told to avoid thickness and age, which vary between reports.

---

## 10. MCP and the agent

**MCP (Model Context Protocol)** is a standard way to expose tools to any LLM
client. `mcp_server.py` exposes two tiers:

- **answer tools:** `compare_all`, `ask_rag`, `ask_kg`, `ask_graphrag` run a
  full pipeline;
- **primitive tools:** `vector_search`, `run_cypher`, `graph_schema`,
  `unit_facts` are the raw retrieval steps.

An agent with only the primitives *is* an agentic GraphRAG system: it decides
per question whether to search text, query the graph, or both, and can inspect
a failure and retry. That is the direction the field is moving (experiment #7).

**Strands** runs the agent loop (`agent.py`): model → tool call → tool result →
model, until it answers. Strands starts the MCP server as a subprocess over
stdio and gets its tools from it; the agent never imports the retrievers. That
separation is the point of MCP: the same server works unchanged from Claude
Desktop, Claude Code, or LM Studio's own MCP client.

**A practical constraint:** each tool result stays in the conversation. A
`compare_all` result is ~1–2k tokens, and eight tool schemas are another ~1.5k.
An 8k context holds a couple of turns. Load the chat model with ≥16k context
for the agent (`georag doctor` warns you).
