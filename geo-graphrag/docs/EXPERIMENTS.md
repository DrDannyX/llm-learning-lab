# Experiments

Each one changes **one thing**, reruns the benchmark, and compares reports.
Start a fresh run directory per experiment, e.g. `georag bench run --out runs/exp2-no-prefix`.
The benchmark question set (`data/bench/bench-v1.jsonl`) stays fixed across all of them.
Regenerating it changes the test, not the system.

Ordered by what they teach per hour.

---

### 1. LLM extraction vs rule extraction: *the* KG experiment

**Question:** does an LLM build a better graph than geo-sft's regex labeller?

Replace `labeller.label(r)` in `ingest/extract.py` with a Strands call that
returns the geo-sft `GeoExtraction` schema (Strands `structured_output_model=`
works with LM Studio). ~8,700 calls; with `reasoning_effort: none` expect
2–4 hours unattended. Or point it at the geo-sft **fine-tuned adapter** served
by `mlx_lm.server`, which closes the loop across all three labs: the SFT
model becomes the KG's extractor.

Watch: `resolution` stats, `OVERLIES` edge count, and the `relation` and
`multi_hop` categories. Expect more edges *and* more wrong edges; the
provenance field is how you would later separate them.

### 2. Drop the nomic task prefixes

Set `DOC_PREFIX = QUERY_PREFIX = ""` in `llm.py`, delete
`data/interim/embeddings.npy`, re-ingest. A silent-degradation bug in the
geo-cpt tradition: nothing errors. Watch RAG's `context_recall` on `descriptive`.
Then try removing the contextual header (`f"{unit_name}. "` in
`load.embed_passages`) instead.

### 3. Thinking on for text-to-Cypher

`reasoning_effort: none` → `medium` (or remove it). Every call gets ~8× slower.
Does the KG's score on `multi_hop` and `count` move? Check `kg_query_errors`
and the `attempts` in `results.jsonl`. Thinking helps multi-constraint queries
most; single-hop lookups should not change.

### 4. A different judge

Set `judge_model: google/gemma-4-e4b` (or any other model you have) and run
into a new directory. Compare `judge_agreement` and the `descriptive` row. If the ranking of systems flips when only the judge changes,
the descriptive numbers were measuring the judge.

### 5. k sweep for RAG

`rag_k: 4, 8, 16, 32`. `count` and `multi_hop` will not fix themselves; watch
at what k the answer model starts *losing* things that are in its context
(context_recall up, score flat or down: the "lost in the middle" effect,
measured).

### 6. Global GraphRAG (community summaries)

Microsoft GraphRAG's other half. Neo4j community edition has no GDS, so:
export `Unit`–`Unit` edges to `networkx`, run Louvain/Leiden
(`networkx.community.louvain_communities`), LLM-summarise each community's
unit cards, store `(:Community {summary})<-[:IN_COMMUNITY]-(:Unit)`.
Add questions like "What are the main Pennsylvanian stratigraphic groups of the
mid-continent?". This is the question type **none** of the three current systems
handles well, because the answer is a synthesis, not a lookup.

### 7. Agentic GraphRAG over MCP

Give the Strands agent **only the primitive tools** (`vector_search`,
`run_cypher`, `graph_schema`, `unit_facts`) via
`MCPClient(..., tool_filters={"allowed": [...]})` in `agent.py`, and run the
benchmark questions through it. The agent can route per question, inspect an
empty Cypher result and retry, and combine text and graph. It is the most
capable design here and the slowest: count the tool calls.

### 8. Entity linking off

Pass an empty `Linked()` to the KG and GraphRAG retrievers. The model now has to
match units by name in Cypher. Measure how much of the KG's advantage was really
the deterministic linker, and how many "no rows" results appear.

### 9. A router

A 20-line classifier (LLM or keyword) that sends count/list/filter questions to
the KG and everything else to GraphRAG. If it beats every single system on
`ALL`, you have reproduced the main practical conclusion of the GraphRAG
literature: **the hybrid is a routing problem.**

### 10. An independent test set

Fix the circularity caveat. Write 20 questions by hand, from a geologist's point
of view, with answers checked against Geolex web pages rather than the graph.
Rerun. If the KG's lead shrinks, that is the measured size of the circularity.
