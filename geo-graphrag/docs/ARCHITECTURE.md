# Architecture

Lookup, not cover to cover.

## Layout

```
geo-graphrag/
├── docker-compose.yml        Neo4j 5 community (graph + vector index)
├── configs/default.yaml      models, Neo4j, k's, benchmark size
├── src/georag/
│   ├── config.py  paths.py
│   ├── llm.py                Strands model factory + one-shot `complete()`; embeddings over HTTP
│   ├── db.py                 Neo4j driver; `read_untrusted` = read-only tx + timeout
│   ├── ingest/
│   │   ├── extract.py        corpus -> Graph (pure Python, testable)  -> data/interim/graph.json
│   │   └── load.py           Graph -> Neo4j (UNWIND batches) + embeddings + vector index
│   ├── retrievers/
│   │   ├── base.py           Retrieval / Answer dataclasses
│   │   ├── link.py           entity linking: question -> unit keys, intervals, states, lithologies
│   │   ├── rag.py            vector top-k
│   │   ├── kg.py             text-to-Cypher with schema, examples, repair round
│   │   ├── graphrag.py       vector entry -> seed units -> unit cards + graph-routed passages
│   │   └── pipeline.py       ONE answer prompt for all systems; `compare()`
│   ├── bench/
│   │   ├── build.py          8 question generators -> data/bench/<name>.jsonl
│   │   ├── score.py          string match, context recall, LLM judge
│   │   └── run.py            resumable runner -> runs/<name>/{results.jsonl,report.md,report.json}
│   ├── mcp_server.py         8 MCP tools (4 answer, 4 primitive), stdio or HTTP
│   ├── agent.py              Strands agent + MCPClient over stdio
│   └── cli.py                `georag ...`
└── tests/test_lab.py         offline tier + `-m live` tier
```

## Data flow

```
geo-sft/data/interim/passages.jsonl ─┐
geo-sft/data/interim/vocab.json ─────┤
geo-sft/data/raw/asud (snapshot) ────┤
geo-sft Labeller (imported) ─────────┴─► extract.build() ─► graph.json ─► load_graph() ─┐
                                                                                        ├─► Neo4j
                          passages ─► LM Studio /v1/embeddings ─► embeddings.npy ───────┘
```

Nothing is downloaded: the corpus is whatever geo-sft built. `geo-sft` is an
editable path dependency (as in geo-cpt), for its `Labeller`, `Gazetteer` and
`STATES`.

## Where the LLM is called

| call | module | model | per question |
|---|---|---|---|
| answer | `pipeline.answer` | `chat_model` | 1 per system |
| text-to-Cypher | `kg.retrieve` | `chat_model` | 1, or 2 with repair |
| grade | `score.judge` | `judge_model` | 1 per system |
| descriptive question generation | `build.gen_descriptive` | `chat_model` | once, at `bench build` |
| agent loop | `agent.session` | `chat_model` | 2+ |

All go through `llm.complete()` → a fresh Strands `Agent` per call, so no
conversation history leaks between questions. The agent is the only multi-turn
use.

## Graph model

| node | key | from |
|---|---|---|
| `Unit` | `key` = `asud:<stratno>` or `name:<core>` | every current ASUD unit (18,387, `has_text` marks the 8,094 with passages); placeholders for names resolved to nothing |
| `Passage` | `id` = `<stratno>:<n>` | one per geo-sft passage (definition card or reference note); carries `text`, `embedding` |
| `Interval` | `name` | Macrostrat timescale + Precambrian and the unnamed Cambrian series/stages |
| `Lithology`, `Mineral`, `Province` | `name` | curated + rule-extracted |
| `State` | `code` | ASUD jurisdiction codes (NSW, QLD, ..., OFF, ATA), with `name` |

| relationship | from | props |
|---|---|---|
| `PART_OF` Unit→Unit | ASUD parent unit | `sources` |
| `OVERLIES` Unit→Unit | ASUD curated relations **and** rule labeller (overlies/underlies/unconformable_on, canonicalised) | `sources`, `passages`, `unconformable`, `contact` |
| `INTRUDES` Unit→Unit | ASUD curated **and** rule labeller (intrudes/intruded_by, canonicalised) | `sources`, `passages` |
| `EQUIVALENT_TO`, `GRADES_INTO`, `INTERTONGUES_WITH` | ASUD curated **and** rule labeller | `sources`, `passages` |
| `HAS_AGE` Unit→Interval | ASUD oldest/youngest age names, canonicalised | `sources` |
| `WITHIN` Interval→Interval | computed from numeric ages | — |
| `HAS_LITHOLOGY` | ASUD lithology description (tagged with the rock vocabulary) **and** rule labeller | `sources` |
| `HAS_MINERAL` | rule labeller | `sources` |
| `OCCURS_IN`, `IN_PROVINCE` | ASUD | `sources` |
| `DESCRIBES` Passage→Unit | ASUD filing | — |
| `MENTIONS` Passage→Unit | relation targets found in the passage | — |

Indexes: uniqueness constraints on every key; `unit_name` (range);
`unit_names` (fulltext, unused by the pipelines, handy in the browser);
`passage_embedding` (vector, 768-d cosine).

## MCP tools

| tool | tier | returns |
|---|---|---|
| `compare_all(question)` | answer | three answers + evidence (Cypher, source ids) |
| `ask_rag` / `ask_kg` / `ask_graphrag(question)` | answer | one answer + evidence |
| `vector_search(query, k)` | primitive | passages with scores |
| `run_cypher(cypher)` | primitive | ≤50 rows, read-only |
| `graph_schema()` | primitive | schema + example queries |
| `unit_facts(name)` | primitive | linked unit(s)' card and key |

### Using the server from other clients

Claude Desktop / Claude Code / LM Studio (`mcp.json`) all take the same shape:

```json
{
  "mcpServers": {
    "geo-graphrag": {
      "command": "/ABSOLUTE/PATH/geo-graphrag/.venv/bin/python",
      "args": ["-m", "georag.mcp_server"],
      "cwd": "/ABSOLUTE/PATH/geo-graphrag"
    }
  }
}
```

For Claude Code: `claude mcp add geo-graphrag -- /ABS/.venv/bin/python -m georag.mcp_server`.
For HTTP clients: `georag mcp --http` serves on `http://localhost:8765/mcp`.

## Benchmark files

`runs/<name>/results.jsonl`: one row per (question, system): answer, `score`,
`judge` + `judge_reason`, `context_recall`, `abstained`, timings,
`context_chars`, and `debug` (linked entities, Cypher attempts, seed units).
`report.md` / `report.json` are derived from it; `georag bench report <dir>`
regenerates them.

## Operational notes

- **Pin both models in LM Studio** (`lms load <id>`). With JIT loading and
  auto-evict, the embedding model and the chat model swap on *every question*.
  Nothing errors; the benchmark just takes hours. `georag doctor` warns.
- The chat model's context must fit the largest prompt: GraphRAG's ~5.5k chars
  of context is ~1.5k tokens, so 8k is plenty for `ask`/`bench`. The agent
  needs ≥16k.
- Re-ingest is ~15 s once `data/interim/embeddings.npy` exists (keyed on
  passage ids + embedding model; delete it to re-embed).
- `configs/smoke.yaml` loads 150 units into the **same** database. Re-run
  `make ingest` afterwards.
