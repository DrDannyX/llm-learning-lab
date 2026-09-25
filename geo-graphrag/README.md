# geo-graphrag

A hands-on lab comparing three ways of answering questions from a document
collection: a **knowledge graph**, **vector RAG** and **GraphRAG**. All three
run over the same geoscience corpus as [geo-sft](../geo-sft/) and
[geo-cpt](../geo-cpt/), on one Mac, with a local LLM.

Ask one question and get three answers side by side:

```
$ georag ask "How many Ordovician units occur in Tasmania?"

 Vector RAG                    Knowledge graph                GraphRAG
 There are 3 Ordovician        There are 58 Ordovician        There are 3 Ordovician units
 units mentioned in the        units that occur in            identified in the context
 context for Tasmania ...      Tasmania.                      for Tasmania ...
```

Only one of those is right, and it is not always the same system. Building an
intuition for *which one, and why* is what this lab is for.

| | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| retrieves | 8 most similar passages | rows from an LLM-written Cypher query | similar passages + graph facts around the units involved |
| sees prose | yes | no | yes |
| can count, filter, traverse | no | yes | partly |
| fails by | retrieving the wrong 8 | a query that silently returns nothing | expanding around the wrong unit |

## Stack

| | |
|---|---|
| LLM | local, via LM Studio at `localhost:1234` (default `google/gemma-4-12b`) |
| embeddings | `nomic-embed-text-v1.5` via LM Studio |
| LLM integration | [Strands Agents](https://strandsagents.com/): every LLM call and the agent loop |
| graph + vectors | Neo4j 5 (Docker): property graph **and** vector index in one store |
| tool interface | MCP server (8 tools) + a Strands agent that uses it |
| corpus | Geoscience Australia's ASUD: all 18,387 units, 22,311 passages about 8,094 of them, read from `../geo-sft/data` |

## Quick start

Needs Docker, [uv](https://docs.astral.sh/uv/), LM Studio, and geo-sft's corpus
already built (`cd ../geo-sft && make fetch`).

```bash
# in LM Studio (or with its `lms` CLI): pin BOTH models; see "Gotchas"
lms load google/gemma-4-12b --context-length 16384
lms load text-embedding-nomic-embed-text-v1.5

make setup          # venv + deps (pulls geo-sft in as an editable dependency)
make up             # Neo4j in Docker; browser at http://localhost:7474 (neo4j / geograph-lab)
make doctor         # checks everything
make ingest         # corpus -> graph + embeddings -> Neo4j (~3 min first time, 15 s after)

make ask Q="Which unit overlies the Alsace Quartzite?"
make inspect Q="Which Cretaceous units in Texas contain chalk?"
make agent          # chat with a Strands agent that uses all three via MCP
make web            # the same comparison in a browser: http://localhost:8000
```

`make web` serves a page with one question box and three columns. The systems
run in parallel, and each column has an **Evidence** panel showing exactly what
that system retrieved (passages, Cypher attempts, units expanded in the graph).
A comparison can be linked: `http://localhost:8000/?q=How many ...`.

The Neo4j browser is at http://localhost:7474. Log in with user `neo4j`,
password `geograph-lab` (set in `docker-compose.yml`).

The benchmark:

```bash
make bench-build    # 64 questions, 8 categories
make bench          # ~40 min; writes runs/<name>/report.md
```

## How to learn from this repo

As in the other labs, **do not read the docs front to back.** Run things, and
read the section that explains what you just saw.

| session | read ([LEARNING.md](docs/LEARNING.md)) | run |
|---|---|---|
| 1. building a graph | §1–3: the three systems; what a KG is; extraction, entity resolution, canonicalisation | `make ingest`, then read the `resolution` stats it prints. Explore in the Neo4j browser. |
| 2. querying it | §4: Cypher, text-to-Cypher, entity linking | `make inspect Q=...` on a count question, then a "what overlies X" question. Read the Cypher attempts. |
| 3. RAG and GraphRAG | §5–6: embeddings, prefixes, headers; the GraphRAG family | `make ask` on the three example questions below. Predict each system's answer first. |
| 4. measuring | §7–9: the controlled comparison; score vs context recall; the circularity caveat; LLM judges | `make bench`, then read [RESULTS.md](docs/RESULTS.md) against your own report |
| 5. MCP and agents | §10 | `make agent`; then [experiment #7](docs/EXPERIMENTS.md), primitives only |
| 6. your own experiment | [EXPERIMENTS.md](docs/EXPERIMENTS.md) | #1 (LLM extraction) is the most instructive; #2 is the quickest |

**The single most useful screen is `georag inspect`.** It shows the exact context
each system hands the answer model. Most wrong answers are decided there, before
any text is generated.

Three questions that separate the systems:

```bash
make ask Q="How many Pennsylvanian units occur in Kansas?"          # only the KG can count
make ask Q="What trace fossils occur in the Tumblagooda Sandstone?"  # the KG has no prose
make ask Q="Which Cretaceous units in Texas contain chalk?"         # complete vs. what's written
```

## Results

See **[docs/RESULTS.md](docs/RESULTS.md)** for the full report and what it does
and does not establish. Overall: **RAG 0.29, knowledge graph 0.88, GraphRAG
0.74** (64 questions). Each system wins the categories its design predicts,
and **context recall** separates retrieval failures from generation failures.
RAG scores lower than on the Geolex version of this lab (0.61) because ASUD
prose rarely states a unit's age or state in words — those facts live in
ASUD's tables, which only the graph reads.

## Gotchas

- **The graph holds the whole lexicon; the text does not.** All 18,387 current
  ASUD units are nodes, but only 8,094 have passages. Counts and filters range
  over every unit, so RAG can never see most of what a count question asks
  about. That is by design, and it is the lesson. The web UI's *About the
  data* panel shows the full picture.

- **Pin both models in LM Studio.** With JIT loading and auto-evict, the
  embedding model and the chat model evict each other on *every question*. Nothing
  errors; everything takes 5× longer. `make doctor` checks.
- **Context length.** `ask` and `bench` fit in 8k tokens. The agent accumulates
  tool results and needs ≥16k.
- **`make smoke` loads 150 units into the same database.** Run `make ingest`
  afterwards.
- **Gemma 4 thinks by default.** `reasoning_effort: none` in the config turns it
  off (~8× faster). Turning it back on is experiment #3.

## Docs

| | |
|---|---|
| [LEARNING.md](docs/LEARNING.md) | concepts: the only part worth reading before running |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | lookup: modules, graph model, MCP tools, client configs |
| [RESULTS.md](docs/RESULTS.md) | what was measured, and what the numbers do *not* establish |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | ten things to change, one at a time |
