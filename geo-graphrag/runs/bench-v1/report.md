# Benchmark report (64 questions)

## Score (headline)

String-matched against gold for structured categories; LLM-judged for `descriptive`.

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

## Context recall

Did the retrieved context contain the answer at all? Low here = retrieval failure; high here but low score = generation failure.

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| age | 0.12 | **1.00** | **1.00** |
| states | 0.25 | **1.00** | **1.00** |
| parent | 0.50 | **1.00** | **1.00** |
| members | 0.69 | **1.00** | **1.00** |
| relation | 0.50 | 0.88 | **1.00** |
| multi_hop | 0.06 | **0.88** | 0.06 |
| count | 0.00 | **1.00** | 0.00 |
| descriptive | **0.88** | 0.00 | 0.75 |
| **ALL** | 0.38 | **0.84** | 0.73 |

## LLM judge (all questions)

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| age | 0.38 | **1.00** | **1.00** |
| states | 0.00 | **1.00** | **1.00** |
| parent | 0.50 | **1.00** | **1.00** |
| members | 0.75 | **1.00** | **1.00** |
| relation | 0.38 | 0.88 | **1.00** |
| multi_hop | 0.00 | **0.88** | 0.12 |
| count | 0.00 | **1.00** | 0.00 |
| descriptive | **0.88** | 0.25 | **0.88** |
| **ALL** | 0.36 | **0.88** | 0.75 |

Judge agrees with the string matcher on 97% of structured questions.

## Cost

| | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| abstain rate | 0.50 | 0.08 | 0.22 |
| median seconds | 14.5 | 23.4 | 16.3 |
| median context chars | 3,999 | 320 | 4,457 |

KG queries that still errored after repair: 0
