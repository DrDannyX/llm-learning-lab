# Benchmark report (64 questions)

## Score (headline)

String-matched against gold for structured categories; LLM-judged for `descriptive`.

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

## Context recall

Did the retrieved context contain the answer at all? Low here = retrieval failure; high here but low score = generation failure.

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| age | 0.82 | **1.00** | **1.00** |
| states | 0.88 | **1.00** | **1.00** |
| parent | 0.75 | **1.00** | **1.00** |
| members | 0.61 | **1.00** | **1.00** |
| relation | 0.88 | **1.00** | **1.00** |
| multi_hop | 0.13 | **1.00** | 0.16 |
| count | 0.25 | **1.00** | 0.12 |
| descriptive | **1.00** | 0.00 | **1.00** |
| **ALL** | 0.66 | **0.88** | 0.79 |

## LLM judge (all questions)

| category | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| age | 0.75 | **1.00** | **1.00** |
| states | 0.75 | **1.00** | **1.00** |
| parent | 0.88 | **1.00** | **1.00** |
| members | 0.50 | **1.00** | **1.00** |
| relation | 0.88 | **1.00** | **1.00** |
| multi_hop | 0.00 | **1.00** | 0.00 |
| count | 0.00 | **1.00** | 0.00 |
| descriptive | **1.00** | 0.00 | **1.00** |
| **ALL** | 0.59 | **0.88** | 0.75 |

Judge agrees with the string matcher on 98% of structured questions.

## Cost

| | Vector RAG | Knowledge graph | GraphRAG |
|---|---|---|---|
| abstain rate | 0.11 | 0.11 | 0.08 |
| median seconds | 8.0 | 4.6 | 6.4 |
| median context chars | 5,489 | 320 | 5,074 |

KG queries that still errored after repair: 0
