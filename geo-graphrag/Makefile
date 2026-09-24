CLI := .venv/bin/georag
CFG ?= configs/default.yaml

.PHONY: help setup up down doctor ingest smoke ask inspect bench-build bench bench-report mcp agent web test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

setup:        ## create the venv and install everything
	uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"

up:           ## start Neo4j (Docker); browser at http://localhost:7474
	docker compose up -d --wait

down:         ## stop Neo4j (data is kept in data/neo4j)
	docker compose down

doctor:       ## check Neo4j, LM Studio, corpus and loaded graph
	$(CLI) doctor -c $(CFG)

ingest:       ## corpus -> graph + embeddings -> Neo4j (~3 min)
	$(CLI) ingest -c $(CFG)

smoke:        ## tiny end-to-end run: 150 units, 16 questions
	$(CLI) ingest -c configs/smoke.yaml && $(CLI) bench build -c configs/smoke.yaml && \
	$(CLI) bench run -c configs/smoke.yaml --out runs/smoke

ask:          ## three answers side by side: make ask Q="..."
	$(CLI) ask "$(Q)" -c $(CFG)

inspect:      ## what each system retrieved: make inspect Q="..."
	$(CLI) inspect "$(Q)" -c $(CFG)

bench-build:  ## generate the question set (64 questions)
	$(CLI) bench build -c $(CFG)

bench:        ## run the benchmark (~30-60 min on a 12B model)
	$(CLI) bench run -c $(CFG)

bench-report: ## re-render a report: make bench-report RUN=runs/<dir>
	$(CLI) bench report $(RUN)

mcp:          ## serve the three systems over MCP (stdio)
	$(CLI) mcp

web:          ## browser UI, three answers side by side: http://localhost:8000
	$(CLI) web

agent:        ## chat with a Strands agent that uses the MCP server
	$(CLI) agent -c $(CFG)

test:         ## unit tests (no services needed)
	.venv/bin/pytest -q -m "not live"

lint:
	.venv/bin/ruff check src tests

clean:        ## delete generated data (keeps the embedding cache)
	rm -rf data/interim/graph.json data/bench
