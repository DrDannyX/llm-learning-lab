PY := .venv/bin/python
CLI := .venv/bin/geosft
CFG ?= configs/default.yaml

.PHONY: help setup doctor fetch build tokenizer extend train eval all smoke test clean-models clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:      ## create the venv and install everything
	uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"

doctor:     ## check hardware, libraries, disk and data
	$(CLI) doctor

fetch:      ## download Geolex corpus + Macrostrat gazetteers
	$(CLI) fetch -c $(CFG)

build:      ## label, split and write train/valid/test/gold
	$(CLI) build -c $(CFG)

tokenizer:  ## train the geoscience tokenizer + fertility report
	$(CLI) tokenizer -c $(CFG)

extend:     ## graft domain tokens onto the base model, then quantise
	$(CLI) extend -c $(CFG)

train:      ## QLoRA fine-tune (MLX)
	$(CLI) train -c $(CFG)

eval:       ## score base vs tuned (pass ADAPTER=runs/<name>)
	$(CLI) eval -c $(CFG) --adapter $(ADAPTER)

all:        ## the whole workflow end to end
	$(CLI) all -c $(CFG)

smoke:      ## tiny end-to-end run, proves the wiring in minutes
	$(CLI) all -c configs/smoke.yaml --skip-extend

test:       ## unit tests
	.venv/bin/pytest -q

clean-models: ## delete downloaded/quantised weights (the big stuff)
	rm -rf artifacts/hf-cache artifacts/models

clean:      ## delete everything generated except the HTTP cache
	rm -rf data/processed data/gold data/interim runs artifacts/tokenizers
