PY := .venv/bin/python
CLI := .venv/bin/geocpt
CFG ?= configs/tapt.yaml

.PHONY: help setup doctor corpus pack train eval probe transfer all smoke test clean-ckpt clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

setup:     ## create the venv and install (pulls in ../geo-sft)
	uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"

doctor:    ## environment, disk, geo-sft data, stage status
	$(CLI) doctor

corpus:    ## gather, filter, dedup, mix
	$(CLI) corpus -c $(CFG)

pack:      ## tokenise and pack into fixed blocks
	$(CLI) pack -c $(CFG)

train:     ## continued pre-training
	$(CLI) train -c $(CFG)

eval:      ## domain vs general perplexity (pass CKPT=runs/<name>/checkpoint-NNNNNN)
	$(CLI) eval -c $(CFG) --checkpoint $(CKPT)

probe:     ## cloze probes: knowledge vs style
	$(CLI) probe -c $(CFG) --checkpoint $(CKPT)

transfer:  ## the payoff: SFT on top of the CPT'd model
	$(CLI) transfer -c $(CFG) --checkpoint $(CKPT)

all:       ## corpus -> pack -> train -> eval -> probe
	$(CLI) all -c $(CFG)

smoke:     ## tiny end-to-end run on Qwen3-0.6B
	$(CLI) all -c configs/smoke.yaml

test:      ## unit tests
	.venv/bin/pytest -q

clean-ckpt: ## delete model checkpoints (the big stuff)
	find runs -maxdepth 2 -name "checkpoint-*" -type d -exec rm -rf {} +

clean:     ## delete all generated data and runs
	rm -rf data/corpus data/packed data/interim runs
