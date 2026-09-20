#!/usr/bin/env bash
# The two control arms the transfer result needs to be interpretable.
#
# A: Qwen3-1.7B, NO CPT, NO SFT, zero-shot extraction.
#    Isolates how much instruction-following CPT destroyed. The CPT'd model
#    scored macro F1 0.002 / parse 0.015 -- but we never measured what the
#    same model scored BEFORE CPT, so we cannot attribute the loss.
#
# B: Qwen3-1.7B, NO CPT, WITH SFT.
#    Isolates TAPT's downstream contribution. The transfer run compared
#    1.7B+TAPT+SFT (0.842) against 4B+SFT (0.831) -- size and CPT varied
#    together, so that comparison measures nothing on its own.
set -uo pipefail
cd /Users/daniel/git/llm-scratchpad/geo-sft
export HF_HOME=$PWD/artifacts/hf-cache
export HF_HUB_DISABLE_XET=1
BASE=mlx-community/Qwen3-1.7B-4bit   # same weights, same quantisation as the CPT model

echo "### CONTROL A: 1.7B zero-shot extraction (no CPT, no SFT)"
.venv/bin/geosft eval -c configs/default.yaml --model $BASE --n 200 2>&1 \
  | tr '\r' '\n' | grep -viE "^\s*$|warning|Fetching|generate "

echo; echo "### CONTROL B: 1.7B + SFT (no CPT)"
.venv/bin/geosft train -c configs/default.yaml --model $BASE --name sft-1.7b-nocpt 2>&1 \
  | tr '\r' '\n' | grep -viE "^\s*$" | tail -12
.venv/bin/geosft eval -c configs/default.yaml --adapter runs/sft-1.7b-nocpt \
  --model $BASE --n 200 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|warning|Fetching|generate "

echo; echo "### controls complete"
