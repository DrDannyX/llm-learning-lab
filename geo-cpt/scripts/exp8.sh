#!/usr/bin/env bash
# EXPERIMENT 8: vocabulary extension + CPT, then SFT, vs the stock-vocab arm.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/Users/daniel/git/llm-scratchpad/geo-sft/artifacts/hf-cache
export HF_HUB_DISABLE_XET=1
CFG=configs/tapt-vocabext.yaml
CLI=.venv/bin/geocpt

echo "### 1/4 corpus (same text, tokenised by the EXTENDED tokenizer)"
$CLI corpus -c $CFG 2>&1 | grep -viE "^\s*$|━|warning" | tail -12
echo; echo "### 2/4 pack"
$CLI pack -c $CFG 2>&1 | grep -viE "^\s*$|━|warning|Fetching" | tail -14
echo; echo "### 3/4 CPT full fine-tune (trains the grafted embedding rows)"
$CLI train -c $CFG --name tapt-vocabext 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|warning|Fetching" | tail -20
echo; echo "### 4/4 transfer: SFT on top, vs stock-vocab 0.842"
CKPT=$(ls -d runs/tapt-vocabext/checkpoint-* 2>/dev/null | sort | tail -1)
$CLI transfer -c $CFG --checkpoint "$CKPT" --name sft-after-tapt-vocabext 2>&1 \
  | tr '\r' '\n' | grep -viE "^\s*$|warning|Fetching|generate " | tail -25
echo; echo "### experiment 8 complete"
