#!/usr/bin/env bash
# Wait for the in-flight CPT run, then evaluate it end to end.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/Users/daniel/git/llm-scratchpad/geo-sft/artifacts/hf-cache
export HF_HUB_DISABLE_XET=1
CLI=.venv/bin/geocpt
CFG=configs/tapt.yaml
RUN=runs/tapt-v1

echo "### waiting for CPT training to finish"
while pgrep -f "geocpt train" >/dev/null; do sleep 30; done

CKPT=$(ls -d $RUN/checkpoint-* 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then echo "!!! no checkpoint produced"; exit 1; fi
echo "### checkpoint: $CKPT"

echo; echo "### 1/3 perplexity: did it learn, what did it forget?"
$CLI eval -c $CFG --checkpoint "$CKPT" 2>&1 | grep -viE "^\s*$|warning|Fetching"

echo; echo "### 2/3 cloze probes: knowledge vs style"
$CLI probe -c $CFG --checkpoint "$CKPT" 2>&1 | grep -viE "^\s*$|warning|Fetching"

echo; echo "### 3/3 transfer: SFT on top, vs geo-sft's macro F1 0.831"
$CLI transfer -c $CFG --checkpoint "$CKPT" --name sft-after-tapt-v1 2>&1 \
  | tr '\r' '\n' | grep -viE "^\s*$|warning|Fetching"

echo; echo "### chain complete"
