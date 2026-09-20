#!/usr/bin/env bash
# Runs once the transfer stage finishes: corrected forgetting measurement.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/Users/daniel/git/llm-scratchpad/geo-sft/artifacts/hf-cache
export HF_HUB_DISABLE_XET=1

echo "### waiting for the transfer stage to finish"
while pgrep -f "run_chain.sh" >/dev/null; do sleep 60; done

CKPT=$(ls -d runs/tapt-v1/checkpoint-* 2>/dev/null | sort | tail -1)
echo "### re-measuring forgetting for $CKPT"
.venv/bin/python scripts/refresh_forgetting.py configs/tapt.yaml "$CKPT" 2>&1 \
  | grep -viE "^\s*$|warning|Fetching"
echo "### forgetting refresh complete"
