#!/usr/bin/env bash
# Arm 1 main recipe, seed 17: train, then the primary evaluation on the plain set.
source "$(dirname "$0")/fix_common.sh"
data
train "$F/data/antidef-all.jsonl" "$F/qwen14b-antidef-s17" 17 2
cross "$Q14" "$Q14R" "$F/plain-qwen14b-antidef-s17" "$PLAIN" reviewer "$F/qwen14b-antidef-s17"
echo "=== arm1 main done ==="
