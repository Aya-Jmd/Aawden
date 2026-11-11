#!/usr/bin/env bash
# Arm 1 controls: neutral-only adapter at the SAME number of gradient steps (268 rows x 6 epochs =
# 804 rows x 2 epochs), and a second seed of the main recipe.
source "$(dirname "$0")/fix_common.sh"
data
train "$F/data/antidef-neutral.jsonl" "$F/qwen14b-neutralonly-s17" 17 6
cross "$Q14" "$Q14R" "$F/plain-qwen14b-neutralonly-s17" "$PLAIN" reviewer "$F/qwen14b-neutralonly-s17"
train "$F/data/antidef-all.jsonl" "$F/qwen14b-antidef-s29" 29 2
cross "$Q14" "$Q14R" "$F/plain-qwen14b-antidef-s29" "$PLAIN" reviewer "$F/qwen14b-antidef-s29"
echo "=== arm1 controls done ==="
