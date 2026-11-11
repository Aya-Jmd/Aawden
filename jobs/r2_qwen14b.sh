#!/usr/bin/env bash
# Round 2 on Qwen-14B (1 card): R1 third seed, R4 pressure wordings on unedited and seed-17 adapter.
source "$(dirname "$0")/fix_common.sh"
data
train "$F/data/antidef-all.jsonl" "$F/qwen14b-antidef-s43" 43 2
cross "$Q14" "$Q14R" "$F/plain-qwen14b-antidef-s43" "$PLAIN" reviewer "$F/qwen14b-antidef-s43"
for w in $PRESSURE; do
  cross "$Q14" "$Q14R" "$F/wording-qwen14b-base-$w" "$PLAIN" "$w"
  cross "$Q14" "$Q14R" "$F/wording-qwen14b-antidef-$w" "$PLAIN" "$w" "$F/qwen14b-antidef-s17"
done
echo "=== r2 qwen14b done ==="
