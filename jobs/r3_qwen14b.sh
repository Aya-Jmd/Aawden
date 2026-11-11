#!/usr/bin/env bash
# Round 3 (R9) on Qwen2.5-14B (1 card): same data, to check the extra rows cost nothing where the
# round-2 adapter already followed the code under pushback.
source "$(dirname "$0")/fix_common.sh"
data3
train "$F/data/antidef-prior.jsonl" "$F/qwen14b-prior-s17" 17 2
cross "$Q14" "$Q14R" "$F/plain-qwen14b-prior-s17" "$PLAIN" reviewer "$F/qwen14b-prior-s17"
for w in pushback argued insistent; do
  cross "$Q14" "$Q14R" "$F/wording-qwen14b-prior-$w" "$PLAIN" "$w" "$F/qwen14b-prior-s17"
done
echo "=== r3 qwen14b done ==="
