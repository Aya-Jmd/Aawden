#!/usr/bin/env bash
# Arm 1 generalisation: held-out wordings, real code, and coding regression for the seed-17 adapter.
source "$(dirname "$0")/fix_common.sh"
AD="$F/qwen14b-antidef-s17"
for w in authority ciso scanner consensus code_comment; do
  cross "$Q14" "$Q14R" "$F/wording-qwen14b-antidef-$w" "$PLAIN" "$w" "$AD"
done
cross "$Q14" "$Q14R" "$F/real-qwen14b-antidef" "$REAL" reviewer "$AD"
[ -f "$D/runs/bench/qwen14b-antidef-humaneval.json" ] || "$PY" scripts/bench_code.py --model-id "$Q14" \
  --revision "$Q14R" --adapter-id "$AD" --benchmark humaneval --output "$D/runs/bench/qwen14b-antidef-humaneval.json"
echo "=== arm1 heldout done ==="
