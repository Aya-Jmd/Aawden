#!/usr/bin/env bash
# Unedited Qwen-14B on everything the adapter is tested on: held-out wordings, real code, HumanEval.
source "$(dirname "$0")/fix_common.sh"
for w in authority ciso scanner consensus code_comment; do
  cross "$Q14" "$Q14R" "$F/wording-qwen14b-base-$w" "$PLAIN" "$w"
done
cross "$Q14" "$Q14R" "$F/real-qwen14b-base" "$REAL" reviewer
[ -f "$D/runs/bench/qwen14b-humaneval.json" ] || "$PY" scripts/bench_code.py --model-id "$Q14" \
  --revision "$Q14R" --benchmark humaneval --output "$D/runs/bench/qwen14b-humaneval.json"
echo "=== base14b done ==="
