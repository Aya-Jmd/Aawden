#!/usr/bin/env bash
# Round 2 on Aya 32B adapters (2 cards): R2 neutral-only control, R1 second seed, R4 pressure wordings
# on the seed-17 adapter, R6 HumanEval with the adapter.
source "$(dirname "$0")/fix_common.sh"
data
train_on "$A32" "$A32R" "$F/data/antidef-neutral.jsonl" "$F/aya32b-neutralonly-s17" 17 6
cross "$A32" "$A32R" "$F/plain-aya32b-neutralonly-s17" "$PLAIN" reviewer "$F/aya32b-neutralonly-s17"
for w in $PRESSURE; do
  cross "$A32" "$A32R" "$F/wording-aya32b-antidef-$w" "$PLAIN" "$w" "$F/aya32b-antidef-s17"
done
train_on "$A32" "$A32R" "$F/data/antidef-all.jsonl" "$F/aya32b-antidef-s29" 29 2
cross "$A32" "$A32R" "$F/plain-aya32b-antidef-s29" "$PLAIN" reviewer "$F/aya32b-antidef-s29"
humaneval "$A32" "$A32R" "$D/runs/bench/aya32b-antidef-humaneval.json" "$F/aya32b-antidef-s17"
echo "=== r2 aya32b adapters done ==="
