#!/usr/bin/env bash
# Round 2 on unedited Aya 32B (3 cards: two-card forward runs of this model crashed twice in round 1):
# R4 pressure wordings, R6 HumanEval.
source "$(dirname "$0")/fix_common.sh"
for w in $PRESSURE; do
  cross "$A32" "$A32R" "$F/wording-aya32b-base-$w" "$PLAIN" "$w"
done
humaneval "$A32" "$A32R" "$D/runs/bench/aya32b-base-humaneval.json"
echo "=== r2 aya32b base done ==="
