#!/usr/bin/env bash
# Round 3 (R9) on Aya 32B (2 cards): the recipe plus prior-answer rows, then the plain set and the
# three pressure wordings. `pushback` is not in the training data.
source "$(dirname "$0")/fix_common.sh"
data3
train_on "$A32" "$A32R" "$F/data/antidef-prior.jsonl" "$F/aya32b-prior-s17" 17 2
cross "$A32" "$A32R" "$F/plain-aya32b-prior-s17" "$PLAIN" reviewer "$F/aya32b-prior-s17"
for w in pushback argued insistent; do
  cross "$A32" "$A32R" "$F/wording-aya32b-prior-$w" "$PLAIN" "$w" "$F/aya32b-prior-s17"
done
echo "=== r3 aya32b done ==="
