#!/usr/bin/env bash
# Round 4 (R10), one card: the round-1 recipe on a small Aya model.  usage: r4_small_aya.sh <aya8b|tinyaya>
source "$(dirname "$0")/fix_common.sh"
case "$1" in
  aya8b)   M="$A8"; R="$A8R" ;;
  tinyaya) M="$TA"; R="$TAR" ;;
  *) echo "unknown model $1"; exit 2 ;;
esac
K="$1"
data; data3
train_on "$M" "$R" "$F/data/antidef-all.jsonl" "$F/$K-antidef-s17" 17 2
cross "$M" "$R" "$F/plain-$K-antidef-s17" "$PLAIN" reviewer "$F/$K-antidef-s17"
train_on "$M" "$R" "$F/data/antidef-neutral.jsonl" "$F/$K-neutralonly-s17" 17 6
cross "$M" "$R" "$F/plain-$K-neutralonly-s17" "$PLAIN" reviewer "$F/$K-neutralonly-s17"
train_on "$M" "$R" "$F/data/antidef-all.jsonl" "$F/$K-antidef-s29" 29 2
cross "$M" "$R" "$F/plain-$K-antidef-s29" "$PLAIN" reviewer "$F/$K-antidef-s29"
for w in $ALLW; do
  cross "$M" "$R" "$F/wording-$K-antidef-$w" "$PLAIN" "$w" "$F/$K-antidef-s17"
  cross "$M" "$R" "$F/wording-$K-base-$w" "$PLAIN" "$w"
done
train_on "$M" "$R" "$F/data/antidef-prior.jsonl" "$F/$K-prior-s17" 17 2
cross "$M" "$R" "$F/plain-$K-prior-s17" "$PLAIN" reviewer "$F/$K-prior-s17"
cross "$M" "$R" "$F/wording-$K-prior-pushback" "$PLAIN" pushback "$F/$K-prior-s17"
cross "$M" "$R" "$F/real-$K-base" "$REAL" reviewer
cross "$M" "$R" "$F/real-$K-antidef" "$REAL" reviewer "$F/$K-antidef-s17"
if [ "$K" = aya8b ]; then
  humaneval "$M" "$R" "$D/runs/bench/aya8b-base-humaneval.json"
  humaneval "$M" "$R" "$D/runs/bench/aya8b-antidef-humaneval.json" "$F/aya8b-antidef-s17"
fi
echo "=== r4 $K done ==="
