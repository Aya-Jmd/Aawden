#!/usr/bin/env bash
# Round 5 (R11), one card: train on four wordings, test on the five held out.  usage: r5_multi.sh <aya8b|tinyaya> [seeds]
source "$(dirname "$0")/fix_common.sh"
case "$1" in
  aya8b)   M="$A8"; R="$A8R" ;;
  tinyaya) M="$TA"; R="$TAR" ;;
  *) echo "unknown model $1"; exit 2 ;;
esac
K="$1"; SEEDS="${2:-17 29 43}"
data5
for s in $SEEDS; do
  train_on "$M" "$R" "$F/data/antidef-multi.jsonl" "$F/$K-multi-s$s" "$s" 2
  cross "$M" "$R" "$F/plain-$K-multi-s$s" "$PLAIN" reviewer "$F/$K-multi-s$s"
  for w in $HELDW authority scanner code_comment; do
    cross "$M" "$R" "$F/wording-$K-multi-s$s-$w" "$PLAIN" "$w" "$F/$K-multi-s$s"
  done
done
cross "$M" "$R" "$F/real-$K-multi" "$REAL" reviewer "$F/$K-multi-s17"
echo "=== r5 $K done ==="
