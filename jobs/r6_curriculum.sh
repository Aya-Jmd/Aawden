#!/usr/bin/env bash
# Round 6 (R12), one card: stage 1 = the neutral-only adapter (already trained), stage 2 = continue it on
# the multi-wording data.  usage: r6_curriculum.sh <aya8b|tinyaya> [seeds]
source "$(dirname "$0")/fix_common.sh"
case "$1" in
  aya8b)   M="$A8"; R="$A8R" ;;
  tinyaya) M="$TA"; R="$TAR" ;;
  *) echo "unknown model $1"; exit 2 ;;
esac
K="$1"; SEEDS="${2:-17 29 43}"
data5
[ -f "$F/$K-neutralonly-s17/adapter_model.safetensors" ] || { echo "!!! FAILED stage-1 adapter missing"; exit 1; }
for s in $SEEDS; do
  out="$F/$K-curr-s$s"
  if [ ! -f "$out/adapter_model.safetensors" ]; then
    echo "=== train $(basename "$out") seed=$s (stage 2 from neutralonly-s17) ==="
    "$PY" scripts/distill_sft.py --base "$M" --revision "$R" --traces "$F/data/antidef-multi.jsonl" --output "$out" \
      --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --grad-accum 8 --max-len 2048 --seed "$s" \
      --device-map auto --max-gpu-gib 35 --init-adapter "$F/$K-neutralonly-s17"
  fi
  cross "$M" "$R" "$F/plain-$K-curr-s$s" "$PLAIN" reviewer "$out"
  for w in $HELDW authority scanner code_comment; do
    cross "$M" "$R" "$F/wording-$K-curr-s$s-$w" "$PLAIN" "$w" "$out"
  done
done
cross "$M" "$R" "$F/real-$K-curr" "$REAL" reviewer "$F/$K-curr-s17"
echo "=== r6 $K done ==="
