#!/usr/bin/env bash
# Arm 5: the method that PASSED on Qwen-14B (anti-deference LoRA, identical recipe and data), on
# Aya Expanse 32B -- the most deferent model measured (json flip 0.944). Sharded across the cards.
source "$(dirname "$0")/fix_common.sh"
data
AD="$F/aya32b-antidef-s17"
if [ ! -f "$AD/adapter_model.safetensors" ]; then
  echo "=== train aya32b-antidef-s17 ==="
  "$PY" scripts/distill_sft.py --base "$A32" --revision "$A32R" --traces "$F/data/antidef-all.jsonl" \
    --output "$AD" --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --grad-accum 8 --max-len 2048 --seed 17 \
    --device-map auto --max-gpu-gib 35
fi
cross "$A32" "$A32R" "$F/plain-aya32b-antidef-s17" "$PLAIN" reviewer "$AD"
cross "$A32" "$A32R" "$F/real-aya32b-antidef" "$REAL" reviewer "$AD"
for w in ciso scanner code_comment authority consensus; do
  cross "$A32" "$A32R" "$F/wording-aya32b-antidef-$w" "$PLAIN" "$w" "$AD"
done
echo "=== arm5 adapter done ==="
