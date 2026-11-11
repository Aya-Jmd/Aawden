#!/usr/bin/env bash
# Every new code path once, tiny: data -> 14B LoRA (memory check at real lengths) -> adapter eval -> direction edit.
source "$(dirname "$0")/fix_common.sh"
S="$D/runs/fix-smoke"; rm -rf "$S"; mkdir -p "$S"
data
"$PY" scripts/distill_sft.py --base "$Q14" --revision "$Q14R" --traces "$F/data/antidef-all.jsonl" \
  --output "$S/adapter" --epochs 1 --grad-accum 8 --limit 64 --log-every 2 --seed 17
"$PY" scripts/wsp_cross_model.py --model-id "$Q14" --revision "$Q14R" --adapter-id "$S/adapter" \
  --dataset "$PLAIN" --limit 4 --output "$S/cross" --context-limit 4096
"$PY" scripts/wsp_cross_model.py --model-id "$Q14" --revision "$Q14R" --dataset "$REAL" --limit 4 \
  --wording ciso --output "$S/cross-real" --context-limit 4096
"$PY" scripts/wsp_direction_edit.py --limit 6 --layer-step 12 --seeds 17 --output "$S/direction"
echo "=== smoke done ==="
