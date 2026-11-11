#!/usr/bin/env bash
# Round 2 smoke (1 card, minutes): every new code path once on 6 cases with Qwen-14B.
source "$(dirname "$0")/fix_common.sh"
S="$D/runs/fix-smoke/r2"; rm -rf "$S"; mkdir -p "$S"
"$PY" scripts/scale_adapter.py --adapter "$F/qwen14b-antidef-s17" --scale 0.5 --output "$S/x0.5"
for w in $PRESSURE; do
  "$PY" scripts/wsp_cross_model.py --model-id "$Q14" --revision "$Q14R" --dataset "$PLAIN" --wording "$w" \
    --output "$S/base-$w" --limit 6 --context-limit 4096
done
"$PY" scripts/wsp_cross_model.py --model-id "$Q14" --revision "$Q14R" --adapter-id "$S/x0.5" \
  --dataset "$PLAIN" --wording pushback --output "$S/x0.5-pushback" --limit 6 --context-limit 4096
"$PY" scripts/wsp_flip_analysis.py "$S"/base-* "$S/x0.5-pushback"
echo "=== r2 smoke done ==="
