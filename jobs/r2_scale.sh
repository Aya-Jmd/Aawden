#!/usr/bin/env bash
# Round 2 R5 (exploratory): the seed-17 Qwen-14B adapter at 0.25 / 0.5 / 0.75 strength, plain + real code.
source "$(dirname "$0")/fix_common.sh"
for s in 0.25 0.5 0.75; do
  AD="$F/qwen14b-antidef-s17-x$s"
  [ -f "$AD/adapter_model.safetensors" ] || "$PY" scripts/scale_adapter.py \
    --adapter "$F/qwen14b-antidef-s17" --scale "$s" --output "$AD"
  cross "$Q14" "$Q14R" "$F/plain-qwen14b-antidef-x$s" "$PLAIN" reviewer "$AD"
  cross "$Q14" "$Q14R" "$F/real-qwen14b-antidef-x$s" "$REAL" reviewer "$AD"
done
echo "=== r2 scale done ==="
