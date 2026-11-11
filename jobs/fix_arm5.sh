#!/usr/bin/env bash
# Arm 5 (stretch, only after Arm 1 or 2 passes on Qwen-14B): the direction edit on Aya Expanse 32B,
# the most deferent model measured (flip 0.944). Forward-only, so it runs on 2 cards.
source "$(dirname "$0")/fix_common.sh"
[ -f "$F/direction-aya32b/direction.pt" ] || "$PY" scripts/wsp_direction_edit.py \
  --model-id "$A32" --revision "$A32R" --output "$F/direction-aya32b"
echo "=== arm5 done ==="
