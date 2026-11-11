#!/usr/bin/env bash
source "$(dirname "$0")/fix_common.sh"
[ -f "$F/direction-qwen14b/direction.pt" ] || "$PY" scripts/wsp_direction_edit.py --output "$F/direction-qwen14b"
echo "=== arm2 done ==="
