#!/usr/bin/env bash
# Round 2 R8 (exploratory): off-task opinion test. $1 = qwen14b (1 card) | aya32b (3 cards)
source "$(dirname "$0")/fix_common.sh"
case "$1" in qwen14b) M=$Q14; R=$Q14R;; aya32b) M=$A32; R=$A32R;; *) echo "unknown $1"; exit 2;; esac
off() {  # out [adapter]
  [ -f "$1/table.json" ] && { echo "skip $1"; return 0; }
  echo "=== offdomain $(basename "$1") ==="
  local extra=(); [ -n "${2:-}" ] && extra=(--adapter-id "$2")
  "$PY" scripts/offdomain_deference.py --model-id "$M" --revision "$R" "${extra[@]}" --output "$1" || echo "!!! FAILED $1"
}
off "$F/offdomain-$1-base"
off "$F/offdomain-$1-antidef" "$F/$1-antidef-s17"
echo "=== r2 offdomain $1 done ==="
