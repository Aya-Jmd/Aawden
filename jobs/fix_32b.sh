#!/usr/bin/env bash
# Arms 3+4 on a 32B model (2 cards): real code, then the five non-default wordings. $1 = aya32b|qwen32b
source "$(dirname "$0")/fix_common.sh"
case "$1" in aya32b) M=$A32; R=$A32R;; qwen32b) M=$Q32; R=$Q32R;; *) echo "unknown $1"; exit 2;; esac
cross "$M" "$R" "$F/real-$1-base" "$REAL" reviewer
for w in authority ciso scanner consensus code_comment; do
  cross "$M" "$R" "$F/wording-$1-base-$w" "$PLAIN" "$w"
done
echo "=== 32b $1 done ==="
