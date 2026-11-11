#!/usr/bin/env bash
# Round 2 R3 (2 cards): the unchanged recipe on a third model, Qwen2.5-32B. Plain, real code, five wordings.
source "$(dirname "$0")/fix_common.sh"
data
AD="$F/qwen32b-antidef-s17"
train_on "$Q32" "$Q32R" "$F/data/antidef-all.jsonl" "$AD" 17 2
cross "$Q32" "$Q32R" "$F/plain-qwen32b-antidef-s17" "$PLAIN" reviewer "$AD"
cross "$Q32" "$Q32R" "$F/real-qwen32b-antidef" "$REAL" reviewer "$AD"
for w in authority ciso scanner consensus code_comment; do
  cross "$Q32" "$Q32R" "$F/wording-qwen32b-antidef-$w" "$PLAIN" "$w" "$AD"
done
echo "=== r2 qwen32b done ==="
