# Sourced by every fix-arm job. Offline: every model and benchmark is already in the cache.
set -euo pipefail
D=/cfs/earth/scratch/mekalris/aawden; cd "$D"
export HF_HOME=/cfs/earth/scratch/mekalris/.cache/huggingface HF_HUB_CACHE=/cfs/earth/scratch/mekalris/.cache/huggingface/hub
export HF_TOKEN="$(cat private/hf_token_multi)" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PY="$D/env/bin/python"
Q14=Qwen/Qwen2.5-14B-Instruct;   Q14R=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
Q32=Qwen/Qwen2.5-32B-Instruct;   Q32R=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
A32=CohereLabs/aya-expanse-32b;  A32R=b306ea27e360683b50c005d7fcbad6a242317910
A8=CohereLabs/aya-expanse-8b;   A8R=5062468bf9bc0c6035fd64e06274333ec127d980
TA=CohereLabs/tiny-aya-global;   TAR=af89d219b53ed9b13b8a4645f9c8028973510324
F="$D/runs/fix"; mkdir -p "$F"

cross() {  # model rev out dataset wording [adapter]  -- skip if done
  local m="$1" r="$2" out="$3" ds="$4" w="$5" ad="${6:-}"
  [ -f "$out/table.json" ] && { echo "skip $out (exists)"; return 0; }
  echo "=== cross $(basename "$out") wording=$w ==="
  local extra=(); [ -n "$ad" ] && extra=(--adapter-id "$ad")
  "$PY" scripts/wsp_cross_model.py --model-id "$m" --revision "$r" "${extra[@]}" \
    --dataset "$ds" --wording "$w" --output "$out" --context-limit 4096 || echo "!!! FAILED $out"
}

train() {  # traces out seed epochs
  local tr="$1" out="$2" seed="$3" ep="$4"
  [ -f "$out/adapter_model.safetensors" ] && { echo "skip train $out (exists)"; return 0; }
  echo "=== train $(basename "$out") seed=$seed epochs=$ep ==="
  "$PY" scripts/distill_sft.py --base "$Q14" --revision "$Q14R" --traces "$tr" --output "$out" \
    --epochs "$ep" --lr 1e-4 --rank 16 --alpha 32 --grad-accum 8 --max-len 2048 --seed "$seed"
}

data() {
  mkdir -p "$F/data"
  [ -f "$F/data/antidef-all.jsonl" ] || "$PY" scripts/antidef_make_data.py --output "$F/data/antidef-all.jsonl"
  [ -f "$F/data/antidef-neutral.jsonl" ] || "$PY" scripts/antidef_make_data.py --conditions neutral --output "$F/data/antidef-neutral.jsonl"
}
PLAIN="$D/datasets/plain-pooled.json"; REAL="$D/datasets/realcode-sinkchecked.json"

train_on() {  # base rev traces out seed epochs  -- multi-card, shards balanced under 35 GiB each
  local b="$1" r="$2" tr="$3" out="$4" seed="$5" ep="$6"
  [ -f "$out/adapter_model.safetensors" ] && { echo "skip train $out (exists)"; return 0; }
  echo "=== train $(basename "$out") seed=$seed epochs=$ep ==="
  "$PY" scripts/distill_sft.py --base "$b" --revision "$r" --traces "$tr" --output "$out" \
    --epochs "$ep" --lr 1e-4 --rank 16 --alpha 32 --grad-accum 8 --max-len 2048 --seed "$seed" \
    --device-map auto --max-gpu-gib 35
}

humaneval() {  # model rev out [adapter]
  local m="$1" r="$2" out="$3" ad="${4:-}"
  [ -f "$out" ] && { echo "skip $out (exists)"; return 0; }
  echo "=== humaneval $(basename "$out") ==="
  local extra=(); [ -n "$ad" ] && extra=(--adapter-id "$ad")
  "$PY" scripts/bench_code.py --model-id "$m" --revision "$r" "${extra[@]}" --benchmark humaneval \
    --output "$out" || echo "!!! FAILED $out"
}
PRESSURE="argued insistent pushback"
data3() {  # round 3 (R9): the main recipe plus prior-answer rows
  mkdir -p "$F/data"
  [ -f "$F/data/antidef-prior.jsonl" ] || "$PY" scripts/antidef_make_data.py --prior-answer --output "$F/data/antidef-prior.jsonl"
}
ALLW="authority ciso scanner consensus code_comment argued insistent pushback"
TRAINW="reviewer,authority,scanner,code_comment"; HELDW="ciso consensus argued insistent pushback"
data5() {  # round 5 (R11): four trained wordings plus prior-answer rows
  mkdir -p "$F/data"
  [ -f "$F/data/antidef-multi.jsonl" ] || "$PY" scripts/antidef_make_data.py --wordings "$TRAINW" --prior-answer --output "$F/data/antidef-multi.jsonl"
}
