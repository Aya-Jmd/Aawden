#!/usr/bin/env bash
# Competence -> deference chain, in one job.
#
# Why this exists
#   The submission's central claim is that deference cannot be measured on a model with no skill.
#   This runs the counterfactual directly: it trains coding/review competence INTO Aya 8B from a
#   teacher's traces, then measures the same model twice --
#       (1) on a real public benchmark (can it code now?), and
#       (2) by flip rate on our review fixtures (does competence arrive with deference?).
#
#   Either outcome is a result:
#     competent + deferent  -> deference follows competence; the earlier null was a precondition
#                              failure, and the small model is now a valid test bed.
#     competent + robust    -> deference is a separate capacity that scale provides; the 32B result
#                              is not explained by competence alone.
#
# Order matters: training first, then benchmark, then flip rate on the SAME adapter, so a partial
# failure leaves the earlier stages' artifacts on disk.

set -euo pipefail
D=/cfs/earth/scratch/mekalris/aawden
cd "$D"

export HF_HOME="/cfs/earth/scratch/mekalris/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_TOKEN="$(cat "$D/private/hf_token_multi")"

ADAPTER="runs/distill/aya8b-sft"
mkdir -p runs/distill runs/bench

echo "=== [1/4] SFT on teacher traces ==="
"$D/env/bin/python" scripts/distill_sft.py \
  --base CohereLabs/aya-expanse-8b \
  --revision 5062468bf9bc0c6035fd64e06274333ec127d980 \
  --traces "$D/runs/distill/traces-review.jsonl" \
  --output "$D/$ADAPTER" \
  --epochs 2 --lr 1e-4 --rank 16 --alpha 32 \
  --batch-size 1 --grad-accum 8 --max-len 2048 \
  || { echo "!!! SFT FAILED"; exit 1; }

echo "=== [2/4] pass@1 on HumanEval WITH the adapter (competence check) ==="
"$D/env/bin/python" scripts/bench_code.py \
  --model-id CohereLabs/aya-expanse-8b \
  --revision 5062468bf9bc0c6035fd64e06274333ec127d980 \
  --adapter-id "$D/$ADAPTER" \
  --benchmark humaneval \
  --output "$D/runs/bench/aya8b-sft-humaneval.json" || echo "!!! bench FAILED"

echo "=== [3/4] flip rate on the plain review set WITH the adapter (deference check) ==="
"$D/env/bin/python" scripts/wsp_cross_model.py \
  --model-id CohereLabs/aya-expanse-8b \
  --revision 5062468bf9bc0c6035fd64e06274333ec127d980 \
  --adapter-id "$D/$ADAPTER" \
  --dataset "$D/datasets/plain-pooled.json" \
  --output "$D/runs/distill/plain-aya8b-sft" \
  --context-limit 4096 || echo "!!! flip FAILED"

echo "=== [4/4] analysis ==="
"$D/env/bin/python" scripts/wsp_flip_analysis.py \
  "$D/runs/wsp-v3/plain-aya8b" "$D/runs/distill/plain-aya8b-sft" \
  --instrument json_prefill \
  --output "$D/runs/distill/deference-before-after.json" || echo "!!! analysis FAILED"

echo "=== chain done ==="
