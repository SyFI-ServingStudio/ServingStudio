#!/usr/bin/env bash
# Measure the rms_norm target module on the 27931 before vs after B200 images across a token
# sweep, to (a) confirm the vec8 bandwidth win is resolvable and (b) set before_us/after_us for
# issue_27931.json. PR 27931 is a scalar->vec8 rms_norm kernel: null at decode (few tokens),
# only shows at high token counts, so we sweep tokens to find where it separates.
#
# Runs the extractor INSIDE each image; binds the GPU from DOCKER_GPU_ARG (set by slurm_gpu.sh)
# so this obeys the slurm allocation. Standalone falls back to device=3.
# Usage (under slurm):  sbatch slurm_gpu.sh ./baseline_27931.sh
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SNAP=/raid/yilegu/models/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
TARGET=model.layers.0.input_layernorm
BEFORE=rga-local/vllm-pr-27931:b200-before
AFTER=rga-local/vllm-pr-27931:b200-after
EXTRACTOR="$HERE/extract_and_profile.py"
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
GOLDEN=/raid/yilegu/eval_goldens/27931_input_layernorm.pt
OUT="$HERE/iter_opt_eval_27931"; mkdir -p "$OUT"

echo "==== 27931 baseline: target=$TARGET gpu=$GPU ===="

run_in_image () {  # <image> <mode-args...>
  local img="$1"; shift
  local cname="bl27931_$(date +%s%N)"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run -d --name "$cname" --gpus "$GPU" \
    -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -v /raid/yilegu/models:/raid/yilegu/models:ro \
    -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer \
    -v /raid/yilegu/eval_goldens:/raid/yilegu/eval_goldens \
    --workdir /tmp "$img" sleep infinity >/dev/null
  docker cp "$EXTRACTOR" "$cname:/tmp/extract_and_profile.py" >/dev/null
  docker exec --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 "$cname" \
    python3 /tmp/extract_and_profile.py --model "$SNAP" --framework vllm --target "$TARGET" "$@"
  local rc=$?
  docker rm -f "$cname" >/dev/null 2>&1 || true
  return $rc
}

for TOK in 8 512 2048 4096; do
  echo "==================== tokens=$TOK ===================="
  # golden from the BEFORE image at this token count (fresh per token count)
  run_in_image "$BEFORE" --tokens "$TOK" --iters 3000 --capture "$GOLDEN" \
      2>&1 | tee "$OUT/before_tok${TOK}.log" | grep -E "LATENCY|CHECK|us" || true
  echo "--- replay on AFTER (tokens=$TOK) ---"
  run_in_image "$AFTER" --tokens "$TOK" --iters 3000 --replay "$GOLDEN" \
      2>&1 | tee "$OUT/after_tok${TOK}.log" | grep -E "LATENCY|CHECK|us" || true
done
echo "==== done -> $OUT ===="
