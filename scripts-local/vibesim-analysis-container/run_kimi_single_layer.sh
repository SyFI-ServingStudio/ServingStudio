#!/usr/bin/env bash
# Docker wrapper: run kimi_single_layer_decode.py inside lmsysorg/sglang:v0.5.16-runtime
# on the slurm-assigned GPU.  Invoke under slurm:
#   sbatch slurm_gpu.sh ./run_kimi_single_layer.sh [extra args to the .py]
# GPU is bound via $DOCKER_GPU_ARG (exported by slurm_gpu.sh); falls back to device=3
# for a manual (non-slurm) smoke run only.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMG="${SGLANG_IMG:-lmsysorg/sglang:v0.5.20}"
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
OUT="$HERE/kimi_single_layer"; mkdir -p "$OUT"
CNAME="kimi_single_layer_$(date +%s%N)"
STAMP="$(date +%Y%m%d_%H%M%S)"
# RESULT_TAG lets callers name the JSON (e.g. RESULT_TAG=mla_ -> result_mla_<stamp>.json)
JSON="/work/kimi_single_layer/result_${RESULT_TAG:-}${STAMP}.json"

echo "==== kimi_single_layer: img=$IMG gpu=$GPU args=$* ===="
docker rm -f "$CNAME" >/dev/null 2>&1 || true
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# --shm-size large: fused-MoE / triton need scratch; long-context KV lives in GPU mem.
docker run --rm --name "$CNAME" --gpus "$GPU" \
  --shm-size 32g \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -e SGLANG_DISABLE_LEAN_ATTENTION="${SGLANG_DISABLE_LEAN_ATTENTION:-}" \
  -e TOKENIZERS_PARALLELISM=false \
  -e SGLANG_OPT_FUSED_KDA_VERIFY=0 \
  -v /raid/hf/hub:/raid/hf/hub:ro \
  -v "$HERE":/work \
  --workdir /tmp \
  --entrypoint python3 "$IMG" \
  /work/kimi_single_layer_decode.py --json-out "$JSON" "$@" 2>&1

echo "==== done; json -> $OUT/result_${STAMP}.json ===="
