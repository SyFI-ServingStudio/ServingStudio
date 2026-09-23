#!/usr/bin/env bash
# Measure both K3 layers with the realistic routing flags (--local-topk 2 --hidden-scale 1.0) on
# GPU 3 directly, idle-gated (<1000 MiB) before each layer so we never overlap another job.
#   setsid nohup ./run_realistic_sanity.sh > codex_runs/driver_realistic_sanity.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="${K3_GPU_INDEX:-3}"
export DOCKER_GPU_ARG="\"device=$GPU_IDX\""
wait_idle () {
  while :; do
    used="$(nvidia-smi -i "$GPU_IDX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    [ "${used:-99999}" -lt 1000 ] && return 0
    echo "$(date +%T) GPU $GPU_IDX busy (${used} MiB) -> wait 60s"; sleep 60
  done
}
KDA="--attn-type kda --moe-backend flashinfer_mxfp4 --attn-heads 12 --mamba-ssm-dtype bfloat16 --experts 112 --ep 8 --cuda-graph --iters 40 --warmup 10 --point 128,8192;32,8192;1,8192"
MLA="--attn-type mla --moe-backend flashinfer_mxfp4 --attention-backend cutedsl_mla --attn-heads 12 --kv-cache-dtype fp8_e4m3 --experts 112 --ep 8 --cuda-graph --iters 40 --warmup 10 --point 128,8192;1,1048576;16,65536"
cd "$HERE"
for at in kda mla; do
  wait_idle
  [ "$at" = kda ] && A="$KDA" || A="$MLA"
  echo "==== $(date +%T) $at realistic (local_topk 2, hidden_scale 1.0)"
  RESULT_TAG="${at}_realistic_" ./run_kimi_single_layer.sh $A --local-topk 2 --hidden-scale 1.0 \
    --profile-kernels "/work/kimi_single_layer/profiles/${at}_realistic.json" 2>&1 \
    | grep -E "^JSON|Error|error|Traceback" | cut -c1-220
done
echo REALISTIC_SANITY_DONE
