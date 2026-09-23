#!/usr/bin/env bash
# Run the K3 JIT fill (B8's new qkvbfg leaves: 136 missing rows) on GPU 3 directly, but only
# AFTER the direct trial queue has finished, so kernel profiling never overlaps the trials'
# timing measurements on the same device.
#   setsid nohup ./after_trials_fill.sh > codex_runs/after_trials_fill.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="${K3_GPU_INDEX:-3}"
while pgrep -f "[r]un_k3_trials_direct.sh" >/dev/null; do
  echo "$(date +%T) trials still running -> wait 120s"; sleep 120
done
while :; do
  used="$(nvidia-smi -i "$GPU_IDX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [ "${used:-99999}" -lt 1000 ] && break
  echo "$(date +%T) GPU $GPU_IDX busy (${used} MiB) -> wait 60s"; sleep 60
done
echo "==== $(date +%F_%T) trials done, GPU $GPU_IDX idle -> JIT fill (repo main-k3-rust @ $(git -C "$HERE/../../main-k3-rust" rev-parse --short HEAD))"
K3_GPU_INDEX="$GPU_IDX" "$HERE/run_k3_jit_fill.sh" \
  presets/predict_kimi_k3_b200_rank1_layer_kda.json \
  presets/predict_kimi_k3_b200_rank1_layer_mla.json
echo "==== $(date +%F_%T) fill done exit=$?"
