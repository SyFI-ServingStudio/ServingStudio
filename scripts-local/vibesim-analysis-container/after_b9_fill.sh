#!/usr/bin/env bash
# After Codex B9 exits: JIT-fill the new 112-expert mxfp4_fused_moe rows (rank-1 presets) on
# GPU 3 directly, so the post-B9 predictions can be compared with the measured graph times.
#   setsid nohup ./after_b9_fill.sh > codex_runs/after_b9_fill.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="${K3_GPU_INDEX:-3}"
while pgrep -f "[b]9_mxfp4_row_fidelity" >/dev/null; do echo "$(date +%T) B9 still running -> wait 120s"; sleep 120; done
while :; do
  used="$(nvidia-smi -i "$GPU_IDX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [ "${used:-99999}" -lt 1000 ] && break
  echo "$(date +%T) GPU $GPU_IDX busy (${used} MiB) -> wait 60s"; sleep 60
done
echo "==== $(date +%F_%T) B9 done -> JIT fill (repo main-k3-rust @ $(git -C "$HERE/../../main-k3-rust" rev-parse --short HEAD))"
K3_GPU_INDEX="$GPU_IDX" "$HERE/run_k3_jit_fill.sh" \
  presets/predict_kimi_k3_b200_rank1_layer_kda.json \
  presets/predict_kimi_k3_b200_rank1_layer_mla.json
echo "==== $(date +%F_%T) fill done exit=$?"
for r in kda mla; do
  echo "-- $r"; grep -E '"batch_tokens"|^total:' "$HERE/../../main-k3-rust/logs/predict_kimi_k3_b200_rank1_layer_$r/reports/iter_breakdown.ans" \
    | sed 's/"prefill_chunk_pairs":\[\]//' | paste - - | cut -c1-160
done
