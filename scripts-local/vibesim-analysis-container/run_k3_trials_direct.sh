#!/usr/bin/env bash
# Run K3 optimizer trials DIRECTLY on one GPU (no slurm allocation), sequentially.
#
# Why: a trial holds a GPU only while the driver/judge run (~10 of ~55 min); under
# sbatch the whole session pins an allocation. Direct docker `--gpus device=N` on the
# authorized repro GPU shares it instead (user decision 2026-09-23: "do not use slurm
# for your run just directly use GPU"; GPU 2/3 are the authorized ones, never 0,1,4-7).
#
# Usage: run_k3_trials_direct.sh GPU_INDEX "<case>:<trial>" ["<case>:<trial>" ...]
#   e.g. run_k3_trials_direct.sh 3 kda:6 mla:5 kda:7 mla:6
# Waits for the GPU to be idle (< 1000 MiB used) before each trial.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="$1"; shift
case "$GPU_IDX" in 2|3) ;; *) echo "!! GPU $GPU_IDX not authorized (only 2,3)"; exit 1;; esac
export DOCKER_GPU_ARG="\"device=$GPU_IDX\""
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-1800}"

wait_idle () {
  while :; do
    used="$(nvidia-smi -i "$GPU_IDX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    [ "${used:-99999}" -lt 1000 ] && return 0
    echo "== GPU $GPU_IDX busy (${used} MiB used) -> waiting 60s"; sleep 60
  done
}

for spec in "$@"; do
  case_name="${spec%%:*}"; k="${spec##*:}"
  wait_idle
  echo "==== $(date +%F_%T) start ${case_name} trial ${k} on GPU $GPU_IDX (agent budget ${AGENT_TIMEOUT}s)"
  START="$k" "$HERE/run_iter_opt_eval.sh" "$HERE/issue_k3_${case_name}.json" "$k" max \
    > "$HERE/direct_${case_name}_trial_${k}.out" 2>&1 || echo "!! ${case_name} trial ${k} exited non-zero"
  echo "==== $(date +%F_%T) done ${case_name} trial ${k}"
done
