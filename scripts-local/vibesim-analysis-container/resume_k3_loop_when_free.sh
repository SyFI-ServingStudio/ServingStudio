#!/usr/bin/env bash
# Resume the K3 continuous loop as soon as an authorized GPU has room again.
#   GPU 2/3: must be idle (< 1000 MiB used)      -> idle-gated rounds
#   GPU 7:   user-shared; needs >= 60 GB free   -> K3_SHARED_GPU=1 (no idle gate)
# 2026-09-24 20:25: all three were full (vLLM TP workers on 2/3 at 168 GB, sglang scheduler on 7 at
# 167 GB) and the judge OOM'd (KDA round 20) -> the loop must not run under those conditions.
# Usage: resume_k3_loop_when_free.sh <first_round_k> <n_rounds_per_case>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
K0="$1"; N="$2"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}"
# A GPU counts as idle only after IDLE_CHECKS consecutive idle readings (default 30 = 1 h): the
# user's vLLM jobs on 2/3 cycle with 5-10 min gaps (22:44 and 22:53 on 2026-09-24 both fooled a
# 10-min streak), and grabbing one during a gap double-books their next run.
IDLE_CHECKS="${IDLE_CHECKS:-30}"
declare -A IDLE_STREAK=([2]=0 [3]=0)
while :; do
  for g in 3 2; do
    used="$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [ "${used:-99999}" -lt 1000 ]; then IDLE_STREAK[$g]=$((IDLE_STREAK[$g] + 1)); else IDLE_STREAK[$g]=0; fi
    if [ "${IDLE_STREAK[$g]}" -ge "$IDLE_CHECKS" ]; then
      echo "==== $(date +%F_%T) GPU $g idle for $IDLE_CHECKS checks -> resuming rounds $K0..$((K0+N-1)) there"
      exec "$HERE/run_k3_alternate.sh" "$g" "$K0" "$N"
    fi
  done
  used7="$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  total7="$(nvidia-smi -i 7 --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
  if [ $((total7 - used7)) -ge 61440 ]; then
    echo "==== $(date +%F_%T) GPU 7 has $((total7 - used7)) MiB free -> resuming (shared) rounds $K0..$((K0+N-1))"
    K3_SHARED_GPU=1 exec "$HERE/run_k3_alternate.sh" 7 "$K0" "$N"
  fi
  echo "$(date +%T) GPUs 2/3 busy, GPU 7 free=$((total7 - used7)) MiB -> wait 120s"; sleep 120
done
