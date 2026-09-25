#!/usr/bin/env bash
# Print the index of the first GPU that stays idle (memory.used < 1000 MiB) for CHECKS consecutive readings
# INTERVAL seconds apart. User policy 2026-09-25: any free GPU may be used opportunistically; never one with a
# live tenant. Usage: pick_free_gpu.sh [exclude_list e.g. "2 7"]   env: K3_GPU_CANDIDATES="0 1 2 3 4 5 6 7"
#        CHECKS=3 INTERVAL=20 ; blocks until one is found.
set -uo pipefail
EXCL=" ${1:-} "; CANDS="${K3_GPU_CANDIDATES:-0 1 2 3 4 5 6 7}"; CHECKS="${CHECKS:-3}"; INTERVAL="${INTERVAL:-20}"
# GPUs bound to OUR running containers (trial agents think for minutes with 0 MiB resident, so memory alone
# is not enough): read the device requests of every running container and exclude those indices.
held () { docker ps -q 2>/dev/null | xargs -r docker inspect -f '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}} {{end}}{{end}}' 2>/dev/null | tr -d '"' | tr '\n' ' '; }
declare -A streak
while :; do
  H=" $(held) "
  for g in $CANDS; do
    case "$EXCL" in *" $g "*) continue;; esac
    case "$H" in *" $g "*) streak[$g]=0; continue;; esac
    used="$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [ -n "$used" ] && [ "$used" -lt 1000 ]; then streak[$g]=$(( ${streak[$g]:-0} + 1 )); else streak[$g]=0; fi
    if [ "${streak[$g]}" -ge "$CHECKS" ]; then echo "$g"; exit 0; fi
  done
  sleep "$INTERVAL"
done
