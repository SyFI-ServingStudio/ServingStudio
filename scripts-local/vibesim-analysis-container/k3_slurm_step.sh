#!/usr/bin/env bash
# Runs INSIDE a slurm_gpu.sh allocation (DOCKER_GPU_ARG / SLURM_GPU_UUID exported). Two jobs:
#   1. guard against a foreign (non-slurm) tenant on the allocated GPU: slurm does not see direct
#      docker users, so a "free" allocation can still be occupied -> wait up to WAIT_MAX s for
#      memory.used < 1000 MiB, then proceed with a loud warning (the judge's sigma catches noise);
#   2. `--docker <args>`: exec `docker run --rm --name k3m_<jobid> --gpus $DOCKER_GPU_ARG <args>`
#      and remove that container if slurm kills the job (time limit / scancel);
#      otherwise exec "$@" as-is (e.g. the judge).
set -uo pipefail
WAIT_MAX="${K3_GPU_BUSY_WAIT:-300}"
uuid="${SLURM_GPU_UUID%%,*}"
t=0
while :; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$uuid" 2>/dev/null | tr -d ' ')"
  [ -z "$used" ] && { echo "[step] cannot read memory.used of $uuid; continuing"; break; }
  [ "$used" -lt 1000 ] && break
  if [ "$t" -ge "$WAIT_MAX" ]; then echo "[step] GPU_BUSY_WARNING: $uuid has ${used} MiB resident from a non-slurm tenant after ${t}s; proceeding"; break; fi
  echo "[step] $(date +%T) GPU $uuid busy (${used} MiB, non-slurm tenant) -> wait 30s"; sleep 30; t=$((t + 30))
done
if [ "${1:-}" = "--docker" ]; then
  shift
  name="k3m_${SLURM_JOB_ID:-$$}"
  trap 'echo "[step] signal -> removing container $name"; docker rm -f "$name" >/dev/null 2>&1; exit 143' TERM INT
  docker run --rm --name "$name" --gpus "$DOCKER_GPU_ARG" "$@" &
  wait $!; rc=$?
  echo "[step] EXIT_RC=$rc"; exit $rc
fi
"$@"; rc=$?
echo "[step] EXIT_RC=$rc"; exit $rc
