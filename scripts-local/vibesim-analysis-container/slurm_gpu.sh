#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --job-name=vibesim-gpu
#SBATCH --output=%x-%j.out
# Generic slurm GPU wrapper for VibeSim repro / eval GPU work.
# Requests ONE GPU on partition `main` (user override for the Kimi-K3 task; preemptible was
# stuck on QOSGrpGRES since it can't preempt the long-partition jobs). slurm's cgroup constrains
# this job to that
# device. We resolve the allocated GPU by UUID (cgroup-scoped, so `nvidia-smi` only
# lists what we were granted) and export it as DOCKER_GPU_ARG so the docker-based
# steps bind to exactly the slurm-assigned GPU instead of a hardcoded index.
#
# Usage:
#   sbatch slurm_gpu.sh <command> [args...]
# The command runs on the allocated node with:
#   - CUDA_VISIBLE_DEVICES : set by slurm (cgroup-relative)
#   - DOCKER_GPU_ARG       : '"device=<uuid>"' for `docker run --gpus "$DOCKER_GPU_ARG"'
#   - SLURM_GPU_UUID       : the raw UUID(s)
set -Eeuo pipefail

UUIDS="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
if [ -z "$UUIDS" ]; then echo "!! no GPU visible in this allocation" >&2; exit 1; fi
export SLURM_GPU_UUID="$UUIDS"
export DOCKER_GPU_ARG="\"device=${UUIDS}\""

echo "==== slurm_gpu.sh on $(hostname) job=${SLURM_JOB_ID:-?} ===="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "allocated GPU UUID(s): $UUIDS"
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader || true
echo "==== exec: $* ===="
exec "$@"
