#!/usr/bin/env bash
# Build one before/after repro image for a case, ported to B200 (SM 10.0).
# Usage: build_repro_image.sh <case_id> <variant> <commit> <tag> [MAX_JOBS] [NVCC_THREADS]
#
# vLLM's setup.py sets ninja jobs = MAX_JOBS // NVCC_THREADS. On this 224-core box
# the case default (16 // 8 = 2) is far too slow; MAX_JOBS=64 NVCC_THREADS=2 -> -j32.
set -Eeuo pipefail
BENCH=/raid/yilegu/roofline_guided_agent/roofline_guided_agent_benchmark
CASE_ID="$1"; VARIANT="$2"; COMMIT="$3"; TAG="$4"
MAX_JOBS="${5:-64}"; NVCC_THREADS="${6:-2}"
CASE_DIR="$BENCH/cases/$CASE_ID"

cd "$BENCH"
exec docker buildx build --load \
  --platform linux/amd64 \
  --file "$CASE_DIR/Dockerfile.9.0" \
  --tag "$TAG" \
  --build-arg CUDA_VERSION=12.9.1 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg SOURCE_REPOSITORY=https://github.com/vllm-project/vllm.git \
  --build-arg SOURCE_COMMIT="$COMMIT" \
  --build-arg CASE_ID="$CASE_ID" --build-arg CASE_VARIANT="$VARIANT" \
  --build-arg FRAMEWORK=vllm --build-arg PR_NUMBER=28103 \
  --build-arg PR_URL=https://github.com/vllm-project/vllm/pull/28103 \
  --build-arg PROJECT_REPOSITORY=https://github.com/roofline-agent/roofline_guided_agent_benchmark \
  --build-arg TARGET_PLATFORM=linux/amd64 \
  --build-arg GPU_ARCHITECTURES=10.0 \
  --build-arg MAX_JOBS="$MAX_JOBS" --build-arg NVCC_THREADS="$NVCC_THREADS" \
  "$CASE_DIR"
