#!/usr/bin/env bash
# Build the B200 before/after/codex images for case vllm-pr-27931 (rms_norm vec8).
# Uses a LOCAL patched Dockerfile (Dockerfile.27931-b200) whose final smoke-test runs from
# /tmp so the installed /opt/venv wheel is imported (the canonical case Dockerfile asserts
# vllm.__file__ under /opt/venv but runs from /workspace/vllm, where the source tree shadows
# the wheel -> false failure). GPU_ARCHITECTURES=10.0 for B200.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH=/raid/yilegu/roofline_guided_agent/roofline_guided_agent_benchmark
CASE=vllm-pr-27931
CASE_DIR="$BENCH/cases/$CASE"
DF="$HERE/Dockerfile.27931-b200"
BEFORE_COMMIT=684f2545851ee0ee49be9a80545ed497324f1a96
AFTER_COMMIT=6c3c0f8235cacce28982687e362b80d953ea7617
BEFORE_TAG=rga-local/vllm-pr-27931:b200-before
AFTER_TAG=rga-local/vllm-pr-27931:b200-after
CODEX_TAG=rga-local/vllm-pr-27931:b200-codex

build_variant () {
  local variant="$1" commit="$2" tag="$3"
  echo "==== building $variant ($commit) -> $tag ===="
  docker buildx build --load --platform linux/amd64 --file "$DF" --tag "$tag" \
    --build-arg CUDA_VERSION=12.9.1 --build-arg PYTHON_VERSION=3.12 \
    --build-arg SOURCE_REPOSITORY=https://github.com/vllm-project/vllm.git \
    --build-arg SOURCE_COMMIT="$commit" \
    --build-arg CASE_ID="$CASE" --build-arg CASE_VARIANT="$variant" \
    --build-arg FRAMEWORK=vllm --build-arg PR_NUMBER=27931 \
    --build-arg PR_URL=https://github.com/vllm-project/vllm/pull/27931 \
    --build-arg PROJECT_REPOSITORY=https://github.com/roofline-agent/roofline_guided_agent_benchmark \
    --build-arg TARGET_PLATFORM=linux/amd64 --build-arg GPU_ARCHITECTURES=10.0 \
    --build-arg MAX_JOBS=64 --build-arg NVCC_THREADS=2 \
    "$CASE_DIR"
}

build_variant before "$BEFORE_COMMIT" "$BEFORE_TAG"
build_variant after  "$AFTER_COMMIT"  "$AFTER_TAG"

echo "==== building CODEX runner (from BEFORE) -> $CODEX_TAG ===="
docker buildx build --load -f "$HERE/Dockerfile.codex-runner" \
  --build-arg BASE_IMAGE="$BEFORE_TAG" -t "$CODEX_TAG" "$HERE"

echo "==== done ===="; docker images | grep 'vllm-pr-27931'
