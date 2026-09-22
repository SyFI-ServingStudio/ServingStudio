#!/usr/bin/env bash
# Build the B200 AFTER image for sglang-pr-32296 (per_token_group_quant non-finite sanitization
# halving), then DERIVE the BEFORE image by swapping the single JIT .cuh + clearing the JIT cache.
#
# Why one build, not two: the PR's ONLY functional change is a runtime-JIT-compiled CUDA header
# (python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh). sglang is installed EDITABLE,
# so the .cuh in the source tree is what gets JIT-compiled at first kernel launch. So we build the
# expensive full sglang once at the merge (AFTER) commit, then produce BEFORE by overwriting that
# one .cuh with kernels_32296/per_token_group_quant.base.cuh and clearing any cached compiled
# artifacts. This is host-CPU work (buildx + docker commit); no GPU needed.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH=/raid/yilegu/roofline_guided_agent/roofline_guided_agent_benchmark
CASE=sglang-pr-32296
CASE_DIR="$BENCH/cases/$CASE"
DF="$CASE_DIR/Dockerfile.9.0"            # arch is set via GPU_ARCHITECTURES build-arg, not the name
AFTER_COMMIT=9402012f0f19857ed9a5a0922dafebaf6382a0e3   # PR merge = AFTER
BASE_COMMIT=d059b0f56eaa44dd0edc34318316258b723e1fa4     # provenance only (BEFORE derived by .cuh swap)
AFTER_TAG=rga-local/sglang-pr-32296:b200-after
BEFORE_TAG=rga-local/sglang-pr-32296:b200-before
# In-image path of the JIT .cuh (editable source tree). Verified against the built image below.
CUH=/sgl-workspace/sglang/python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh
LOG="$HERE/build_32296_after.log"

echo "==== [1/4] buildx AFTER ($AFTER_COMMIT) -> $AFTER_TAG  (log: $LOG) ===="
docker buildx build --load --platform linux/amd64 --file "$DF" --tag "$AFTER_TAG" \
  --build-arg CUDA_VERSION=13.0.1 --build-arg PYTHON_VERSION=3.12 \
  --build-arg SOURCE_REPOSITORY=https://github.com/sgl-project/sglang.git \
  --build-arg SOURCE_COMMIT="$AFTER_COMMIT" \
  --build-arg CASE_ID="$CASE" --build-arg CASE_VARIANT=after \
  --build-arg FRAMEWORK=sglang --build-arg PR_NUMBER=32296 \
  --build-arg PR_URL=https://github.com/sgl-project/sglang/pull/32296 \
  --build-arg PROJECT_REPOSITORY=https://github.com/roofline-agent/roofline_guided_agent_benchmark \
  --build-arg TARGET_PLATFORM=linux/amd64 --build-arg GPU_ARCHITECTURES=10.0 \
  --build-arg MAX_JOBS=64 --build-arg NVCC_THREADS=2 \
  "$CASE_DIR" 2>&1 | tee "$LOG"

echo "==== [2/4] verify AFTER image ships the merge .cuh ===="
CN="derive32296_$(date +%s)"
docker rm -f "$CN" >/dev/null 2>&1 || true
docker run -d --name "$CN" --entrypoint sleep "$AFTER_TAG" 3600 >/dev/null
cleanup() { docker rm -f "$CN" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker exec "$CN" test -f "$CUH" || { echo "FATAL: $CUH not found in AFTER image"; \
  docker exec "$CN" bash -lc 'find /sgl-workspace/sglang -name per_token_group_quant.cuh'; exit 3; }
docker cp "$CN:$CUH" "$HERE/_after_installed.cuh"
if diff -q "$HERE/_after_installed.cuh" "$HERE/kernels_32296/per_token_group_quant.merge.cuh" >/dev/null; then
  echo "OK: AFTER installed .cuh == kernels_32296/per_token_group_quant.merge.cuh"
else
  echo "WARN: AFTER installed .cuh differs from saved merge.cuh -- showing diff (installed<vs>saved):"
  diff "$HERE/_after_installed.cuh" "$HERE/kernels_32296/per_token_group_quant.merge.cuh" | head -40 || true
fi

echo "==== [3/4] derive BEFORE: overwrite .cuh with base + clear JIT cache, commit ===="
docker cp "$HERE/kernels_32296/per_token_group_quant.base.cuh" "$CN:$CUH"
# Clear any cached JIT-compiled artifacts so BEFORE recompiles from the swapped .cuh.
docker exec "$CN" bash -lc 'rm -rf /root/.cache/flashinfer /root/.cache/sglang /root/.cache/torch_extensions \
  /tmp/sglang_jit* 2>/dev/null; find / -type d -name "*jit*cache*" 2>/dev/null | head; true'
docker commit --change="LABEL io.roofline-agent.variant=before io.roofline-agent.derived-from=$AFTER_COMMIT io.roofline-agent.base-commit=$BASE_COMMIT" \
  "$CN" "$BEFORE_TAG" >/dev/null
echo "committed $BEFORE_TAG"

echo "==== [4/4] confirm BEFORE .cuh == base AND differs from AFTER ===="
CNB="verify32296_$(date +%s)"
docker rm -f "$CNB" >/dev/null 2>&1 || true
docker run -d --name "$CNB" --entrypoint sleep "$BEFORE_TAG" 600 >/dev/null
docker cp "$CNB:$CUH" "$HERE/_before_installed.cuh"
docker rm -f "$CNB" >/dev/null 2>&1 || true
if diff -q "$HERE/_before_installed.cuh" "$HERE/kernels_32296/per_token_group_quant.base.cuh" >/dev/null; then
  echo "OK: BEFORE installed .cuh == base.cuh"
else
  echo "FATAL: BEFORE .cuh != base.cuh"; exit 4
fi
if diff -q "$HERE/_before_installed.cuh" "$HERE/_after_installed.cuh" >/dev/null; then
  echo "FATAL: BEFORE and AFTER .cuh are IDENTICAL -- swap failed"; exit 5
else
  echo "OK: BEFORE and AFTER installed .cuh DIFFER ($(diff "$HERE/_before_installed.cuh" "$HERE/_after_installed.cuh" | grep -c '^<') lines changed)"
fi
echo "==== done: $AFTER_TAG + $BEFORE_TAG ===="
docker images | grep 'sglang-pr-32296' || true
