#!/usr/bin/env bash
# Build the B200 AFTER image for sglang-pr-20469 (Triton causal_conv1d avoids a .contiguous()
# copy for NON-contiguous conv input), then DERIVE the BEFORE image by swapping the single
# pure-python file causal_conv1d.py + clearing its __pycache__.
#
# Why one build, not two: the PR's ONLY functional change is one pure-python module
# (python/sglang/srt/layers/attention/mamba/causal_conv1d.py). sglang is installed EDITABLE, so
# the .py in the source tree is what runs. Build the expensive full sglang once at merge (AFTER),
# then produce BEFORE by overwriting that one .py with kernels_20469/causal_conv1d.before.py and
# dropping compiled __pycache__. Host-CPU work (buildx + docker commit); no GPU needed.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH=/raid/yilegu/roofline_guided_agent/roofline_guided_agent_benchmark
CASE=sglang-pr-20469
CASE_DIR="$BENCH/cases/$CASE"
DF="$CASE_DIR/Dockerfile.9.0"            # arch set via GPU_ARCHITECTURES build-arg, not the name
AFTER_COMMIT=87549f8f0bf9e3514c1f6d30ba79844163b9221d   # PR merge = AFTER (the fix)
BASE_COMMIT=db995fba4790c9010af1aeff6022569504706f17     # provenance only (BEFORE derived by .py swap)
AFTER_TAG=rga-local/sglang-pr-20469:b200-after
BEFORE_TAG=rga-local/sglang-pr-20469:b200-before
# In-image path of the editable-install pure-python module. Verified against the built image below.
PY=/sgl-workspace/sglang/python/sglang/srt/layers/attention/mamba/causal_conv1d.py
LOG="$HERE/build_20469_after.log"

echo "==== [1/4] buildx AFTER ($AFTER_COMMIT) -> $AFTER_TAG  (log: $LOG) ===="
docker buildx build --load --platform linux/amd64 --file "$DF" --tag "$AFTER_TAG" \
  --build-arg SOURCE_REPOSITORY=https://github.com/sgl-project/sglang.git \
  --build-arg SOURCE_COMMIT="$AFTER_COMMIT" \
  --build-arg CASE_ID="$CASE" --build-arg CASE_VARIANT=after \
  --build-arg FRAMEWORK=sglang --build-arg PR_NUMBER=20469 \
  --build-arg PR_URL=https://github.com/sgl-project/sglang/pull/20469 \
  --build-arg PROJECT_REPOSITORY=https://github.com/roofline-agent/roofline_guided_agent_benchmark \
  --build-arg TARGET_PLATFORM=linux/amd64 --build-arg GPU_ARCHITECTURES=10.0 \
  --build-arg MAX_JOBS=64 --build-arg NVCC_THREADS=2 \
  "$CASE_DIR" 2>&1 | tee "$LOG"

echo "==== [2/4] verify AFTER image ships the merge causal_conv1d.py ===="
CN="derive20469_$(date +%s)"
docker rm -f "$CN" >/dev/null 2>&1 || true
docker run -d --name "$CN" --entrypoint sleep "$AFTER_TAG" 3600 >/dev/null
cleanup() { docker rm -f "$CN" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker exec "$CN" test -f "$PY" || { echo "FATAL: $PY not found in AFTER image"; \
  docker exec "$CN" bash -lc 'find /sgl-workspace/sglang -name causal_conv1d.py'; exit 3; }
docker cp "$CN:$PY" "$HERE/_after_installed.py"
if diff -q "$HERE/_after_installed.py" "$HERE/kernels_20469/causal_conv1d.after.py" >/dev/null; then
  echo "OK: AFTER installed .py == kernels_20469/causal_conv1d.after.py"
else
  echo "WARN: AFTER installed .py differs from saved after.py -- showing diff (installed<vs>saved):"
  diff "$HERE/_after_installed.py" "$HERE/kernels_20469/causal_conv1d.after.py" | head -60 || true
fi

echo "==== [3/4] derive BEFORE: overwrite .py with base + drop __pycache__, commit ===="
docker cp "$HERE/kernels_20469/causal_conv1d.before.py" "$CN:$PY"
docker exec "$CN" bash -lc 'find /sgl-workspace/sglang -path "*mamba*__pycache__*causal_conv1d*" -delete 2>/dev/null; \
  find /sgl-workspace/sglang -name "*.pyc" -path "*mamba*" -delete 2>/dev/null; true'
docker commit --change="LABEL io.roofline-agent.variant=before io.roofline-agent.derived-from=$AFTER_COMMIT io.roofline-agent.base-commit=$BASE_COMMIT" \
  "$CN" "$BEFORE_TAG" >/dev/null
echo "committed $BEFORE_TAG"

echo "==== [4/4] confirm BEFORE .py == base AND differs from AFTER ===="
CNB="verify20469_$(date +%s)"
docker rm -f "$CNB" >/dev/null 2>&1 || true
docker run -d --name "$CNB" --entrypoint sleep "$BEFORE_TAG" 600 >/dev/null
docker cp "$CNB:$PY" "$HERE/_before_installed.py"
docker rm -f "$CNB" >/dev/null 2>&1 || true
if diff -q "$HERE/_before_installed.py" "$HERE/kernels_20469/causal_conv1d.before.py" >/dev/null; then
  echo "OK: BEFORE installed .py == before.py"
else
  echo "FATAL: BEFORE .py != before.py"; exit 4
fi
if diff -q "$HERE/_before_installed.py" "$HERE/_after_installed.py" >/dev/null; then
  echo "FATAL: BEFORE and AFTER .py are IDENTICAL -- swap failed"; exit 5
else
  echo "OK: BEFORE and AFTER installed .py DIFFER ($(diff "$HERE/_before_installed.py" "$HERE/_after_installed.py" | grep -c '^<') lines changed)"
fi
echo "==== done: $AFTER_TAG + $BEFORE_TAG ===="
docker images | grep 'sglang-pr-20469' || true
