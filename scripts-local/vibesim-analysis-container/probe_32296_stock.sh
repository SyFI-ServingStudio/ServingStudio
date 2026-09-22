#!/usr/bin/env bash
# 32296 feasibility on the ALREADY-BUILT stock image (no from-source build). The PR touches the
# group-quant JIT kernel, which only fires on the BLOCK-quant fp8 path (weight_block_size set).
# On stock sglang the renamed kernel is sglang.jit_kernel.per_token_group_quant_8bit; mxfp8 sets
# weight_block_size=[1,32] -> block_quant=True, so --quantization mxfp8 on bf16 Qwen3-0.6B should
# route through it on B200. This probe answers, with a MEASUREMENT:
#   (1) does the kernel FIRE under mxfp8 on a small model?  (capture succeeds)
#   (2) what is its ISOLATED latency?  -> compare to the ~8-13us launch floor seen for 28103/27931.
# If the isolated kernel latency is at/under the launch floor, then a PR that halves a sub-step of
# it is UNRESOLVABLE for small models (structural NULL) and building the PR images adds nothing.
# Uses the hardened function-capture (rebind-all-aliases-by-identity) so from-imported call sites
# are caught. Usage (under slurm):  sbatch slurm_gpu.sh ./probe_32296_stock.sh
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# BLOCK-fp8 checkpoint (weight_block_size=[128,128], dynamic act) -> routes through
# w8a8_block_fp8_linear -> sglang_per_token_group_quant_fp8 (the block-fp8 activation quant, the
# python wrapper over the PR's group-quant kernel). This is the ONLY path that fires the PR's
# kernel; mxfp8/per-tensor fp8 do not. Target the wrapper: it always fires (fp8_utils.py:555),
# yields real group_size=128 inputs, and is the callable an optimizer would actually target.
SNAP=/raid/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/snapshots/8591804019c8b22094c3b5b4454e0edc05dffc98
IMG="${SGLANG_IMG:-lmsysorg/sglang:latest}"
TARGET="sglang.srt.layers.quantization.fp8_kernel:sglang_per_token_group_quant_fp8"
QUANT="${QUANT:-fp8}"
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
GOLDEN=/raid/yilegu/eval_goldens/probe_32296_stock.pt
OUT="$HERE/probe_32296_stock"; mkdir -p "$OUT" /raid/yilegu/eval_goldens
CNAME="p32stk_$(date +%s%N)"

echo "==== 32296 stock probe: img=$IMG target=$TARGET quant=$QUANT gpu=$GPU ===="
docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run -d --name "$CNAME" --gpus "$GPU" \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -v /raid/yilegu/models:/raid/yilegu/models:ro \
  -v /raid/hf/hub:/raid/hf/hub:ro \
  -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer \
  -v /raid/yilegu/eval_goldens:/raid/yilegu/eval_goldens \
  --workdir /tmp --entrypoint sleep "$IMG" infinity >/dev/null
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker cp "$HERE/sglang_profile_patch.py" "$CNAME:/tmp/sglang_profile_patch.py" >/dev/null
docker cp "$HERE/extract_and_profile.py" "$CNAME:/tmp/extract_and_profile.py" >/dev/null
docker exec -i "$CNAME" python3 - <<'PY' >/dev/null
import os, sglang
sched = os.path.join(os.path.dirname(sglang.__file__), "srt", "managers", "scheduler.py")
marker = "# --- ep capture patch ---"; src = open(sched).read()
if marker not in src:
    open(sched, "a").write(f"\n\n{marker}\ntry:\n import sys as _s\n if '/tmp' not in _s.path: _s.path.insert(0,'/tmp')\n import sglang_profile_patch as _e; _e.attach(Scheduler)\nexcept Exception as _x:\n import logging; logging.getLogger(__name__).warning('ep patch failed: %s', _x)\n")
PY

echo "-------------------- CAPTURE (mxfp8, group-quant should fire) --------------------"
docker exec --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 "$CNAME" \
  python3 /tmp/extract_and_profile.py --model "$SNAP" --framework sglang \
  --target "$TARGET" --target-kind function --sglang-quant "$QUANT" \
  --tokens 128 --iters 2000 --capture "$GOLDEN" 2>&1 \
  | tee "$OUT/capture.log" | grep -E "CAPTURE|CHECK|JSON|saved|never fired|Error|Traceback|assert" || true
echo "-------------------- REPLAY (isolated latency sanity) --------------------"
docker exec --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 "$CNAME" \
  python3 /tmp/extract_and_profile.py --model "$SNAP" --framework sglang \
  --target "$TARGET" --target-kind function --sglang-quant "$QUANT" \
  --replay "$GOLDEN" --iters 3000 2>&1 \
  | tee "$OUT/replay.log" | grep -E "REPLAY|CHECK|JSON|never fired|Error|Traceback" || true
echo "==== done -> $OUT ===="
