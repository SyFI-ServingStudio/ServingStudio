#!/usr/bin/env bash
# Feasibility probe for sglang-pr-32296 (per_token_group_quant sanitization halving) AND the
# end-to-end validation of the new FUNCTION-target path (extract_and_profile.py kind=function +
# sglang_profile_patch.py ep_* function routing).
#
# The PR edits a JIT .cuh; the production python callable is
#   sglang.kernels.ops.quantization.per_token_group_quant:per_token_group_quant
# Two unknowns before we build before/after images:
#   (1) does stock lmsysorg/sglang:latest still expose that module path?  -> import check
#   (2) does the quantizer FIRE under Qwen3-0.6B + --quantization fp8?     -> capture check
# This probe answers both on stock sglang (no image build), capturing the callable's real
# inputs. If capture succeeds we have proof the function target works AND that a small-model fp8
# run reaches the kernel -> 32296 becomes buildable as a JIT-.cuh-swap case.
#
# Usage (under slurm):  sbatch slurm_gpu.sh ./probe_32296.sh
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SNAP=/raid/yilegu/models/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
IMG="${SGLANG_IMG:-lmsysorg/sglang:latest}"
TARGET="sglang.kernels.ops.quantization.per_token_group_quant:per_token_group_quant"
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
GOLDEN=/raid/yilegu/eval_goldens/probe_32296_ptgq.pt
OUT="$HERE/probe_32296"; mkdir -p "$OUT" /raid/yilegu/eval_goldens
CNAME="probe32296_$(date +%s%N)"

echo "==== 32296 probe: img=$IMG target=$TARGET quant=fp8 gpu=$GPU ===="
docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run -d --name "$CNAME" --gpus "$GPU" \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -v /raid/yilegu/models:/raid/yilegu/models:ro \
  -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer \
  -v /raid/yilegu/eval_goldens:/raid/yilegu/eval_goldens \
  --workdir /tmp --entrypoint sleep "$IMG" infinity >/dev/null
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker cp "$HERE/sglang_profile_patch.py" "$CNAME:/tmp/sglang_profile_patch.py" >/dev/null
docker cp "$HERE/extract_and_profile.py" "$CNAME:/tmp/extract_and_profile.py" >/dev/null

# unknown (1): does the module path resolve in stock sglang? cheap import check, print alternatives.
echo "-------------------- MODULE PATH CHECK --------------------"
docker exec -i "$CNAME" python3 - <<'PY' 2>&1 | tee "$OUT/modpath.log" || true
import importlib
spec = "sglang.kernels.ops.quantization.per_token_group_quant:per_token_group_quant"
mod, attr = spec.split(":")
try:
    m = importlib.import_module(mod)
    print("IMPORT_OK", mod, "has", attr, "=", hasattr(m, attr))
except Exception as e:
    print("IMPORT_FAIL", mod, "->", repr(e))
    # scan for the callable's real home so we can retarget
    import pkgutil, sglang
    for sm in pkgutil.walk_packages(sglang.__path__, "sglang."):
        if "per_token_group_quant" in sm.name:
            print("  candidate module:", sm.name)
PY

# bake the ep patch onto the scheduler subprocess (same recipe as validate_sglang_capture.sh)
docker exec -i "$CNAME" python3 - <<'PY' >/dev/null
import os, sglang
sched = os.path.join(os.path.dirname(sglang.__file__), "srt", "managers", "scheduler.py")
marker = "# --- ep capture patch ---"; src = open(sched).read()
if marker not in src:
    open(sched, "a").write(f"\n\n{marker}\ntry:\n import sys as _s\n if '/tmp' not in _s.path: _s.path.insert(0,'/tmp')\n import sglang_profile_patch as _e; _e.attach(Scheduler)\nexcept Exception as _x:\n import logging; logging.getLogger(__name__).warning('ep patch failed: %s', _x)\n")
PY

# unknown (2): does the callable FIRE under fp8? capture its real inputs.
echo "-------------------- CAPTURE (function target, fp8) --------------------"
docker exec --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 "$CNAME" \
  python3 /tmp/extract_and_profile.py --model "$SNAP" --framework sglang \
  --target "$TARGET" --target-kind function --sglang-quant fp8 \
  --tokens 8 --iters 2000 --capture "$GOLDEN" 2>&1 \
  | tee "$OUT/capture.log" | grep -E "CAPTURE|CHECK|JSON|saved|never fired|Error|Traceback" || true
echo "==== done -> $OUT (see modpath.log + capture.log) ===="
