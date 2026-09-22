#!/usr/bin/env bash
# Validate the sglang capture mechanism (extract_and_profile.py --framework sglang +
# sglang_profile_patch.py) end-to-end on a stock sglang image + local Qwen3-0.6B.
#
# Proves: (1) the patch attaches ep_install/ep_time/ep_replay onto the scheduler subprocess,
# (2) collective_rpc reaches the model, (3) a real forward captures a target module's inputs,
# (4) standalone CUDA timing + golden replay + correctness all round-trip.
#
# Target = model.layers.0.input_layernorm (RMSNorm, always fires) -- mechanism check, not a PR.
# Runs the extractor INSIDE the sglang image; binds the GPU from DOCKER_GPU_ARG (slurm_gpu.sh).
# Usage (under slurm):  sbatch slurm_gpu.sh ./validate_sglang_capture.sh
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SNAP=/raid/yilegu/models/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
IMG="${SGLANG_IMG:-lmsysorg/sglang:latest}"
TARGET=model.layers.0.input_layernorm
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
GOLDEN=/raid/yilegu/eval_goldens/sglang_mech_input_layernorm.pt
OUT="$HERE/sglang_capture_validate"; mkdir -p "$OUT" /raid/yilegu/eval_goldens
CNAME="sgl_capval_$(date +%s%N)"

echo "==== sglang capture validation: img=$IMG target=$TARGET gpu=$GPU ===="
docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run -d --name "$CNAME" --gpus "$GPU" \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -v /raid/yilegu/models:/raid/yilegu/models:ro \
  -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer \
  -v /raid/yilegu/eval_goldens:/raid/yilegu/eval_goldens \
  --workdir /tmp --entrypoint sleep "$IMG" infinity >/dev/null

cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# 1) stage the patch + extractor, attach the profiling methods onto Scheduler at import time.
docker cp "$HERE/sglang_profile_patch.py" "$CNAME:/tmp/sglang_profile_patch.py" >/dev/null
docker cp "$HERE/extract_and_profile.py" "$CNAME:/tmp/extract_and_profile.py" >/dev/null
docker exec -i "$CNAME" python3 - <<'PY'
import os, sglang
sp = os.path.dirname(sglang.__file__)               # the sglang pkg the subprocess also imports
sched = os.path.join(sp, "srt", "managers", "scheduler.py")
marker = "# --- ep capture patch ---"
src = open(sched).read()
if marker not in src:
    # /tmp/sglang_profile_patch.py is staged in the container; prepend /tmp so the spawned
    # scheduler subprocess can import it regardless of how sglang itself is installed
    # (editable /sgl-workspace vs dist-packages).
    with open(sched, "a") as f:
        f.write(f"\n\n{marker}\n"
                "try:\n"
                "    import sys as _sys\n"
                "    if '/tmp' not in _sys.path: _sys.path.insert(0, '/tmp')\n"
                "    import sglang_profile_patch as _epp\n"
                "    _epp.attach(Scheduler)\n"
                "    import logging; logging.getLogger(__name__).warning('ep patch attached: %s', hasattr(Scheduler, 'ep_install'))\n"
                "except Exception as _e:\n"
                "    import logging; logging.getLogger(__name__).warning('ep patch failed: %s', _e)\n")
    print(f"appended ep patch to {sched}")
else:
    print("ep patch already present")
PY

run() { docker exec --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 "$CNAME" \
        python3 /tmp/extract_and_profile.py --model "$SNAP" --framework sglang --target "$TARGET" "$@"; }

echo "==================== CAPTURE (golden) ===================="
run --tokens 8 --iters 3000 --capture "$GOLDEN" 2>&1 | tee "$OUT/capture.log" | grep -E "CAPTURE|CHECK|JSON|saved|Error|Traceback" || true
echo "==================== REPLAY (correctness + timing) ===================="
run --tokens 8 --iters 3000 --replay "$GOLDEN" 2>&1 | tee "$OUT/replay.log" | grep -E "REPLAY|CHECK|JSON|Error|Traceback" || true
echo "==== done -> $OUT ===="
