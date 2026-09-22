#!/usr/bin/env bash
# Broad diagnostic: does ANY group-quant callable fire on Qwen3-0.6B under fp8/mxfp8 on stock
# sglang? Wraps every sys.modules callable whose name contains a group-quant needle, runs a
# forward, and reports which fired + their input shapes. Decides whether 32296's kernel is
# reachable at all on a small dense model (vs. only in the block-fp8/deep_gemm MoE regime).
# Usage (under slurm):  QUANT=mxfp8 sbatch slurm_gpu.sh ./probe_32296_scan.sh
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# Qwen3-4B-Instruct-2507-FP8 is a genuine BLOCK-fp8 checkpoint (weight_block_size=[128,128],
# activation_scheme=dynamic) -> routes through w8a8_block_fp8_linear -> sglang_per_token_group_quant_fp8
# -> the PR's group-quant kernel, on a 4B dense model that fits one GPU. mxfp8 on a bf16 model
# uses the microscaling path (mxfp8_group_quantize) and NEVER touches per_token_group_quant.
SNAP=/raid/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/snapshots/8591804019c8b22094c3b5b4454e0edc05dffc98
IMG="${SGLANG_IMG:-lmsysorg/sglang:latest}"
QUANT="${QUANT:-fp8}"
NEEDLE="${NEEDLE:-per_token_group_quant,group_quant,quant_fp8,quant_8bit}"
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
OUT="$HERE/probe_32296_scan"; mkdir -p "$OUT"
CNAME="p32scan_$(date +%s%N)"

echo "==== 32296 scan: img=$IMG quant=$QUANT needle=$NEEDLE gpu=$GPU ===="
docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run -d --name "$CNAME" --gpus "$GPU" \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -v /raid/yilegu/models:/raid/yilegu/models:ro \
  -v /raid/hf/hub:/raid/hf/hub:ro \
  -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer \
  --workdir /tmp --entrypoint sleep "$IMG" infinity >/dev/null
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker cp "$HERE/sglang_profile_patch.py" "$CNAME:/tmp/sglang_profile_patch.py" >/dev/null
docker exec -i "$CNAME" python3 - <<'PY' >/dev/null
import os, sglang
sched = os.path.join(os.path.dirname(sglang.__file__), "srt", "managers", "scheduler.py")
marker = "# --- ep capture patch ---"; src = open(sched).read()
if marker not in src:
    open(sched, "a").write(f"\n\n{marker}\ntry:\n import sys as _s\n if '/tmp' not in _s.path: _s.path.insert(0,'/tmp')\n import sglang_profile_patch as _e; _e.attach(Scheduler)\nexcept Exception as _x:\n import logging; logging.getLogger(__name__).warning('ep patch failed: %s', _x)\n")
PY

# sglang's Engine uses multiprocessing 'spawn', which re-execs the MAIN module by path -- a
# stdin heredoc yields '/tmp/<stdin>' (FileNotFoundError). So the driver must be a real FILE.
cat > "$OUT/scan_main.py" <<'PY'
import os, json
from sglang.srt.entrypoints.engine import Engine
RES="/tmp/_scan.json"
eng=Engine(model_path=os.environ["SNAP"], dtype="bfloat16", tp_size=1, mem_fraction_static=0.55,
           disable_cuda_graph=True, trust_remote_code=True, quantization=os.environ["QUANT"])
try:
    eng.collective_rpc("ep_scan_install", needle=os.environ["NEEDLE"])
    eng.generate(input_ids=[list(range(1,9))], sampling_params={"max_new_tokens":4,"temperature":0.0})
    eng.collective_rpc("ep_scan_report", out_path=RES)
finally:
    eng.shutdown()
d=json.load(open(RES))
print("SCAN patched", d["patched"], "callables")
if not d["fired"]:
    print("SCAN fired: NONE -- no group-quant callable is invoked on this model/quant")
for name,info in sorted(d["fired"].items(), key=lambda kv:-kv[1]["calls"]):
    print(f"SCAN fired: {name}  calls={info['calls']}  first_shapes={info['shapes']}")
PY
docker cp "$OUT/scan_main.py" "$CNAME:/tmp/scan_main.py" >/dev/null
docker exec -i --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -e SNAP="$SNAP" -e QUANT="$QUANT" -e NEEDLE="$NEEDLE" "$CNAME" python3 /tmp/scan_main.py 2>&1 \
  | tee "$OUT/scan.log" | grep -E "SCAN|fired|patched|Error|Traceback|assert" || true
echo "==== done -> $OUT/scan.log ===="
