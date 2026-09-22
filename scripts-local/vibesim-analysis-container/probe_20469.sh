#!/usr/bin/env bash
# Gate + floor for sglang-pr-20469. The PR's win exists ONLY if causal_conv1d_fn is invoked with
#   (1) a NON-contiguous x (x.stride(-1) != 1)  AND  (2) seq_lens_cpu present in kwargs.
# So before trusting a floor number we CAPTURE on the BEFORE image and inspect the golden:
#   - x contiguity (from input_meta and the packed stride)
#   - whether seq_lens_cpu is a kwarg
# If both hold, the before/after latency gap = the cost of the eliminated .contiguous() copy and
# is meaningful. If x is contiguous or seq_lens_cpu is absent, both code paths are identical =>
# NULL-by-construction (report that, skip the floor as vacuous).
# Then run the standard per-case floor measurement.
#
# Usage (under slurm):  sbatch slurm_gpu.sh ./probe_20469.sh
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SNAP=/raid/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
IMG=rga-local/sglang-pr-20469:b200-before
TARGET="sglang.srt.layers.attention.mamba.causal_conv1d:causal_conv1d_fn"
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
GOLDEN_DIR=/raid/yilegu/eval_goldens; mkdir -p "$GOLDEN_DIR"
GOLDEN="$GOLDEN_DIR/probe_20469_conv.pt"
OUT="$HERE/probe_20469"; mkdir -p "$OUT"
CNAME="probe20469_$(date +%s%N)"
TOKENS="${TOKENS:-8192}"

echo "==== 20469 GATE: capture causal_conv1d_fn on BEFORE, inspect x contiguity + seq_lens_cpu ===="
docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run -d --name "$CNAME" --gpus "$GPU" \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -v /raid/hf/hub:/raid/hf/hub:ro \
  -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer \
  -v "$GOLDEN_DIR":"$GOLDEN_DIR" \
  --workdir /tmp --entrypoint sleep "$IMG" infinity >/dev/null
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker cp "$HERE/sglang_profile_patch.py" "$CNAME:/tmp/sglang_profile_patch.py" >/dev/null
docker cp "$HERE/extract_and_profile.py" "$CNAME:/tmp/extract_and_profile.py" >/dev/null
docker exec -i "$CNAME" python3 - <<'PY' >/dev/null
import os, sglang
sched=os.path.join(os.path.dirname(sglang.__file__),"srt","managers","scheduler.py")
m="# --- ep capture patch ---"; src=open(sched).read()
if m not in src:
    open(sched,"a").write("\n\n"+m+"\ntry:\n import sys as _s\n if '/tmp' not in _s.path: _s.path.insert(0,'/tmp')\n import sglang_profile_patch as _e; _e.attach(Scheduler)\nexcept Exception as _x:\n import logging; logging.getLogger(__name__).warning('ep patch failed: %s', _x)\n")
PY

echo "-------------------- CAPTURE (tokens=$TOKENS) --------------------"
docker exec --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 "$CNAME" \
  python3 /tmp/extract_and_profile.py --model "$SNAP" --framework sglang \
  --target "$TARGET" --target-kind function --tokens "$TOKENS" --iters 500 \
  --capture "$GOLDEN" 2>&1 | tee "$OUT/capture.log" \
  | grep -E "CAPTURE|input_meta|never fired|Error|Traceback|RuntimeError" || true

echo "-------------------- GOLDEN INSPECT --------------------"
docker exec --workdir /tmp "$CNAME" python3 - "$GOLDEN" <<'PY' 2>&1 | tee "$OUT/inspect.log"
import torch, sys
g = torch.load(sys.argv[1], weights_only=False)
def meta(d):
    if isinstance(d, dict) and d.get("__t__"):
        v = torch.as_strided(d["flat"], d["shape"], d["stride"], d["off"])
        return dict(shape=list(d["shape"]), stride=list(d["stride"]),
                    contiguous=v.is_contiguous(), last_stride=d["stride"][-1] if d["stride"] else None,
                    dtype=str(d["flat"].dtype))
    return type(d).__name__
print("ARGS:")
for i,a in enumerate(g["args"]):
    print(f"  arg[{i}] =", meta(a))
print("KWARGS keys:", list(g["kwargs"].keys()))
for k,v in g["kwargs"].items():
    print(f"  kwarg {k!r} =", meta(v) if isinstance(v,dict) else (v if not torch.is_tensor(v) else "tensor"))
x = g["args"][0] if g["args"] else None
xc = None
if isinstance(x, dict) and x.get("__t__"):
    xc = (x["stride"][-1] == 1)
has_slc = "seq_lens_cpu" in g["kwargs"]
print("---- VERDICT ----")
print("x_contiguous_lastdim =", xc, "(win needs FALSE)")
print("seq_lens_cpu_in_kwargs =", has_slc, "(win needs TRUE)")
win = (xc is False) and has_slc
print("WIN_CAN_MANIFEST =", win)
PY

echo "==== GATE done. If WIN_CAN_MANIFEST=True, running floor measurement ===="
docker rm -f "$CNAME" >/dev/null 2>&1 || true
trap - EXIT
if grep -q "WIN_CAN_MANIFEST = True" "$OUT/inspect.log"; then
  echo "==== running measure_floor.sh issue_20469.json ===="
  REPS="${REPS:-5}" MARGIN="${MARGIN:-3}" "$HERE/measure_floor.sh" issue_20469.json
else
  echo "==== NULL-BY-CONSTRUCTION: win cannot manifest (x contiguous or seq_lens_cpu absent) -- floor is vacuous ===="
fi
echo "==== probe_20469 done -> $OUT ===="
