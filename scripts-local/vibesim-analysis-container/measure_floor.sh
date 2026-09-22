#!/usr/bin/env bash
# Per-case launch-floor + resolvability measurement (replaces the ad-hoc global ~8.5us).
#
# For a case, the question "is this PR measurable in isolation?" has no universal answer -- it
# depends on the module, dtype, token count and GPU. So we MEASURE it per case: time the target
# module REPS times on the BEFORE image and REPS times on the AFTER image, then report
#   before_us (mean+-std), after_us (mean+-std), gap = before-after, noise = pooled std,
#   resolvable = gap > MARGIN * noise      (MARGIN default 3 -> gap must clear 3 sigma).
# The per-case "floor" is `MARGIN * noise`: the smallest before/after gap this harness can call
# real for THIS module. A case is a viable eval iff before/after clears its own measured floor;
# 28103 clears it, 27931 does not -- determined, not assumed.
#
# Config JSON must carry: before_image, after_image, framework(vllm|sglang), model_snap,
# targets[] (first = primary), tokens; optional sglang_quant. Runs the extractor INSIDE each
# image; binds the GPU from DOCKER_GPU_ARG (slurm_gpu.sh). For sglang it bakes
# sglang_profile_patch.py into the image the same way validate_sglang_capture.sh does.
#
# Usage (under slurm):  REPS=5 MARGIN=3 sbatch slurm_gpu.sh ./measure_floor.sh issue_28103.json
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:?usage: measure_floor.sh <config.json>}"; [ -f "$CONFIG" ] || CONFIG="$HERE/$1"
REPS="${REPS:-5}"; MARGIN="${MARGIN:-3}"
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
GOLDEN_DIR=/raid/yilegu/eval_goldens; mkdir -p "$GOLDEN_DIR"
EXTRACTOR="$HERE/extract_and_profile.py"

eval "$(python3 - "$CONFIG" <<'PY'
import json,sys,shlex
c=json.load(open(sys.argv[1])); r=c.get("render",{})
def e(k,v): print(f'{k}={shlex.quote(str(v))}')
e("CASE", c.get("case","case")); e("SNAP", c["model_snap"])
e("FW", r.get("FRAMEWORK", c.get("framework","vllm")))
e("BEFORE", c["before_image"]); e("AFTER", c.get("after_image",""))
e("TARGET", c["targets"][0]); e("TOKENS", c.get("tokens",8))
e("QUANT", c.get("sglang_quant",""))
PY
)"
[ -n "$AFTER" ] || { echo "config lacks after_image -- add it to run the floor measurement"; exit 2; }
OUT="$HERE/floor_${CASE}"; mkdir -p "$OUT"
GOLDEN="$GOLDEN_DIR/floor_${CASE}.pt"
echo "==== floor measure: case=$CASE fw=$FW target=$TARGET reps=$REPS margin=$MARGIN gpu=$GPU ===="

start_container () {  # <image> -> prints container name
  local img="$1" cname="floor_$(date +%s%N)"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run -d --name "$cname" --gpus "$GPU" \
    -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -v /raid/yilegu/models:/raid/yilegu/models:ro \
    -v /raid/hf/hub:/raid/hf/hub:ro \
    -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer \
    -v "$GOLDEN_DIR":"$GOLDEN_DIR" \
    --workdir /tmp --entrypoint sleep "$img" infinity >/dev/null
  docker cp "$EXTRACTOR" "$cname:/tmp/extract_and_profile.py" >/dev/null
  if [ "$FW" = sglang ]; then
    docker cp "$HERE/sglang_profile_patch.py" "$cname:/tmp/sglang_profile_patch.py" >/dev/null
    docker exec -i "$cname" python3 - <<'PY' >/dev/null
import os, sglang
sp=os.path.dirname(sglang.__file__); sched=os.path.join(sp,"srt","managers","scheduler.py")
m="# --- ep capture patch ---"; src=open(sched).read()
if m not in src:
    open(sched,"a").write("\n\n"+m+"\ntry:\n import sys as _s\n if '/tmp' not in _s.path: _s.path.insert(0,'/tmp')\n import sglang_profile_patch as _e; _e.attach(Scheduler)\nexcept Exception as _x:\n import logging; logging.getLogger(__name__).warning('ep patch failed: %s', _x)\n")
PY
  fi
  echo "$cname"
}

extra=(); [ -n "$QUANT" ] && extra=(--sglang-quant "$QUANT")

# BEFORE: capture golden once, then time REPS times (capture+time each run gives a fresh timing)
CB="$(start_container "$BEFORE")"
docker exec --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 "$CB" \
  python3 /tmp/extract_and_profile.py --model "$SNAP" --framework "$FW" --target "$TARGET" \
  --tokens "$TOKENS" --iters 3000 --capture "$GOLDEN" "${extra[@]}" >/dev/null 2>&1 || true
for i in $(seq 1 "$REPS"); do
  docker exec --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 "$CB" \
    python3 /tmp/extract_and_profile.py --model "$SNAP" --framework "$FW" --target "$TARGET" \
    --replay "$GOLDEN" --iters 3000 "${extra[@]}" 2>/dev/null | grep '^JSON ' | sed 's/^JSON //'
done > "$OUT/before_samples.jsonl"
docker rm -f "$CB" >/dev/null 2>&1 || true

# AFTER: replay the SAME before-golden REPS times (measures the fixed kernel on identical input)
CA="$(start_container "$AFTER")"
for i in $(seq 1 "$REPS"); do
  docker exec --workdir /tmp -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 "$CA" \
    python3 /tmp/extract_and_profile.py --model "$SNAP" --framework "$FW" --target "$TARGET" \
    --replay "$GOLDEN" --iters 3000 "${extra[@]}" 2>/dev/null | grep '^JSON ' | sed 's/^JSON //'
done > "$OUT/after_samples.jsonl"
docker rm -f "$CA" >/dev/null 2>&1 || true

python3 - "$OUT/before_samples.jsonl" "$OUT/after_samples.jsonl" "$MARGIN" "$CASE" "$OUT/floor_report.json" <<'PY'
import json,sys,statistics as st
bef=[json.loads(l)["latency_us"] for l in open(sys.argv[1]) if l.strip()]
aft=[json.loads(l)["latency_us"] for l in open(sys.argv[2]) if l.strip()]
margin=float(sys.argv[3]); case=sys.argv[4]; outp=sys.argv[5]
def ms(x): return (st.mean(x), (st.pstdev(x) if len(x)>1 else 0.0))
bm,bs=ms(bef); am,ap=ms(aft); gap=bm-am
noise=((bs**2+ap**2)**0.5) or 1e-9; floor=margin*noise
rep={"case":case,"reps":len(bef),"before_us":round(bm,4),"before_std":round(bs,4),
     "after_us":round(am,4),"after_std":round(ap,4),"gap_us":round(gap,4),
     "noise_us":round(noise,4),"floor_us":round(floor,4),
     "resolvable":bool(gap>floor),"margin":margin,
     "before_samples":[round(x,3) for x in bef],"after_samples":[round(x,3) for x in aft]}
json.dump(rep,open(outp,"w"),indent=2)
print("==== FLOOR REPORT ("+case+") ====")
print(f"  before {bm:.3f}+-{bs:.3f} us   after {am:.3f}+-{ap:.3f} us")
print(f"  gap {gap:.3f} us   noise(pooled sd) {noise:.3f} us   floor({margin}sigma) {floor:.3f} us")
print(f"  RESOLVABLE = {gap>floor}  ({'gap clears floor -> viable case' if gap>floor else 'gap below floor -> NULL for this harness'})")
PY
echo "==== done -> $OUT/floor_report.json ===="
