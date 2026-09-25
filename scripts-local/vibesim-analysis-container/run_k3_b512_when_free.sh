#!/usr/bin/env bash
# Large-batch decode campaign (B=512/256 @ 8k; user request 2026-09-24), warm-started from the
# decode best trees. Waits for an authorized GPU (same policy as resume_k3_loop_when_free.sh:
# GPU 2/3 idle for IDLE_CHECKS consecutive checks, or GPU 7 with >= 60 GB free), then per case:
#   1. seed iter_opt_eval_k3_<case>_b512/best_tree from iter_opt_eval_k3_<case>/best_tree,
#   2. TRANSFER CHECK: judge that seed tree on the new points against the pristine tree
#      (captures the new goldens; tells whether the inherited levers transfer and stay correct).
#      If CHECK fails on any new point, the case starts from the pristine tree instead
#      (seed kept as best_tree.transfer_failed).
#   3. then alternate continuous rounds mla_b512 / kda_b512 (K0..K0+N-1).
# Usage: run_k3_b512_when_free.sh <first_round_k> <n_rounds_per_case>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
K0="$1"; N="$2"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}"
IDLE_CHECKS="${IDLE_CHECKS:-30}"
declare -A IDLE_STREAK=([2]=0 [3]=0)
GPU_IDX=""
while [ -z "$GPU_IDX" ]; do
  for g in 3 2; do
    used="$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [ "${used:-99999}" -lt 1000 ]; then IDLE_STREAK[$g]=$((IDLE_STREAK[$g] + 1)); else IDLE_STREAK[$g]=0; fi
    if [ "${IDLE_STREAK[$g]}" -ge "$IDLE_CHECKS" ]; then GPU_IDX="$g"; break; fi
  done
  [ -n "$GPU_IDX" ] && break
  used7="$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  total7="$(nvidia-smi -i 7 --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
  if [ $((total7 - used7)) -ge 61440 ]; then GPU_IDX=7; export K3_SHARED_GPU=1; break; fi
  echo "$(date +%T) GPUs 2/3 busy, GPU 7 free=$((total7 - used7)) MiB -> wait 120s"; sleep 120
done
echo "==== $(date +%F_%T) using GPU $GPU_IDX (shared=${K3_SHARED_GPU:-0})"
export DOCKER_GPU_ARG="\"device=$GPU_IDX\""

# 0. VibeSim side: JIT-fill the rows the B=512/256 presets need (branch profile.db), bake the
#    large-batch predictions into the oracle image and serve them on 8803 (KDA) / 8804 (MLA).
if [ "${SKIP_ORACLE:-0}" != 1 ]; then
  echo "==== $(date +%F_%T) JIT fill for the b512 presets on GPU $GPU_IDX"
  K3_GPU_INDEX="$GPU_IDX" "$HERE/run_k3_jit_fill.sh" \
    presets/predict_kimi_k3_b200_rank1_layer_kda_b512.json \
    presets/predict_kimi_k3_b200_rank1_layer_mla_b512.json > "$HERE/k3_b512_fill.log" 2>&1
  grep "^RESULT" "$HERE/k3_b512_fill.log"
  echo "==== $(date +%F_%T) baking oracle image with the b512 predictions"
  if "$HERE/build_context_k3.sh" > "$HERE/k3_b512_bake.log" 2>&1 \
     && sed 's#COPY context/#COPY context_k3/#' "$HERE/Dockerfile" | docker build -q -t vibesim-analysis:k3 -f - "$HERE" >> "$HERE/k3_b512_bake.log" 2>&1; then
    for spec in k3_kda_b512:8803 k3_mla_b512:8804; do
      run="${spec%%:*}"; port="${spec##*:}"
      docker rm -f "vibesim_oracle_${run}" >/dev/null 2>&1 || true
      docker run -d --name "vibesim_oracle_${run}" -e VIBESIM_PREBAKED_RUN="$run" -e VIBESIM_API_PORT="$port" \
        -p "172.17.0.1:${port}:${port}" vibesim-analysis:k3 >/dev/null
    done
    sleep 25
    for port in 8803 8804; do
      printf "oracle %s: " "$port"; curl -s "http://172.17.0.1:$port/api/v1/analyze?level=run_summary" | head -c 200; echo
    done
  else
    echo "!! oracle bake failed (see k3_b512_bake.log); b512 cases would talk to dead ports 8803/8804 -> stopping"
    exit 1
  fi
fi

for c in mla kda; do
  SRC="$HERE/iter_opt_eval_k3_${c}"; OUT="$HERE/iter_opt_eval_k3_${c}_b512"; mkdir -p "$OUT"
  [ -d "$OUT/pristine_tree" ] || cp -a "$SRC/pristine_tree" "$OUT/pristine_tree"
  if [ ! -d "$OUT/best_tree" ]; then
    cp -a "$SRC/best_tree" "$OUT/best_tree"; echo "==== seeded ${c}_b512 best_tree from $SRC/best_tree"
    echo "==== $(date +%F_%T) transfer check: ${c}_b512 seed tree vs pristine on the new points"
    python3 "$HERE/judge_k3.py" --agent-container transfer_check --config "$HERE/issue_k3_${c}_b512.json" \
      --golden-dir /raid/yilegu/eval_goldens --out "$OUT/seed_transfer_verdict.json" \
      --tree-dir "$OUT/best_tree" --pristine-dir "$OUT/pristine_tree" --min-improvement 0 \
      > "$OUT/seed_transfer_judge.log" 2>&1
    python3 - "$OUT/seed_transfer_verdict.json" "$OUT" <<'PY'
import json, sys, shutil, os
d = json.load(open(sys.argv[1])); out = sys.argv[2]
print("transfer:", d.get("verdict"), (d.get("reason") or "")[:120])
ok = True
for k, p in (d.get("points") or {}).items():
    c = p["correctness"]; ok &= bool(c.get("pass"))
    print(f"  {k}: pristine {p['before_us']:.1f} -> seed {p['after_us']:.1f} us ({p['improvement']*100:+.1f}%) CHECK {c.get('pass')} rel {c.get('max_rel_err')}")
if not ok or not d.get("points"):
    print("!! seed tree fails CHECK (or no result) on the new points -> starting this case from PRISTINE")
    shutil.move(os.path.join(out, "best_tree"), os.path.join(out, "best_tree.transfer_failed"))
PY
  fi
done
echo "==== $(date +%F_%T) starting alternate rounds $K0..$((K0+N-1)) for mla_b512 kda_b512 on GPU $GPU_IDX"
CASES="mla_b512 kda_b512" exec "$HERE/run_k3_alternate.sh" "$GPU_IDX" "$K0" "$N"
