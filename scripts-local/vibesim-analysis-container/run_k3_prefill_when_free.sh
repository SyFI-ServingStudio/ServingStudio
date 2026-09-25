#!/usr/bin/env bash
# Chunked-prefill campaign (cases k3_{kda,mla}_prefill; user request 2026-09-24), warm-started from the
# decode best trees. Waits until (a) the prefill oracles answer on 8805/8806 (baked after VibeSim B12) and
# (b) an authorized GPU has room (GPU 2/3 idle for IDLE_CHECKS checks, or GPU 7 with >= 60 GB free), then
# per case: seed best_tree from iter_opt_eval_k3_<case>/best_tree, TRANSFER CHECK on the prefill points
# (falls back to the pristine tree if CHECK fails), then alternate continuous rounds mla_prefill / kda_prefill.
# Usage: run_k3_prefill_when_free.sh <first_round_k> <n_rounds_per_case>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
K0="$1"; N="$2"; SUFFIX="${CASE_SUFFIX:-prefill}"   # prefill | prefill_claude
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}"
IDLE_CHECKS="${IDLE_CHECKS:-30}"
oracles_up () {
  for port in 8805 8806; do
    curl -s -m 5 "http://172.17.0.1:$port/api/v1/analyze?level=run_summary" | grep -q '"' || return 1
  done
}
until oracles_up; do echo "$(date +%T) prefill oracles 8805/8806 not up yet -> wait 300s"; sleep 300; done
echo "==== $(date +%F_%T) prefill oracles up"
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
for c in mla kda; do
  SRC="$HERE/iter_opt_eval_k3_${c}"; OUT="$HERE/iter_opt_eval_k3_${c}_${SUFFIX}"; mkdir -p "$OUT"
  [ -d "$OUT/pristine_tree" ] || cp -a "$SRC/pristine_tree" "$OUT/pristine_tree"
  if [ ! -d "$OUT/best_tree" ]; then
    cp -a "$SRC/best_tree" "$OUT/best_tree"; echo "==== seeded ${c}_${SUFFIX} best_tree from $SRC/best_tree"
    echo "==== $(date +%F_%T) transfer check: ${c}_${SUFFIX} seed tree vs pristine on the prefill points"
    python3 "$HERE/judge_k3.py" --agent-container transfer_check --config "$HERE/issue_k3_${c}_${SUFFIX}.json" \
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
    print("!! seed tree fails CHECK (or no result) on the prefill points -> starting this case from PRISTINE")
    shutil.move(os.path.join(out, "best_tree"), os.path.join(out, "best_tree.transfer_failed"))
PY
  fi
done
echo "==== $(date +%F_%T) starting alternate rounds $K0..$((K0+N-1)) for mla_${SUFFIX} kda_${SUFFIX} on GPU $GPU_IDX"
CASES="mla_${SUFFIX} kda_${SUFFIX}" exec "$HERE/run_k3_alternate.sh" "$GPU_IDX" "$K0" "$N"
