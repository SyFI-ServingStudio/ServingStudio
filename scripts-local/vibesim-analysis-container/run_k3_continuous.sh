#!/usr/bin/env bash
# Continuous optimization loop for one Kimi-K3 case (user decision 2026-09-24: no fixed 5% gate;
# keep optimizing from the latest accepted tree, re-analyze, optimize further, never regress
# accuracy). Each round:
#   1. seed the agent with best_tree (pristine on the very first round),
#   2. run one trial (agent + judge) via run_iter_opt_eval.sh with START_TREE/MIN_IMPROVEMENT=0,
#      so the judge measures the round's latency baseline from best_tree and checks the output +
#      post-step state against the PRISTINE goldens (accuracy can never drift across rounds),
#   3. on PASS (primary gain >= max(3 sigma, 0.5%), no secondary regression, CHECK pass) promote
#      the judged tree to best_tree and append to rounds.jsonl; otherwise keep best_tree.
# Usage: run_k3_continuous.sh <gpu> <kda|mla> <first_round_k> <n_rounds> [seed_tree]
#   K3_SHARED_GPU=1 AGENT_TIMEOUT=2700 ./run_k3_continuous.sh 7 mla 20 3 [path/to/seed_tree]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="$1"; CASE="$2"; K0="$3"; N="$4"; SEED="${5:-}"
# GPU arg: an index 0-7 (direct docker, 2026-09-25: any free GPU) or `slurm` (2026-09-26: no GPU in the
# agent container; every measurement and the judge are their own sbatch jobs on partition main).
case "$GPU_IDX" in
  [0-7]) export DOCKER_GPU_ARG="\"device=$GPU_IDX\"" ;;
  slurm) export K3_GPU_MODE=slurm K3_SHARED_GPU=1 ;;
  *) echo "!! bad GPU arg $GPU_IDX (0-7 | slurm)"; exit 1 ;;
esac
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}" MIN_IMPROVEMENT=0
OUT="$HERE/iter_opt_eval_k3_${CASE}"; BEST="$OUT/best_tree"; HIST="$OUT/rounds.jsonl"
mkdir -p "$OUT"
if [ -n "$SEED" ] && [ ! -d "$BEST" ]; then cp -a "$SEED" "$BEST"; echo "seeded best_tree from $SEED"; fi

for k in $(seq "$K0" $((K0 + N - 1))); do
  if [ "${K3_SHARED_GPU:-0}" != 1 ]; then
    while :; do used="$(nvidia-smi -i "$GPU_IDX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      [ "${used:-99999}" -lt 1000 ] && break; echo "$(date +%T) GPU $GPU_IDX busy -> wait 60s"; sleep 60; done
  fi
  echo "==== $(date +%F_%T) $CASE round $k (best_tree: $([ -d "$BEST" ] && echo yes || echo pristine))"
  if [ -d "$BEST" ]; then export START_TREE="$BEST"; else unset START_TREE; fi
  START="$k" "$HERE/run_iter_opt_eval.sh" "$HERE/issue_k3_${CASE}.json" "$k" max \
    > "$HERE/direct_${CASE}_trial_${k}.out" 2>&1 || echo "!! round $k runner exited non-zero"
  V="$OUT/trial_${k}_verdict.json"
  if [ ! -f "$V" ]; then echo "!! no verdict for round $k"; continue; fi
  python3 - "$V" "$k" "$CASE" "$HIST" <<'PY'
import json, sys, time
v = json.load(open(sys.argv[1])); k, case, hist = sys.argv[2], sys.argv[3], sys.argv[4]
pts = {p: (round(d["before_us"], 1), round(d["after_us"], 1), d["improvement"], d["correctness"].get("pass"))
       for p, d in (v.get("points") or {}).items()}
rec = {"round": int(k), "case": case, "verdict": v.get("verdict"), "reason": v.get("reason"),
       "points": pts, "files": v.get("changed_files"), "diff_lines": v.get("diff_lines"),
       "pristine_points": v.get("pristine_points"), "time": time.strftime("%F %T")}
open(hist, "a").write(json.dumps(rec) + "\n")
print(f"round {k}: {v.get('verdict')} {pts}")
PY
  if grep -q '"verdict": "PASS"' "$V"; then
    rm -rf "$BEST.prev"; [ -d "$BEST" ] && mv "$BEST" "$BEST.prev"
    cp -a "$OUT/trial_${k}_tree_judged" "$BEST"
    echo "==== round $k PASS -> promoted trial_${k}_tree_judged to best_tree"
  else
    echo "==== round $k not accepted -> best_tree unchanged"
  fi
done
echo "==== $(date +%F_%T) $CASE continuous loop done; history: $HIST"
