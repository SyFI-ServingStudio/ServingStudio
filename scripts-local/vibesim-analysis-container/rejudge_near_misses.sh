#!/usr/bin/env bash
# Follow-up #2 (user 2026-09-27): re-judge the exact-but-below-3-sigma rounds with a lower noise floor (15 reps
# instead of 5; a reps change re-keys the goldens, so fresh 15-rep goldens/baselines are generated too).
# One slurm job per case (each ~60-90 min; the 2 h partition limit is why they are not chained).
#   rejudge_near_misses.sh kda_b512_claude 2 1     # trial 2 tree vs the trial-1 tree it was judged against
#   rejudge_near_misses.sh kda_lcprefill_claude 3 2
# Verdict: iter_opt_eval_k3_<case>/rejudge15_trial_<k>_verdict.json
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CASE="$1"; K="$2"; BASE_K="$3"; REPS="${REPS:-15}"
OUT="$HERE/iter_opt_eval_k3_${CASE}"
CFG="$OUT/issue_k3_${CASE}_reps${REPS}.json"
python3 - "$HERE/issue_k3_${CASE}.json" "$CFG" "$REPS" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["reps"] = int(sys.argv[3]); c["min_improvement"] = 0.0
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
TREE="$OUT/trial_${K}_tree_judged"; BASE="$OUT/trial_${BASE_K}_tree_judged"
[ -d "$TREE" ] && [ -d "$BASE" ] || { echo "!! missing $TREE or $BASE"; exit 1; }
rm -rf "$OUT/rejudge15_trial_${K}_tree" "$OUT/rejudge15_trial_${K}_tree_judged"; cp -a "$TREE" "$OUT/rejudge15_trial_${K}_tree"
job="$(sbatch --parsable --partition=main --job-name="k3_rejudge_${CASE}_${K}" --output="$OUT/rejudge15_trial_${K}.slurm.out" \
      "$HERE/slurm_gpu.sh" "$HERE/k3_slurm_step.sh" python3 "$HERE/judge_k3.py" \
      --agent-container "rejudge15_${CASE}_${K}" --config "$CFG" --golden-dir /raid/yilegu/eval_goldens \
      --out "$OUT/rejudge15_trial_${K}_verdict.json" --tree-dir "$OUT/rejudge15_trial_${K}_tree" \
      --pristine-dir "$OUT/pristine_tree" --baseline-tree "$BASE" --min-improvement 0 | cut -d';' -f1)"
echo "$(date +%F_%T) $CASE trial $K vs trial $BASE_K, $REPS reps -> slurm job $job ($OUT/rejudge15_trial_${K}.slurm.out)"
