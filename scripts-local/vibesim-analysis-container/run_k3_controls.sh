#!/usr/bin/env bash
# Judge-integrity controls for the Kimi-K3 open-ended case (C1 of the plan). Runs three
# "trials" of run_iter_opt_eval.sh with the agent step disabled (AGENT_TIMEOUT=1):
#   trial 1  null control      pristine tree            -> expect FAIL(latency_improved), CHECK pass,
#                                                          improvement ~ 0 +- noise
#   trial 2  planted slowdown  controls/planted_slowdown -> expect FAIL(latency_improved), CHECK pass,
#                                                          improvement clearly negative
#   trial 3  planted numerics  controls/planted_numerics -> expect FAIL(correctness)
# Usage: sbatch slurm_gpu.sh ./run_k3_controls.sh [issue_k3_kda.json]
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-issue_k3_kda.json}"
CASE="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['case'])" "$HERE/$CFG")"
OUT="$HERE/iter_opt_eval_${CASE}"
export AGENT_TIMEOUT=1
echo "######## control 1/3: null (pristine tree)"
START=1 "$HERE/run_iter_opt_eval.sh" "$CFG" 1 max || true
echo "######## control 2/3: planted slowdown"
START=2 PLANT_PATCH="$HERE/controls/planted_slowdown.patch" "$HERE/run_iter_opt_eval.sh" "$CFG" 2 max || true
echo "######## control 3/3: planted numerics change"
START=3 PLANT_PATCH="$HERE/controls/planted_numerics.patch" "$HERE/run_iter_opt_eval.sh" "$CFG" 3 max || true
echo "######## verdicts"
python3 - "$OUT" <<'PY'
import json,glob,os,sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "trial_*_verdict.json"))):
    d=json.load(open(f))
    pts={k:(round(v.get("improvement",0),4), v.get("correctness",{}).get("pass")) for k,v in (d.get("points") or {}).items()}
    print(os.path.basename(f), d.get("verdict"), d.get("reason"), "required=",d.get("required_improvement"), pts)
PY
