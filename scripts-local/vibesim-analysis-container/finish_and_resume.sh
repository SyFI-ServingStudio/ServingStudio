#!/usr/bin/env bash
# 2026-09-28 01:40: the placeholders released GPUs 0/5/6/7 and the three paused-campaign judges started
# (2150 = MLA mixed r1 judge, 2152 = KDA verify r1 15-rep re-judge, 2149 = stray MLA-mixed measurement).
# Their parent loops were killed for the pause, so nobody records/promotes their verdicts. This script:
#   1. waits for 2150 and 2152 to leave the queue;
#   2. appends the verdicts to rounds.jsonl (same record shape as run_k3_continuous.sh) and promotes PASS trees;
#   3. hands over to resume_mixed_verify.sh for the remaining rounds.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WAIT_JOBS="${WAIT_JOBS:-2150 2152}"
for j in $WAIT_JOBS; do
  while squeue -h -j "$j" -o %T 2>/dev/null | grep -q .; do sleep 60; done
  echo "$(date +%F_%T) job $j finished"
done
python3 - "$HERE" <<'PY'
import json, sys, time, os, shutil
here = sys.argv[1]
def rec(case, k, vpath, note, promote_from=None):
    o = f"{here}/iter_opt_eval_k3_{case}"
    if not os.path.exists(vpath):
        print(f"!! {case} r{k}: no verdict at {vpath}"); return
    v = json.load(open(vpath))
    pts = {p: (round(d["before_us"], 1), round(d["after_us"], 1), d["improvement"], d["correctness"].get("pass"))
           for p, d in (v.get("points") or {}).items()}
    r = {"round": k, "case": case, "verdict": v.get("verdict"), "reason": v.get("reason"), "points": pts,
         "files": v.get("changed_files"), "diff_lines": v.get("diff_lines"), "pristine_points": v.get("pristine_points"),
         "time": time.strftime("%F %T"), "note": note}
    open(f"{o}/rounds.jsonl", "a").write(json.dumps(r) + "\n")
    print(f"{case} r{k}: {v.get('verdict')} {v.get('reason')} {pts}")
    if v.get("verdict") == "PASS" and promote_from and os.path.isdir(promote_from):
        best = f"{o}/best_tree"
        shutil.rmtree(best + ".prev", ignore_errors=True)
        if os.path.isdir(best): os.rename(best, best + ".prev")
        shutil.copytree(promote_from, best, symlinks=True)
        print(f"   promoted {promote_from} -> best_tree")
kv = f"{here}/iter_opt_eval_k3_kda_verify_claude"
rec("kda_verify_claude", 1, f"{kv}/trial_1_verdict.json",
    "5-rep judge: noise-rejected (sigma 141us on the host-bound eager verify step); re-judged at 15 reps below")
rec("kda_verify_claude", 1, f"{kv}/rejudge15_trial_1_verdict.json", "15-rep re-judge of trial 1 (slurm 2152)",
    promote_from=f"{kv}/rejudge15_trial_1_tree_judged")
mm = f"{here}/iter_opt_eval_k3_mla_mixed_claude"
rec("mla_mixed_claude", 1, f"{mm}/trial_1_verdict.json", "judge ran as slurm 2150 after the pause; parent loop killed",
    promote_from=f"{mm}/trial_1_tree_judged")
PY
echo "$(date +%F_%T) handing over to resume_mixed_verify.sh"
exec "$HERE/resume_mixed_verify.sh"
