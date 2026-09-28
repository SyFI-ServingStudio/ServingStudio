#!/usr/bin/env bash
# Resume follow-up #4 (mixed + verify campaigns) after the 2026-09-27 evening GPU starvation (all 8 GPUs held by
# the user's 4-GPU job + four slurm-manager placeholders; KDA-mixed r1 got a FALSE "non-compiling" verdict because its
# rebuild smoke timed out after 90 min in the queue). Run when `sinfo`/`squeue` show free GPUs on partition main.
#   1. re-judge the KDA-mixed r1 tree (agent tree exists; only the smoke/judge never ran) as trial 1;
#   2. verify: KDA rounds 2-3 (r1 verdict is whatever job 2142 produced), MLA rounds 1-3 (from PRISTINE, see the write-up);
#   3. mixed: KDA rounds 2-3, MLA rounds 1-3 (MLA r1 may already have a verdict if its in-flight trial finished).
# Everything sequential (one agent at a time) to keep the queue pressure low.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}" MIN_IMPROVEMENT=0 K3_GPU_MODE=slurm K3_SHARED_GPU=1
O="$HERE/iter_opt_eval_k3_kda_mixed_claude"
if grep -q "non-compiling" "$O/trial_1_verdict.json" 2>/dev/null && [ -d "$O/trial_1_tree" ]; then
  echo "==== $(date +%F_%T) re-judging KDA mixed r1 (false non-compiling verdict from the queue timeout)"
  mv "$O/trial_1_verdict.json" "$O/trial_1_verdict.queue_timeout.json"
  job="$(sbatch --parsable --partition=main --job-name=k3j_kda_mixed_r1_rejudge --output="$O/trial_1_judge_slurm.rejudge.out" \
        "$HERE/slurm_gpu.sh" "$HERE/k3_slurm_step.sh" python3 "$HERE/judge_k3.py" --agent-container rejudge --config "$HERE/issue_k3_kda_mixed_claude.json" \
        --golden-dir /raid/yilegu/eval_goldens --out "$O/trial_1_verdict.json" --tree-dir "$O/trial_1_tree" --pristine-dir "$O/pristine_tree" \
        --baseline-tree "$O/best_tree" --min-improvement 0 | cut -d';' -f1)"
  while squeue -h -j "$job" -o %T 2>/dev/null | grep -q .; do sleep 30; done
  python3 - "$O" <<'PY'
import json, sys, time, shutil, os
o = sys.argv[1]; v = json.load(open(f"{o}/trial_1_verdict.json"))
pts = {p: (round(d["before_us"], 1), round(d["after_us"], 1), d["improvement"], d["correctness"].get("pass")) for p, d in (v.get("points") or {}).items()}
open(f"{o}/rounds.jsonl", "a").write(json.dumps({"round": 1, "case": "kda_mixed_claude", "verdict": v.get("verdict"), "reason": v.get("reason"), "points": pts, "time": time.strftime("%F %T"), "note": "re-judged after queue-timeout false FAIL"}) + "\n")
print("KDA mixed r1 re-judge:", v.get("verdict"), pts)
if v.get("verdict") == "PASS":
    shutil.rmtree(f"{o}/best_tree.prev", ignore_errors=True); os.rename(f"{o}/best_tree", f"{o}/best_tree.prev"); shutil.copytree(f"{o}/trial_1_tree_judged", f"{o}/best_tree", symlinks=True); print("promoted")
PY
fi
# mla_mixed gets 4 because its r1 (slurm 2150, FAIL +0.04%) was a wasted round: the agent's tree came back byte-identical
# to the seed -- it never got past iter_00 while its GPU requests sat behind the placeholders.
for spec in "kda_verify_claude 2 2" "mla_verify_claude 1 3" "kda_mixed_claude 2 2" "mla_mixed_claude 1 4"; do
  set -- $spec; c="$1"; k0="$2"; n="$3"
  # skip rounds that already have a verdict (an in-flight trial may have finished after the pause)
  while [ -f "$HERE/iter_opt_eval_k3_${c}/trial_${k0}_verdict.json" ] && [ "$n" -gt 0 ]; do k0=$((k0+1)); n=$((n-1)); done
  [ "$n" -gt 0 ] || continue
  echo "==== $(date +%F_%T) $c rounds $k0..$((k0+n-1))"
  "$HERE/run_k3_continuous.sh" slurm "$c" "$k0" "$n"
done
echo "==== $(date +%F_%T) mixed/verify campaigns done"
