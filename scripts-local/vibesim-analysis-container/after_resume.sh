#!/usr/bin/env bash
# 2026-09-28 07:10: KDA mixed round 3's agent never ran (Bedrock returned 503 on all 10 retries, 0 tokens, tree ==
# best_tree); its null judge (slurm 2336) was cancelled. Re-run the round once resume_mixed_verify.sh has finished the
# MLA mixed rounds, keeping the failed evidence under trial_3_api503_*.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "[r]esume_mixed_verify.sh" >/dev/null; do sleep 60; done
echo "==== $(date +%F_%T) resume_mixed_verify.sh finished; re-running KDA mixed round 3"
O="$HERE/iter_opt_eval_k3_kda_mixed_claude"
for f in "$O"/trial_3_*; do
  b="$(basename "$f")"; [ -e "$f" ] && mv "$f" "$O/trial_3_api503_${b#trial_3_}"
done
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}" MIN_IMPROVEMENT=0 K3_GPU_MODE=slurm K3_SHARED_GPU=1
"$HERE/run_k3_continuous.sh" slurm kda_mixed_claude 3 1
echo "==== $(date +%F_%T) after_resume done"
