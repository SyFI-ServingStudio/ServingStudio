#!/usr/bin/env bash
# MLA-b512 round 1 (20:13-21:07) was measured while Codex B12 profiled 16k-token prefill kernels on the
# same GPU 7: the B=512 baseline reps were 2104/995/3363/3530/3419 us (clean value 978-992) -> the round
# is infra noise, round 2 was stopped at 21:10. Restart the MLA-b512 rounds once the B12 launcher (PID $1)
# has exited and GPU 7 has been quiet for 5 consecutive checks; drop the polluted pristine-baseline stats
# (baseline.json; the goldens themselves are deterministic and stay) so the judge re-measures them.
# Usage: restart_mla_b512_after_b12.sh <b12_launcher_pid> <first_round_k> <n_rounds>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WAIT_PID="$1"; K0="$2"; N="$3"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "==== $(date +%F_%T) B12 launcher exited"
quiet=0
while [ "$quiet" -lt 5 ]; do
  util="$(nvidia-smi -i 7 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  used="$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [ "${util:-100}" -lt 5 ] && [ "${used:-999999}" -lt 20000 ]; then quiet=$((quiet+1)); else quiet=0; fi
  sleep 30
done
echo "==== $(date +%F_%T) GPU 7 quiet -> dropping polluted baseline stats and restarting MLA-b512 rounds $K0..$((K0+N-1))"
rm -f /raid/yilegu/eval_goldens/golden_k3_16a2cf1adf84/baseline.json
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}" K3_SHARED_GPU=1 DOCKER_GPU_ARG="\"device=7\""
CASES="mla_b512" exec "$HERE/run_k3_alternate.sh" 7 "$K0" "$N"
