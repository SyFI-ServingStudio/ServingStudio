#!/usr/bin/env bash
# Queue MLA-b512 continuous rounds behind the KDA-b512 rounds (same GPU 7; never two trials at once).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WAIT_PID="$1"; K0="$2"; N="$3"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
until curl -s -m 5 "http://172.17.0.1:8804/api/v1/analyze?level=run_summary" | grep -q '"'; do sleep 60; done
echo "==== $(date +%F_%T) KDA-b512 rounds done -> MLA-b512 rounds $K0..$((K0+N-1)) on GPU 7"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}" K3_SHARED_GPU=1 DOCKER_GPU_ARG="\"device=7\""
CASES="mla_b512" exec "$HERE/run_k3_alternate.sh" 7 "$K0" "$N"
