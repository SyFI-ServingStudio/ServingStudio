#!/usr/bin/env bash
# Replaces the tail of launch_b12b_then_prefill.sh (stopped 02:50 when the user switched the agent to Claude):
# wait for the running Codex B12b process, fast-forward main/ kimi-k3, re-bake the oracle image with the prefill
# predictions, start 8805/8806, then run the PREFILL campaign with the Claude agent (cases *_prefill_claude).
# Usage: after_b12b_claude_prefill.sh <b12b_codex_pid> <first_round_k> <n_rounds>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust
C_PID="$1"; K0="$2"; N="$3"
while kill -0 "$C_PID" 2>/dev/null; do sleep 60; done
echo "==== $(date +%F_%T) B12b codex exited; HEAD $(git -C "$WT" log --oneline -1)"
[ -n "$(git -C "$WT" status --short | grep -v '^??')" ] && echo "!! B12b left an uncommitted tree (needs review); baking HEAD"
cd /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main && git merge --ff-only kimi-k3-arch 2>&1 | tail -1
echo "==== $(date +%F_%T) baking oracle image incl. prefill predictions"
if "$HERE/build_context_k3.sh" > "$HERE/k3_prefill_bake.log" 2>&1 \
   && sed 's#COPY context/#COPY context_k3/#' "$HERE/Dockerfile" | docker build -q -t vibesim-analysis:k3 -f - "$HERE" >> "$HERE/k3_prefill_bake.log" 2>&1; then
  for spec in k3_kda_prefill:8805 k3_mla_prefill:8806; do
    run="${spec%%:*}"; port="${spec##*:}"
    docker rm -f "vibesim_oracle_${run}" >/dev/null 2>&1 || true
    docker run -d --name "vibesim_oracle_${run}" -e VIBESIM_PREBAKED_RUN="$run" -e VIBESIM_API_PORT="$port" \
      -p "172.17.0.1:${port}:${port}" vibesim-analysis:k3 >/dev/null
  done
  sleep 25
  for port in 8805 8806; do printf "oracle %s: " "$port"; curl -s "http://172.17.0.1:$port/api/v1/analyze?level=run_summary" | head -c 120; echo; done
else
  echo "!! prefill oracle bake failed (k3_prefill_bake.log)"; exit 1
fi
echo "==== $(date +%F_%T) handing over to the Claude prefill campaign"
CASE_SUFFIX=prefill_claude exec "$HERE/run_k3_prefill_when_free.sh" "$K0" "$N"
