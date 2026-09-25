#!/usr/bin/env bash
# Chain for the prefill dimension (GPU 7 is shared: never profile while a judge trial runs):
#   1. wait for finish_b12.sh (B12 committed) and for the MLA-b512 rounds to finish
#      (restart_mla_b512_after_b12.sh exec's into run_k3_alternate.sh, so its PID lives until the rounds end),
#   2. run Codex B12b (prefill prediction fidelity: currently ~60% below the measured step) in the worktree,
#   3. fast-forward main/ kimi-k3, re-bake the oracle image (prefill predictions -> k3_{kda,mla}_prefill),
#      start the 8805 (KDA) / 8806 (MLA) oracle containers,
#   4. hand over to run_k3_prefill_when_free.sh (transfer checks + prefill rounds).
# Usage: launch_b12b_then_prefill.sh <finish_b12_pid> <mla_b512_rounds_pid> <first_round_k> <n_rounds>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust
F_PID="$1"; R_PID="$2"; K0="$3"; N="$4"
while kill -0 "$F_PID" 2>/dev/null || kill -0 "$R_PID" 2>/dev/null; do sleep 60; done
echo "==== $(date +%F_%T) B12 committed and MLA-b512 rounds done -> Codex B12b"
cd "$WT" && timeout 14400 codex exec -m gpt-5.6-luna -c model_reasoning_effort=max \
  --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check "$(cat "$HERE/codex_tasks/b12b_k3_prefill_fidelity.md")" \
  < /dev/null > "$HERE/codex_runs/b12b.log" 2>&1
echo "==== $(date +%F_%T) B12b codex exited $? ; HEAD $(git -C "$WT" log --oneline -1)"
if [ -n "$(git -C "$WT" status --short | grep -v '^??')" ]; then
  echo "!! B12b left an uncommitted tree; committing as-is is NOT done automatically (needs review) -> baking HEAD"
fi
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
  echo "!! prefill oracle bake failed (k3_prefill_bake.log) -> prefill rounds will keep waiting for 8805/8806"; exit 1
fi
echo "==== $(date +%F_%T) handing over to run_k3_prefill_when_free.sh $K0 $N"
exec "$HERE/run_k3_prefill_when_free.sh" "$K0" "$N"
