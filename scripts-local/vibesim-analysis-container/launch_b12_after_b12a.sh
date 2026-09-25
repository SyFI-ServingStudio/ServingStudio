#!/usr/bin/env bash
# Launch Codex B12 (K3 prefill support) in the worktree once no codex process is running.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "[c]odex exec" >/dev/null; do sleep 60; done
echo "==== $(date +%F_%T) launching B12"
cd /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust && timeout 21600 codex exec -m gpt-5.6-luna -c model_reasoning_effort=max \
  --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check "$(cat "$HERE/codex_tasks/b12_k3_prefill_support.md")" \
  < /dev/null > "$HERE/codex_runs/b12.log" 2>&1
echo "==== $(date +%F_%T) B12 codex exited $?"
