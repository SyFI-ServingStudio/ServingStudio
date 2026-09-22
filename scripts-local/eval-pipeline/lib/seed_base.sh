#!/usr/bin/env bash
# Seed the eval base workspace repo from the known-good 28103 workspace clone,
# stripping spoilers that would leak the PR #28103 answer to the agent.
# Cache-only: copies the warm profile.db + prebuilt target/ + .venv so the
# agent's rebuild is incremental and GPU-free. No token is written here.
set -euo pipefail

WS=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
SRC="$WS/agent-workspaces/w_1471ccd48dec/repo"
DST="$WS/agent-workspaces/w_45893bc43b51/repo"

echo "[seed] $(date -Is) start  SRC=$SRC  DST=$DST"

# Clean the fresh main-based repo the backend created, then mirror the source.
# Spoilers excluded:
#   logs/      -> the *_after answer runs + before scoped reports (agent runs its own)
#   reports/   -> optimality_scoped_pr28103_validation.md and other 28103 analysis
#   tmp/       -> tmp/*/vllm-pr-28103 = the actual PR source tree
#   .git/      -> commit history references the PR/after work
rsync -a --delete \
  --exclude='logs/' \
  --exclude='reports/' \
  --exclude='tmp/' \
  --exclude='.git/' \
  "$SRC/" "$DST/"

echo "[seed] $(date -Is) rsync done; placing clean before-state config"

# Clean before-state config the agent is handed (step 2: before profile+config).
mkdir -p "$DST/eval-configs" "$DST/logs"
# before.json arch/type stays qwen3_dense_vllm_before; log_dir -> a fresh dir;
# cases.json copied alongside. Sourced from the recorded before run.
python3 - "$SRC" "$DST" <<'PY'
import json, sys, pathlib
src, dst = map(pathlib.Path, sys.argv[1:3])
bef = json.loads((src/"logs/20260809_3_qwen3_vllm_before/before.json").read_text())
bef["log_dir"] = "logs/eval_before_run"
(dst/"eval-configs/before.json").write_text(json.dumps(bef, indent=2))
cases = (src/"logs/20260809_3_qwen3_vllm_before/cases.json").read_text()
(dst/"eval-configs/cases.json").write_text(cases)
# before.json's cases_file is relative to its own dir, so keep them together.
b = json.loads((dst/"eval-configs/before.json").read_text())
b["cases_file"] = "cases.json"
(dst/"eval-configs/before.json").write_text(json.dumps(b, indent=2))
print("[seed] wrote eval-configs/before.json + cases.json; arch.iter.type =",
      b.get("arch",{}).get("iter",{}).get("type"))
PY

# Fresh git baseline so the agent starts clean and we can reset between trials.
cd "$DST"
git init -q
git add -A
git -c user.email=eval@local -c user.name=eval commit -qm "eval base: 28103 before-state (spoilers stripped)"
echo "[seed] $(date -Is) committed clean baseline"
du -sh "$DST" 2>/dev/null | sed 's/^/[seed] size /'
echo "[seed] DONE"
