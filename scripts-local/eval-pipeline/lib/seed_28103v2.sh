#!/usr/bin/env bash
# Seed the 28103-v2 implementation workspace from the 28103 eval base.
# This is a DEV workspace (not an eval fixture): keep .git so the sealed
# baseline + v2 work are one history. Reset to the sealed before-state.
set -euo pipefail
WS=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
SRC="$WS/agent-workspaces/w_45893bc43b51/repo"
DST="$WS/agent-workspaces/w_e0e8cb3f76d6/repo"
echo "[seed] $(date -Is) start"
rsync -a --delete --exclude='logs/' --exclude='tmp/' "$SRC/" "$DST/"
cd "$DST"
git checkout -qf main
git reset --hard 55de3a6   # sealed 28103 before-state
git for-each-ref --format='%(refname:short)' refs/heads | grep -vx main | xargs -r git branch -D
git clean -fdqx -e profiling -e target -e .venv
mkdir -p logs
git checkout -qb v2/kernel-capability
echo "[seed] $(date -Is) ready on branch v2/kernel-capability @ $(git rev-parse --short HEAD)"
