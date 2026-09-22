#!/usr/bin/env bash
set -euo pipefail
WS=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
SRC="$WS/agent-workspaces/w_f2c002bbcd89/repo"
DST="$WS/agent-workspaces/w_31e2e24aa68d/repo"
echo "[seed] $(date -Is) start"
rsync -a --delete --exclude='logs/' --exclude='tmp/' "$SRC/" "$DST/"
cd "$DST"
git checkout -qf main
git reset --hard 632d0a7    # sealed 30947 before-state
git for-each-ref --format='%(refname:short)' refs/heads | grep -vx main | xargs -r git branch -D
git clean -fdqx -e profiling -e target -e .venv
mkdir -p logs
echo "[seed] $(date -Is) ready @ $(git rev-parse --short HEAD)"
