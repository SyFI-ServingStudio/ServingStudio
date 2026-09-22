#!/usr/bin/env bash
set -euo pipefail
WS=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
SRC="$WS/agent-workspaces/w_1a73838d022f/repo"
DST="$WS/agent-workspaces/w_9169afebff09/repo"
echo "[seed] $(date -Is) start"
rsync -a --delete --exclude='logs/' --exclude='tmp/' "$SRC/" "$DST/"
cd "$DST"
git checkout -qf main
git reset --hard 01bba52   # sealed 27931 before-state
git for-each-ref --format='%(refname:short)' refs/heads | grep -vx main | xargs -r git branch -D
git clean -fdqx -e profiling -e target -e .venv
mkdir -p logs
git checkout -qb v3/sealed
echo "[seed] $(date -Is) ready on v3/sealed @ $(git rev-parse --short HEAD)"
