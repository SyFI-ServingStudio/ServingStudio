#!/usr/bin/env bash
set -euo pipefail
WS=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
SRC="$WS/agent-workspaces/w_e0e8cb3f76d6/repo"   # v2 kernel-capability impl
DST="$WS/agent-workspaces/w_af3858cf8529/repo"
echo "[seed] $(date -Is) start"
rsync -a --delete --exclude='logs/' --exclude='tmp/' "$SRC/" "$DST/"
cd "$DST"
git checkout -qf v2/kernel-capability
git clean -fdqx -e profiling -e target -e .venv
mkdir -p logs
git checkout -qB main
echo "[seed] $(date -Is) ready @ $(git rev-parse --short HEAD)"
