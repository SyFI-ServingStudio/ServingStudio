#!/bin/bash
# Start the VibeSim backend (user-facing-ui) on Yile's alternate port 8766.
# Standard ports 8765/8787/60033 belong to another account on this host.
set -u
source "$HOME/.cargo/env" 2>/dev/null || true
export PATH="$HOME/.cargo/bin:$HOME/.local/node22/bin:$HOME/.local/protoc/bin:$PATH"
export TMPDIR="${TMPDIR:-/raid/tmp}"

workspace_root=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
cd "$workspace_root/user-facing-ui"

# MAIN_DIR/uv.lock (VibeSim/uv.lock) is not present in this workspace layout, so the
# backend's config.py would crash at import hashing it. Pin the lock SHA the current
# Codex runner image was baked with (containers gate on VIBESIM_EXPECTED_LOCK_SHA).
# Update this if the runner image is rebuilt against a new main/uv.lock.
export CODEX_MAIN_LOCK_SHA="${CODEX_MAIN_LOCK_SHA:-05544906b65cbc2814a5b69a1d0a980b142bed96fd6fbb4a6638bc9334df3108}"
# NOTE: to hard-disable GPU exposure to agent containers (strict zero-GPU pause),
# launch with CODEX_DOCKER_GPUS="" prefixed. Default (unset) exposes all GPUs.

export VIBESIM_WORKSPACES_ROOT="$workspace_root/agent-workspaces"
export ANALYZER_MCP_BASE_URL='http://host.docker.internal:8788'
export VIBESIM_MANAGED_BACKEND_URL='http://host.docker.internal:8766'
export UV_CACHE_DIR="$TMPDIR/uv-cache-user-facing-ui"

exec uv run uvicorn backend.app:app --host 172.17.0.1 --port 8766 \
  >> "$workspace_root/scripts-local/backend_8766.log" 2>&1
