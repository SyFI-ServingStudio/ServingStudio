#!/bin/bash
# Start the VibeSim Analyzer on Yile's alternate port 8788.
set -u
workspace_root=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
cd "$workspace_root/main"
exec ./target/release/analyze serve \
  --bind 172.17.0.1:8788 \
  --workspace-registry "$workspace_root/agent-workspaces/registry.json" \
  >> "$workspace_root/scripts-local/analyzer_8788.log" 2>&1
