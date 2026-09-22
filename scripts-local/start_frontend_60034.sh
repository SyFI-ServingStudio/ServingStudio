#!/bin/bash
# Start the VibeSim Vite frontend on Yile's alternate port 60034.
# Proxy targets MUST be set: the defaults point at the other account's
# 8787/8765 instance. CHOKIDAR polling is required because the host's
# inotify instance quota is exhausted (no root to raise it).
set -u
export PATH="$HOME/.local/node22/bin:$PATH"
workspace_root=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
cd "$workspace_root/viz-ui/app"

export ANALYZER_PROXY_TARGET='http://172.17.0.1:8788'
export CONVERSATION_PROXY_TARGET='http://172.17.0.1:8766'
export VIBESIM_UI_HOST='0.0.0.0'
export CHOKIDAR_USEPOLLING=true
export CHOKIDAR_INTERVAL=2000

exec npx vite --mode live --port 60034 \
  >> "$workspace_root/scripts-local/frontend_60034.log" 2>&1
