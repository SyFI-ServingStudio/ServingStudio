#!/usr/bin/env bash
# Internal tmux entry point; start the stack through start-services.sh.
set -euo pipefail
source "$(dirname -- "$0")/service-env.sh"
case "${1:-}" in
    backend)
        exec >"$TMPDIR/backend.log" 2>&1
        cd "$workspace_root/VibeSimAgent"
        export ANALYZER_MCP_BASE_URL="http://host.docker.internal:$ANALYZER_PORT"
        export VIBESIM_MANAGED_BACKEND_URL="http://host.docker.internal:$AGENT_PORT"
        export HOST="$VIBESIM_API_BIND" PORT="$AGENT_PORT"
        export FRONTEND_SKIP_BUILD=1 CODEX_SKIP_IMAGE_BUILD=1
        exec ./run.sh ;;
    analyzer)
        exec >"$TMPDIR/analyzer.log" 2>&1
        cd "$workspace_root/VibeSim"
        exec target/release/analyze serve --bind "$VIBESIM_API_BIND:$ANALYZER_PORT" \
            --workspace-registry "$VIBESIM_WORKSPACES_ROOT/registry.json" ;;
    frontend)
        exec >"$TMPDIR/ui.log" 2>&1
        cd "$workspace_root/VibeSimUI/app"
        export ANALYZER_PROXY_TARGET="http://$VIBESIM_API_BIND:$ANALYZER_PORT"
        export CONVERSATION_PROXY_TARGET="http://$VIBESIM_API_BIND:$AGENT_PORT"
        export VIBESIM_UI_HOST=${VIBESIM_UI_HOST:-0.0.0.0}
        exec npm run dev:live -- --port "$UI_PORT" --strictPort ;;
    *) echo "Expected backend, analyzer or frontend" >&2; exit 2 ;;
esac
