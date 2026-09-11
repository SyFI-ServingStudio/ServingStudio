#!/usr/bin/env bash
# Internal tmux entry point; start the stack through start-services.sh.
set -euo pipefail
source "$(dirname -- "$0")/service-env.sh"
validate_service_config
case "${1:-}" in
    backend)
        exec >"$TMPDIR/backend.log" 2>&1
        cd "$VIBESIM_AGENT_DIR"
        export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-${OPENROUTE_KEY:-}}
        unset OPENROUTE_KEY
        export VIBESIM_AGENT_ANALYZER_BASE_URL=${VIBESIM_AGENT_ANALYZER_BASE_URL:-http://host.docker.internal:$ANALYZER_PORT}
        export VIBESIM_AGENT_MANAGED_BACKEND_URL=${VIBESIM_AGENT_MANAGED_BACKEND_URL:-http://host.docker.internal:$AGENT_PORT}
        export VIBESIM_AGENT_BIND="$VIBESIM_API_BIND" VIBESIM_AGENT_PORT="$AGENT_PORT"
        exec uv run --frozen python -m vibesim_agent serve ;;
    analyzer)
        exec >"$TMPDIR/analyzer.log" 2>&1
        cd "$VIBESIM_AGENT_MAIN_DIR"
        exec "$VIBESIM_ANALYZER_BIN" serve --bind "$VIBESIM_API_BIND:$ANALYZER_PORT" \
            --workspace-registry "$VIBESIM_AGENT_WORKSPACES_ROOT/registry.json" ;;
    frontend)
        exec >"$TMPDIR/ui.log" 2>&1
        cd "$VIBESIM_UI_DIR"
        export ANALYZER_PROXY_TARGET="http://$VIBESIM_API_BIND:$ANALYZER_PORT"
        export CONVERSATION_PROXY_TARGET="http://$VIBESIM_API_BIND:$AGENT_PORT"
        export VIBESIM_UI_HOST=${VIBESIM_UI_HOST:-0.0.0.0}
        exec npm run dev -- --port "$UI_PORT" --strictPort ;;
    *) echo "Expected backend, analyzer or frontend" >&2; exit 2 ;;
esac
