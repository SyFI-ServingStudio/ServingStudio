#!/usr/bin/env bash
# Shared configuration; source this file from the service entry points.
workspace_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -f "$workspace_root/.env" ]]; then
    echo "Run just setup-env first." >&2
    return 1
fi
set -a
source "$workspace_root/.env"
set +a
export VIBESIM_PORT_BASE=${VIBESIM_PORT_BASE:-$((60030 + 3 * $(id -u)))}
export UI_PORT=$VIBESIM_PORT_BASE
export AGENT_PORT=$((UI_PORT + 1)) ANALYZER_PORT=$((UI_PORT + 2))
export CODEX_DOCKER_IMAGE=${CODEX_DOCKER_IMAGE:-vibesim-ui-codex-runner:$(id -un)}
export VIBESIM_WORKSPACES_ROOT="$workspace_root/agent-workspaces"
service_socket="$TMPDIR/vibesim-services.sock"
