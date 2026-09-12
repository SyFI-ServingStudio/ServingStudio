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
export VIBESIM_AGENT_DIR=${VIBESIM_AGENT_DIR:-$workspace_root/ServingStudioAgent}
export VIBESIM_SIM_DIR=${VIBESIM_SIM_DIR:-$workspace_root/ServingStudioSim}
export VIBESIM_UI_DIR=${VIBESIM_UI_DIR:-$workspace_root/ServingStudioUI/app}
export VIBESIM_ANALYZER_BIN=${VIBESIM_ANALYZER_BIN:-$VIBESIM_SIM_DIR/target/release/analyze}
export VIBESIM_AGENT_MAIN_DIR=${VIBESIM_AGENT_MAIN_DIR:-$VIBESIM_SIM_DIR}
export VIBESIM_AGENT_WORKSPACES_ROOT=${VIBESIM_AGENT_WORKSPACES_ROOT:-$workspace_root/agent-workspaces}
export VIBESIM_RUNNER_IMAGE=${VIBESIM_RUNNER_IMAGE:-vibesim-agent-runner:$(id -un)}
service_socket="$TMPDIR/vibesim-services.sock"

# Status and stop must remain available even when a launch configuration broke.
validate_service_config() {
    local setting
    for setting in VIBESIM_AGENT_DIR VIBESIM_SIM_DIR VIBESIM_UI_DIR VIBESIM_ANALYZER_BIN \
        VIBESIM_AGENT_MAIN_DIR VIBESIM_AGENT_WORKSPACES_ROOT; do
        [[ ${!setting} = /* ]] || { echo "$setting must be an absolute path" >&2; return 1; }
    done
    if [[ -n ${VIBESIM_AGENT_PROVIDERS_FILE:-} ]]; then
        [[ $VIBESIM_AGENT_PROVIDERS_FILE = /* && -r $VIBESIM_AGENT_PROVIDERS_FILE ]] || {
            echo "VIBESIM_AGENT_PROVIDERS_FILE must name a readable absolute YAML path" >&2
            return 1
        }
    fi
}
