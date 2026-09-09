#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/service-env.sh"
python3 "$workspace_root/scripts/check-ports.py"
if tmux -S "$service_socket" list-sessions >/dev/null 2>&1; then
    echo "Workspace service sessions already exist; inspect with just services-status." >&2
    exit 1
fi
export VIBESIM_API_BIND=${VIBESIM_API_BIND:-$(docker network inspect bridge \
    --format '{{(index .IPAM.Config 0).Gateway}}')}
[[ -n "$VIBESIM_API_BIND" ]] || { echo "Docker bridge has no gateway" >&2; exit 1; }
[[ -x "$workspace_root/VibeSim/target/release/analyze" ]] || {
    echo "Run just build first" >&2; exit 1;
}
docker image inspect "$CODEX_DOCKER_IMAGE" >/dev/null
mkdir -p "$VIBESIM_WORKSPACES_ROOT"
start_service() {
    local service_command
    printf -v service_command 'bash %q %q' "$workspace_root/scripts/run-service.sh" "$1"
    tmux -S "$service_socket" new-session -d -s "$1" "$service_command"
}
start_service backend
ready=0
for attempt in $(seq 1 30); do
    tmux -S "$service_socket" has-session -t backend 2>/dev/null || break
    if [[ -r "$VIBESIM_WORKSPACES_ROOT/registry.json" ]] && \
        curl --noproxy '*' -s -o /dev/null "http://$VIBESIM_API_BIND:$AGENT_PORT/api/workspaces"; then
        ready=1
        break
    fi
    sleep 1
done
[[ "$ready" = 1 ]] || { echo "Backend failed to start; inspect $TMPDIR/backend.log" >&2; exit 1; }
start_service analyzer
start_service frontend
ready=0
for attempt in $(seq 1 30); do
    if ! tmux -S "$service_socket" has-session -t analyzer 2>/dev/null || \
       ! tmux -S "$service_socket" has-session -t frontend 2>/dev/null; then
        break
    fi
    if (cd "$workspace_root" && just smoke "http://127.0.0.1:$UI_PORT") \
        >"$TMPDIR/startup-smoke.log" 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
[[ "$ready" = 1 ]] || {
    echo "Service startup failed; inspect $TMPDIR/{backend,analyzer,ui,startup-smoke}.log" >&2
    exit 1
}
printf 'UI: http://<server>:%s\nLogs: %s/{backend,analyzer,ui}.log\n' "$UI_PORT" "$TMPDIR"
