#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/service-env.sh"
case "${1:-}" in
    init) validate_service_config; cd "$VIBESIM_AGENT_DIR"; exec uv run --frozen python -m vibesim_agent init ;;
    check-ports) python3 "$workspace_root/scripts/check-ports.py" ;;
    stop)
        if tmux -S "$service_socket" has-session -t backend 2>/dev/null; then
            tmux -S "$service_socket" send-keys -t backend C-c
            for attempt in $(seq 1 60); do
                tmux -S "$service_socket" has-session -t backend 2>/dev/null || break
                sleep 1
            done
            if tmux -S "$service_socket" has-session -t backend 2>/dev/null; then
                echo "Backend is still draining; inspect $TMPDIR/backend.log." >&2
                exit 1
            fi
        fi
        for service in frontend analyzer backend; do
            if tmux -S "$service_socket" has-session -t "$service" 2>/dev/null; then
                tmux -S "$service_socket" kill-session -t "$service"
            fi
        done ;;
    restart)
        validate_service_config
        bash "$workspace_root/scripts/services.sh" stop
        exec bash "$workspace_root/scripts/start-services.sh" ;;
    status) tmux -S "$service_socket" list-sessions ;;
    smoke) cd "$workspace_root"; exec just smoke "http://127.0.0.1:$UI_PORT" ;;
    *) echo "Expected init, check-ports, status, stop, restart or smoke" >&2; exit 2 ;;
esac
