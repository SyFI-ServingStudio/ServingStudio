#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/service-env.sh"
case "${1:-}" in
    check-ports) python3 "$workspace_root/scripts/check-ports.py" ;;
    stop)
        for service in frontend analyzer backend; do
            if tmux -S "$service_socket" has-session -t "$service" 2>/dev/null; then
                tmux -S "$service_socket" kill-session -t "$service"
            fi
        done ;;
    status) tmux -S "$service_socket" list-sessions ;;
    smoke) cd "$workspace_root"; exec just smoke "http://127.0.0.1:$UI_PORT" ;;
    *) echo "Expected check-ports, status, stop or smoke" >&2; exit 2 ;;
esac
