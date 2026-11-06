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
    preview)
        # A bundled UI on its own port: a few files instead of hundreds of dev
        # modules, for use over a slow link. Rebuilds and replaces on each run.
        validate_service_config
        port=${VIBESIM_UI_BUILD_PORT:-}
        [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1024 && port <= 65535 )) || {
            echo "Set VIBESIM_UI_BUILD_PORT in .env to a free port outside every user's range." >&2
            exit 1
        }
        (cd "$VIBESIM_UI_DIR" && npm run build)
        if tmux -S "$service_socket" has-session -t preview 2>/dev/null; then
            tmux -S "$service_socket" kill-session -t preview
            sleep 1
        fi
        python3 - "$port" <<'PY' || exit 1
import socket, sys
port = int(sys.argv[1])
with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("0.0.0.0", port))
    except OSError as error:
        raise SystemExit(f"Port {port} is in use ({error}); choose another VIBESIM_UI_BUILD_PORT.")
PY
        export VIBESIM_API_BIND=${VIBESIM_API_BIND:-$(docker network inspect bridge \
            --format '{{(index .IPAM.Config 0).Gateway}}')}
        printf -v preview_command 'bash %q preview' "$workspace_root/scripts/run-service.sh"
        tmux -S "$service_socket" new-session -d -s preview "$preview_command"
        for attempt in $(seq 1 30); do
            if (cd "$workspace_root" && just smoke "http://127.0.0.1:$port") \
                >"$TMPDIR/preview-smoke.log" 2>&1; then
                printf 'Built UI: http://<server>:%s\nLog: %s/ui-preview.log\n' "$port" "$TMPDIR"
                exit 0
            fi
            sleep 1
        done
        echo "Preview failed to start; inspect $TMPDIR/{ui-preview,preview-smoke}.log" >&2
        exit 1 ;;
    preview-stop)
        if tmux -S "$service_socket" has-session -t preview 2>/dev/null; then
            tmux -S "$service_socket" kill-session -t preview
        fi ;;
    status) tmux -S "$service_socket" list-sessions ;;
    smoke) cd "$workspace_root"; exec just smoke "http://127.0.0.1:$UI_PORT" ;;
    *) echo "Expected init, check-ports, status, stop, restart, smoke, preview or preview-stop" >&2; exit 2 ;;
esac
