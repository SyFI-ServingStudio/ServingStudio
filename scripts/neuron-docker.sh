#!/usr/bin/env bash
# A workspace-owned Docker daemon with a Unix socket and no network configuration.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$workspace_root/.env"
mode=${1:-status}
case "$mode" in check|start|foreground|status) ;; *) echo 'Usage: neuron-docker.sh check|start|status' >&2; exit 2 ;; esac
runtime_dir=$(realpath -m "${SERVINGSTUDIO_NEURON_DOCKER_DIR:-$TMPDIR/neuron-docker}")
session_name=${SERVINGSTUDIO_NEURON_DOCKER_SESSION:-servingstudio-neuron-docker}
socket_uri="unix://$runtime_dir/docker.sock"
config_path="$runtime_dir/daemon.json"
log_path="$runtime_dir/dockerd.log"
docker_command=(sudo -n docker --host "$socket_uri")

if [[ "$mode" == status ]]; then
    actual_root=$("${docker_command[@]}" info --format '{{.DockerRootDir}}')
    [[ "$actual_root" == "$runtime_dir/data" ]] || { echo 'Docker data root does not match the selected workspace daemon.' >&2; exit 1; }
    printf 'Socket: %s\nData: %s\nLog: %s\n' "$socket_uri" "$actual_root" "$log_path"
    "${docker_command[@]}" ps --format '{{.ID}} {{.Status}} {{.Names}}'
    exit
fi

if [[ "$mode" == foreground ]]; then
    [[ -f "$config_path" ]] || { echo 'Run the start command to prepare this daemon.' >&2; exit 1; }
    ulimit -c 0
    exec >>"$log_path" 2>&1
    exec sudo -n env DOCKER_TMPDIR="$runtime_dir/tmp" TMPDIR="$runtime_dir/tmp" \
        dockerd --config-file "$config_path"
fi

command -v dockerd >/dev/null
command -v docker >/dev/null
command -v tmux >/dev/null
for unit in docker.service docker.socket containerd.service podman.service; do
    if systemctl is-active --quiet "$unit"; then
        echo "Existing service is active: $unit. Select its socket explicitly for image operations." >&2
        exit 1
    fi
done
if pgrep -x dockerd >/dev/null || pgrep -x containerd >/dev/null; then
    echo 'An existing Docker/containerd process is active; use its intended socket explicitly.' >&2
    exit 1
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Existing tmux session: $session_name" >&2
    exit 1
fi
for path in "$runtime_dir/docker.sock" "$runtime_dir/dockerd.pid" /run/docker.sock /run/containerd/containerd.sock; do
    [[ ! -e "$path" ]] || { echo "Existing socket/pid path: $path" >&2; exit 1; }
done
printf 'Session: %s\nCwd: %s\nLog: %s\nSocket: %s\n' "$session_name" "$workspace_root" "$log_path" "$socket_uri"
[[ "$mode" == start ]] || exit 0

umask 077
mkdir -p "$runtime_dir/tmp"
python3 - "$runtime_dir" "$config_path" <<'PY'
import json
from pathlib import Path
import sys

runtime = Path(sys.argv[1])
config = {
    "hosts": ["unix://" + str(runtime / "docker.sock")],
    "group": "root",
    "data-root": str(runtime / "data"),
    "exec-root": str(runtime / "exec"),
    "pidfile": str(runtime / "dockerd.pid"),
    "containerd-namespace": "servingstudio-neuron",
    "containerd-plugins-namespace": "servingstudio-neuron-plugins",
    "storage-driver": "overlay2",
    "features": {"containerd-snapshotter": False},
    "bridge": "none",
    "iptables": False,
    "ip6tables": False,
    "ip-forward": False,
    "ip-masq": False,
    "ipv6": False,
    "userland-proxy": False,
    "labels": ["servingstudio.task=neuron"],
    "max-concurrent-downloads": 2,
}
path = Path(sys.argv[2])
if path.exists():
    if json.loads(path.read_text()) != config:
        raise ValueError("Existing daemon configuration differs; preserve it and select a fresh directory")
else:
    with path.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
PY
sudo -n dockerd --validate --config-file "$config_path"
ss -ltn >"$runtime_dir/tcp-before.txt"
ip -j link show >"$runtime_dir/links-before.json"
cp /proc/sys/net/ipv4/ip_forward "$runtime_dir/ip-forward-before.txt"
cp /proc/sys/net/ipv6/conf/all/forwarding "$runtime_dir/ip6-forward-before.txt"
printf -v launch_command 'exec env SERVINGSTUDIO_NEURON_DOCKER_DIR=%q bash %q foreground' \
    "$runtime_dir" "$workspace_root/scripts/neuron-docker.sh"
tmux new-session -d -s "$session_name" -c "$workspace_root" "$launch_command"
for attempt in {1..30}; do
    if "${docker_command[@]}" info --format '{{.DockerRootDir}}' >"$runtime_dir/data-root.txt" 2>/dev/null; then
        [[ $(cat "$runtime_dir/data-root.txt") == "$runtime_dir/data" ]]
        ss -ltn >"$runtime_dir/tcp-after.txt"
        ip -j link show >"$runtime_dir/links-after.json"
        cmp "$runtime_dir/tcp-before.txt" "$runtime_dir/tcp-after.txt"
        cmp "$runtime_dir/links-before.json" "$runtime_dir/links-after.json"
        cmp "$runtime_dir/ip-forward-before.txt" /proc/sys/net/ipv4/ip_forward
        cmp "$runtime_dir/ip6-forward-before.txt" /proc/sys/net/ipv6/conf/all/forwarding
        echo "Workspace daemon ready: $socket_uri"
        exit
    fi
    sleep 1
done
echo "Daemon readiness not observed; inspect $log_path and tmux $session_name before retrying." >&2
exit 1
