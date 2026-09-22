#!/usr/bin/env bash
# vibesim analysis container entrypoint.
#   serve                 -> start the HTTP API (default)
#   cli <verb> [args...]  -> run the vibesim_api CLI once (from the repo dir)
#   bash                  -> interactive shell
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""        # zero-GPU, always
export VIBESIM_WS_ROOT="${VIBESIM_WS_ROOT:-/opt/vibesim}"
export VIBESIM_PREBAKED_RUN="${VIBESIM_PREBAKED_RUN:-before}"
export TMPDIR="${TMPDIR:-/tmp}"
cd "${VIBESIM_REPO:-/opt/vibesim/repo}"

cmd="${1:-serve}"; shift || true
case "$cmd" in
  serve) exec python3 /opt/vibesim/app/server.py ;;
  cli)   exec python3 /opt/vibesim/app/vibesim_api.py "$@" ;;
  bash)  exec /bin/bash ;;
  *)     exec python3 /opt/vibesim/app/vibesim_api.py "$cmd" "$@" ;;
esac
