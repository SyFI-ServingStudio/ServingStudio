#!/usr/bin/env bash
# Start the Opus 5.5 LiteLLM gateway at http://127.0.0.1:${PORT:-9104}/v1 (detached).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-9104}"
LITELLM_BIN="${LITELLM_BIN:-/raid/yilegu/openclaw-local-model/venvs/main/bin/litellm}"
CONFIG="${CONFIG:-$HERE/litellm_opus55_bedrock_config.yaml}"
LOG="$HERE/litellm_opus55_${PORT}.log"
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT already in use" >&2; exit 1; fi
# Shell init may point ANTHROPIC_* at another proxy; Bedrock routing doesn't need them.
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN
setsid nohup "$LITELLM_BIN" --config "$CONFIG" --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 < /dev/null &
echo "started pid $! on :$PORT (log $LOG)"
