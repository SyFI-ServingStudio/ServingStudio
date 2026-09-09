set shell := ["bash", "-euc"]
set dotenv-load := true
set dotenv-override := true

workspace_root := justfile_directory()
configured_tmp_dir := env_var_or_default("TMPDIR", "tmp")
configured_uv_cache := env_var_or_default("UV_CACHE_DIR", "uv-cache")
export TMPDIR := if configured_tmp_dir =~ '^/' { configured_tmp_dir } else { workspace_root / configured_tmp_dir }
export UV_CACHE_DIR := if configured_uv_cache =~ '^/' { configured_uv_cache } else { workspace_root / configured_uv_cache }
task_tmp_dir := TMPDIR

# Initialize local paths; preserve unrelated .env settings. Re-run after moving the workspace.
setup-env:
    @python3 "{{workspace_root}}/scripts/setup-env.py" "{{workspace_root}}"

# Check workspace-local scratch/cache settings before installing or building.
check-env:
    @mkdir -p "{{task_tmp_dir}}" && test -w "{{task_tmp_dir}}" || \
      { echo "Set TMPDIR to a writable user-owned directory in .env or your shell." >&2; exit 1; }
    @mkdir -p "$UV_CACHE_DIR" && test -w "$UV_CACHE_DIR" || \
      { echo "Set UV_CACHE_DIR to a writable user-owned directory; do not run uv as root." >&2; exit 1; }
    @tmp_probe=$(mktemp "{{task_tmp_dir}}/vibesim-check.XXXXXX") && rm "$tmp_probe"
    @cache_probe=$(mktemp "$UV_CACHE_DIR/vibesim-check.XXXXXX") && rm "$cache_probe"

# Host tools for building the web stack; Docker/GPU/auth are separate checks.
check-tools: check-env
    @for command_name in git uv cargo rustc node npm python3 tmux curl cc c++ make cmake ninja pkg-config protoc mold; do \
      command -v "$command_name" >/dev/null || { echo "missing: $command_name (see reproduce.md bootstrap)" >&2; exit 1; }; \
    done
    @node_major=$(node --version | sed -E 's/^v([0-9]+).*/\1/'); \
      test "$node_major" = 22 || { echo "Node.js 22 required, found $(node --version)" >&2; exit 1; }
    @echo "host build tools ready"

# Reject incomplete recursive checkouts before Cargo follows path dependencies.
check-submodules:
    @status=$(git -C "{{workspace_root}}" submodule status --recursive); \
      if printf '%s\n' "$status" | grep -Eq '^[-+U]'; then \
        echo "Submodules are uninitialized or differ from the recorded revisions. Inspect local work before syncing them." >&2; \
        exit 1; \
      fi

check-docker:
    @command -v docker >/dev/null || { echo "Install Docker Engine and grant this user daemon access (see reproduce.md)." >&2; exit 1; }
    @docker info >/dev/null

check-agent-auth:
    @command -v codex >/dev/null || { echo "Install and authenticate the Codex CLI (see reproduce.md)." >&2; exit 1; }
    @test -r "$HOME/.codex/auth.json" || { echo "missing Codex auth: $HOME/.codex/auth.json" >&2; exit 1; }
    @codex --version

check-gpu: check-docker
    @docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

# Materialize recorded component commits without advancing branch tips.
pull:
    git pull --ff-only
    git submodule sync --recursive
    git submodule update --init --recursive

build-analyzer: check-env
    cd "{{workspace_root}}/VibeSim" && cargo build --locked -p analyzer --release

sync-agent: check-env
    cd "{{workspace_root}}/VibeSimAgent" && uv sync --frozen

build-ui: check-env
    cd "{{workspace_root}}/VibeSimUI/app" && npm ci && npm run build

build-intro: check-env
    cd "{{workspace_root}}/vibesim-intro" && npm ci && npm run build

# CPU web stack; no Agent runner image, GPU, or model credentials required.
build: check-tools check-submodules build-analyzer sync-agent build-ui build-intro

# Optional host simulator/profiling environment, including CUDA Python packages.
sync-simulator: check-env check-submodules
    cd "{{workspace_root}}/VibeSim" && just sync

# Expensive: build the runner image and run its non-GPU acceptance test.
build-runner-image: check-tools check-submodules check-docker
    cd "{{workspace_root}}/VibeSimAgent" && \
      CODEX_FORCE_IMAGE_BUILD=1 ./scripts/build-codex-runner-image.sh

smoke base_url="http://127.0.0.1:60033": check-env
    @for endpoint in / /api/workspaces /api/v1/runs /api/v1/predictions \
      /api/v1/kernel-profiles /api/v1/kernel-measurements; do \
      auth_args=(); if [[ -n "${VIBESIM_API_TOKEN:-}" ]]; then auth_args=(-H "Authorization: Bearer $VIBESIM_API_TOKEN"); fi; \
      status=$(curl --noproxy '*' "${auth_args[@]}" -sS -o "{{task_tmp_dir}}/vibesim-smoke-response" -w '%{http_code}' "{{base_url}}$endpoint"); \
      printf '%s %s\n' "$status" "$endpoint"; \
      test "$status" = 200 || exit 1; \
    done

# Check the configured per-user ports before starting anything.
check-ports: check-env
    @bash "{{workspace_root}}/scripts/services.sh" check-ports

# Start one backend, Analyzer and UI in workspace-specific tmux sessions.
start: check-env check-docker
    @bash "{{workspace_root}}/scripts/start-services.sh"

services-status:
    @bash "{{workspace_root}}/scripts/services.sh" status

# Smoke-test the configured UI port.
smoke-local: check-env
    @bash "{{workspace_root}}/scripts/services.sh" smoke

# Stop only this workspace's three service sessions.
stop:
    @bash "{{workspace_root}}/scripts/services.sh" stop
