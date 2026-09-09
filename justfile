set shell := ["bash", "-uc"]

workspace_root := justfile_directory()
task_tmp_dir := env_var("TMPDIR")

# Check only host-owned prerequisites. GPU and credentials have separate checks
# because a CPU-only UI/API deployment may intentionally omit them.
check-tools:
    @for command_name in git uv cargo rustc node npm docker tmux curl; do \
      command -v "$command_name" >/dev/null || { echo "missing: $command_name" >&2; exit 1; }; \
    done
    @test -n "{{task_tmp_dir}}" && test -d "{{task_tmp_dir}}" && test -w "{{task_tmp_dir}}" || \
      { echo "TMPDIR must name an existing writable directory: {{task_tmp_dir}}" >&2; exit 1; }
    @node_major=$(node --version | sed -E 's/^v([0-9]+).*/\1/'); \
      test "$node_major" = 22 || { echo "Node.js 22 required, found $(node --version)" >&2; exit 1; }
    @echo "host tools ready"

check-agent-auth:
    @test -r "$HOME/.codex/auth.json" || { echo "missing Codex auth: $HOME/.codex/auth.json" >&2; exit 1; }
    @codex --version

check-gpu:
    @docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

# Pull the meta-repo without silently advancing component branch tips. The
# recorded gitlinks remain the deployment source of truth.
pull:
    git pull --ff-only
    git submodule sync --recursive
    git submodule update --init --recursive

build-analyzer:
    cd "{{workspace_root}}/VibeSim" && cargo build -p analyzer --release

sync-agent:
    cd "{{workspace_root}}/VibeSimAgent" && \
      UV_CACHE_DIR="{{task_tmp_dir}}/uv-cache-user-facing-ui" uv sync

build-ui:
    cd "{{workspace_root}}/VibeSimUI/app" && npm ci && npm run build

build-intro:
    cd "{{workspace_root}}/vibesim-intro" && npm ci && npm run build:single

build: build-analyzer sync-agent build-ui build-intro

build-runner-image:
    cd "{{workspace_root}}/VibeSimAgent" && \
      CODEX_FORCE_IMAGE_BUILD=1 ./scripts/build-codex-runner-image.sh

smoke base_url="http://127.0.0.1:60033":
    @for endpoint in / /api/workspaces /api/v1/runs /api/v1/predictions \
      /api/v1/kernel-profiles /api/v1/kernel-measurements; do \
      status=$(curl -sS -o "{{task_tmp_dir}}/vibesim-smoke-response" -w '%{http_code}' "{{base_url}}$endpoint"); \
      printf '%s %s\n' "$status" "$endpoint"; \
      test "$status" = 200; \
    done
