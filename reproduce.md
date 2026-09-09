# VibeSim UI + Agent + Analyzer Reproduction Guide

This document is the tested deployment contract for the three repositories that
form the user-facing VibeSim application. Keep it at the common workspace root;
the relative repository names are part of the runtime contract.

## 1. Released repositories

The workspace repository records the exact verified commit of every component
as a Git submodule. Clone it recursively:

```bash
git clone --recurse-submodules \
  https://github.com/SyFI-VibeSim/VibeSimWorkspace.git vibesim-workspace
cd vibesim-workspace
```

For an existing clone, update the workspace first and then materialize the
commits recorded by that workspace revision:

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

The current verified component baseline is:

| Directory | Repository | Branch | Verified release commit |
| --- | --- | --- | --- |
| `VibeSim/` | `https://github.com/SyFI-VibeSim/VibeSim.git` | `master` | `48864532c291e5cd48d01efb515d63733496e420` |
| `VibeSimAgent/` | `https://github.com/SyFI-VibeSim/VibeSimAgent.git` | `agent-http-api` | `2c80deb6eedf4a99f27790e659cc057b7a085854` |
| `VibeSimUI/` | `https://github.com/SyFI-VibeSim/VibeSimUI.git` | `main` | `3edaf8d8be290bd9a234ad49099ce9cd5a4039ae` |

The GLM development branch is intentionally not part of this released baseline.
Do not run `git submodule update --remote` during deployment: that command moves
to branch tips instead of reproducing the SHAs recorded by the meta-repo.

## 2. Supported topology

The supported deployment is three host processes plus per-conversation Docker
runners:

```text
browser
  -> Vite (60033)
       -> Rust Analyzer (8787)
       -> FastAPI conversation backend (8765)
              -> one Docker Codex runner per active conversation
```

The three repositories must remain siblings. `VibeSimAgent` resolves `../VibeSim`,
`../agent-workspaces`, and `../VibeSimUI` from this layout. The Analyzer reads the
same `agent-workspaces/registry.json` that the backend maintains.

Running the FastAPI backend inside an ordinary container is not currently a
drop-in deployment: it creates sibling Docker runner containers and passes host
workspace paths to the Docker daemon. A containerized backend therefore needs
an explicit host-path translation or a real Docker-in-Docker deployment. The
tested path in this guide runs the three services on the Docker host.

## 3. Host requirements

Tested target: Ubuntu 24.04 on x86-64.

Install or provide:

- Git and CA certificates.
- Docker Engine with the current user allowed to access the daemon.
- NVIDIA driver and NVIDIA Container Toolkit when GPU profiling or timing is
  required. `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04
  nvidia-smi` must succeed.
- Node.js 22 and npm.
- `uv`.
- Rust stable (`cargo` and `rustc`).
- Native build tools: `build-essential`, `pkg-config`, `cmake`, `ninja-build`,
  and `protobuf-compiler`.
- `tmux` for durable services.
- Internet access for GitHub, container registries, npm, PyPI/uv, Rust crates,
  Node downloads, and Codex/OpenRouter endpoints.

Run the services as the non-root user that owns the common workspace. Do not
start Vite/backend as root: it gives generated files the wrong ownership,
changes the Codex runner home/UID contract, and can exhaust root's shared
inotify-instance quota on a busy Docker host.

Ubuntu bootstrap example:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates cmake curl git ninja-build \
  pkg-config protobuf-compiler tmux

curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install Node.js 22 using the organization's approved package source or nvm.
node --version
npm --version
uv --version
cargo --version
docker version
```

Do not hard-code `/tmp` for builds or logs. Set `TMPDIR` to a filesystem with
enough space before starting:

```bash
export TMPDIR=/path/to/large-volume/tmp
mkdir -p "$TMPDIR"
```

Capacity guidance: reserve at least 30 GB for the first runner-image build,
Cargo artifacts, npm modules, and uv caches. More is required for model caches
or multiple managed workspaces.

## 4. Credentials and configuration

### Required for source checkout

All four repositories are private. Configure GitHub HTTPS credentials (or use
equivalent SSH URLs) before cloning. With GitHub CLI:

```bash
gh auth login
gh auth setup-git
git ls-remote https://github.com/SyFI-VibeSim/VibeSimWorkspace.git HEAD
```

Use the organization's credential helper or secret store; never write a token
into `.gitmodules`, `.env`, `reproduce.md`, or a shell script.

### Required for Agent conversations

The backend copies selected entries from the host Codex home into a private
per-conversation Codex home. Configure Codex on the host first and verify that
the authentication file exists:

```bash
test -r "$HOME/.codex/auth.json"
codex --version
```

Do not copy the secret into a repository or into this document. An
`OPENAI_API_KEY` is not separately required when the Codex CLI authentication
file is valid; deployments using API-key-based Codex authentication should
follow the Codex CLI's own supported configuration.

### Recommended when exposed beyond localhost

Set a strong bearer token. Browser/Agent API callers must then send it as
`Authorization: Bearer ...`.

```bash
export VIBESIM_API_TOKEN='replace-with-a-secret'
```

### Optional

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Automatically names new workspaces and conversations. |
| `OPENROUTE_KEY` | Compatibility alias for `OPENROUTER_API_KEY`. |
| `HF_HOME` | Host Hugging Face cache; mounted read-only at `/model` in Agent runners. |
| `CODEX_MODEL` | Codex model override; defaults to the backend's configured model. |
| `CODEX_REASONING_EFFORT` | Reasoning effort override. |
| `VIBESIM_API_TOKEN` | Bearer auth for Agent-facing APIs. Strongly recommended off-host. |

`HF_HOME`, when set, must name an existing readable directory. OpenRouter is
not required for conversations; without it, automatic titles remain pending or
fall back to local names.

## 5. First build

From the common workspace root:

```bash
mkdir -p agent-workspaces

cd VibeSim
cargo build -p analyzer --release
uv sync
cd ..

cd VibeSimUI/app
npm ci
npm run build
cd ../..

cd VibeSimAgent
UV_CACHE_DIR="$TMPDIR/uv-cache-user-facing-ui" uv sync

# This is the expensive first-time step. It builds the CUDA/Codex runner image,
# prewarms the VibeSim Python environment, compiles the Cargo release seed, and
# runs the non-GPU runner-image acceptance test.
export CODEX_DOCKER_IMAGE="vibesim-ui-codex-runner:${USER}"
CODEX_FORCE_IMAGE_BUILD=1 ./scripts/build-codex-runner-image.sh
```

The user-specific image tag is required on a shared Docker daemon because the
runner records the building account's UID/GID and home path. Preserve this
environment variable when starting the backend below; do not rely on another
account's shared `latest` tag.

The workspace `justfile` wraps the already-tested build commands without hiding
system package or credential setup:

```bash
just check-tools
just build
just build-runner-image  # expensive; requires Docker and registry access
```

For a CPU-only UI/API smoke, set `CODEX_DOCKER_GPUS` to the empty string. That
does not make GPU profiling functional:

```bash
export CODEX_DOCKER_GPUS=
```

## 6. Start all services

The Agent containers resolve host services through `host.docker.internal`.
Bind Analyzer and backend to the default Docker bridge gateway, not to every
network interface:

```bash
workspace_root="$PWD"
docker_bridge_gateway=$(docker network inspect bridge \
  --format '{{(index .IPAM.Config 0).Gateway}}')
test -n "$docker_bridge_gateway"

service_socket="$TMPDIR/vibesim-services.sock"

# Backend starts first because it owns and initializes registry.json.
tmux -S "$service_socket" new-session -d -s backend \
  "cd '$workspace_root/VibeSimAgent' && \
   export VIBESIM_WORKSPACES_ROOT='$workspace_root/agent-workspaces' && \
   export ANALYZER_MCP_BASE_URL='http://host.docker.internal:8787' && \
   export VIBESIM_MANAGED_BACKEND_URL='http://host.docker.internal:8765' && \
   export UV_CACHE_DIR='$TMPDIR/uv-cache-user-facing-ui' && \
   export HOST='$docker_bridge_gateway' PORT=8765 FRONTEND_SKIP_BUILD=1 \
     CODEX_SKIP_IMAGE_BUILD=1 && \
   exec ./run.sh"

for attempt in $(seq 1 30); do
  test -r "$workspace_root/agent-workspaces/registry.json" && break
  sleep 1
done
test -r "$workspace_root/agent-workspaces/registry.json"

tmux -S "$service_socket" new-session -d -s analyzer \
  "cd '$workspace_root/VibeSim' && exec target/release/analyze serve \
    --bind '$docker_bridge_gateway:8787' \
    --workspace-registry '$workspace_root/agent-workspaces/registry.json'"

tmux -S "$service_socket" new-session -d -s frontend \
  "cd '$workspace_root/VibeSimUI/app' && \
   export ANALYZER_PROXY_TARGET='http://$docker_bridge_gateway:8787' && \
   export CONVERSATION_PROXY_TARGET='http://$docker_bridge_gateway:8765' && \
   export VIBESIM_UI_HOST='0.0.0.0' && \
   exec npm run dev:live -- --port 60033"
```

For access by hostname, set `VIBESIM_UI_ALLOWED_HOSTS` to a comma-separated
allowlist before starting Vite. Do not use an unrestricted host allowlist.

## 7. Smoke checks

Run through the browser-facing Vite origin so the checks exercise both proxy
paths:

```bash
curl --fail --silent --show-error http://127.0.0.1:60033/ >/dev/null
curl --fail --silent --show-error http://127.0.0.1:60033/api/workspaces
curl --fail --silent --show-error http://127.0.0.1:60033/api/v1/runs
curl --fail --silent --show-error http://127.0.0.1:60033/api/v1/predictions
curl --fail --silent --show-error http://127.0.0.1:60033/api/v1/kernel-profiles
curl --fail --silent --show-error http://127.0.0.1:60033/api/v1/kernel-measurements
```

The same checks are available as `just smoke`.

Runner-image acceptance:

```bash
cd VibeSimAgent
./scripts/test-codex-runner-image.sh build

# Optional GPU acceptance with a warm kernel catalog:
./scripts/test-codex-runner-image.sh timing
```

The final end-to-end gate is one real Agent conversation in workspace `w_main`,
followed by a page refresh while the turn is running. Verify that SSE reconnects,
the answer persists, and any managed simulation/timing/profile result appears in
the Page 0 result catalog and opens through Analyzer.

## 8. Clean-container bootstrap test

A plain Ubuntu container can reproduce source checkout, dependency installation,
Analyzer build, backend import/tests, and Vite build. It cannot prove sibling
Agent-runner creation unless Docker-in-Docker or correct host-path propagation is
configured. The validation log below records exactly which boundary was tested.

## 9. Validation log

Update this section whenever the procedure changes.

- 2026-08-01: release SHAs were read back from GitHub after push.
- 2026-08-01: clean remote clone reproduced all three release SHAs.
- 2026-08-01: a dedicated meta-repo was created with all three components as
  branch-annotated submodules pinned to those verified SHAs.
- 2026-08-01: an older clean meta-repo checkout fast-forwarded with
  `git pull --ff-only`; recursive sync/update retained the pinned component
  SHAs and initialized the nested TraceLab/vLLM submodules.
- 2026-08-01: all user-facing README files were normalized to English and
  updated for the Analyzer-owned result boundary; production Agent contracts
  were updated and the meta-repo advanced to the verified component commits.
- 2026-08-01: naive Ubuntu 24.04 container installed Git 2.43, uv 0.12.1,
  Node 22.14.0/npm 10.9.2, Rust stable, protoc 3.21.12, and mold 2.30.
- 2026-08-01: clean VibeSimUI `npm ci && npm run build` passed. npm reported
  2 moderate and 2 high audit findings; Vite also reported a >500 kB chunk.
- 2026-08-01: clean Agent sync selected CPython 3.12.13; 50 tests and 5
  subtests passed. Two FastAPI `on_event` deprecation warnings remain.
- 2026-08-01: clean Analyzer release build passed in about 3m45s.
- 2026-08-01: clean container three-service smoke passed through the Vite
  same-origin entry: `/`, workspaces, runs, predictions, kernel profiles, and
  kernel measurements all returned HTTP 200.
- 2026-08-01: clean Codex runner image build and non-GPU acceptance passed with
  an isolated image tag. The gate verified non-root execution, mold/protoc,
  simulator and Analyzer release-seed reuse, and the Launcher cache report.
- 2026-08-07: the baseline advanced to the alignment prediction pairing. All
  three release SHAs were read back from GitHub with `git ls-remote` after push
  and match the table in section 1.
- 2026-08-07: `just build` passed on this baseline in 4m25s — Analyzer release
  build, Agent `uv sync`, and UI `npm ci && npm run build`. Vite still reports
  the same >500 kB chunk.
- 2026-08-07: component test suites passed on the pinned SHAs — Analyzer
  `cargo test -p analyzer` 173 tests, Agent `python -m unittest discover -s
  tests` 125 tests, UI `npm run test:unit` 936 tests across 120 files.
- 2026-08-07: the three-service smoke was NOT rerun for this baseline. Ports
  60033/60034 were serving a development instance on the host, and taking them
  over to test the release checkout would have disrupted it.
- 2026-08-18: the baseline advanced to the typed request-frontend integration,
  durable Agent final-handoff recovery, and corrected Analyzer UI selection
  semantics. The three component branch tips and the new `SyFI-VibeSim` URLs
  were verified directly against GitHub before recording their SHAs.
- 2026-08-18: a clean Ubuntu 24.04 container installed Node 22.14.0/npm 10.9.2,
  uv 0.12.5, Rust 1.97.1, protoc 3.21.12, and mold 2.30.0. The Analyzer cold
  release build passed in 4m03s, including the local `req-frontend` dependency,
  and Agent `uv sync` selected CPython 3.12.14 successfully.
- 2026-08-18: a clean Node 22 container completed UI `npm ci`, all 937 unit
  tests, and the production build. npm reported 3 moderate and 4 high audit
  findings; Vite retained the existing >500 kB chunk warning.
- 2026-08-18: the Agent clean-container suite ran 128 tests, with 125 passing.
  Three model-selection assertions require the host
  `~/.codex/models_cache.json` entries that advertise the `fast` service tier
  and `max` effort, so those tests are not currently hermetic. This does not
  block dependency setup, but the clean-container test gate is not fully green.
