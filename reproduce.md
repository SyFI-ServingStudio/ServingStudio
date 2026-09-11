# Reproduce VibeSim

This guide covers the pinned workspace, CPU application builds, and host deployment
with Docker Agent runners. Run commands as the non-root workspace owner on Ubuntu
24.04 x86-64. See [validation](#validation) for service wiring checks.

## Checkout

Configure credentials for the private GitHub repositories before cloning, using
an organization credential helper or, for HTTPS, `gh auth login` followed by
`gh auth setup-git`. Never put tokens in repository files or `.env`.

```bash
git clone --recurse-submodules \
  https://github.com/SyFI-VibeSim/VibeSimWorkspace.git vibesim-workspace
cd vibesim-workspace
```

| Directory | Role | Tracking branch |
| --- | --- | --- |
| `VibeSim/` | Simulator, Analyzer, profiling | `master` |
| `VibeSimAgent/` | Agent backend and Docker runners | `agent-http-api` |
| `VibeSimUI/` | Application UI | `main` |
| `vibesim-intro/` | Introduction website | `main` |

Repository URLs are in `.gitmodules`; Git submodule pointers are authoritative.
Verify them with `git submodule status --recursive`. To update an existing clone:

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

Inspect local work before updating. Do not use `--remote` for reproduction: it
advances to branch tips. Separate GLM development worktrees are outside this baseline.

## Host setup

The CPU build needs Git, Python 3, uv, Rust stable, just, Node.js 22/npm, and the
native tools below. `mold` is required by VibeSim's Cargo linker configuration;
system Python is needed by workspace setup and the Agent startup wrapper.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates cmake curl git mold ninja-build \
  pkg-config protobuf-compiler python3 tmux xz-utils

curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cargo install just --locked

# Install Node.js 22 through your approved package source or nvm.
node --version
npm --version
```

Downloads require access to GitHub, npm, PyPI/uv, Rust crates and Node distribution
servers. Runner builds additionally need container registries; conversations need
Codex endpoints, and optional automatic naming uses OpenRouter. Reserve at least
30 GB for initial runner images, Cargo artifacts, npm modules and uv caches;
models and additional workspaces need more.

From the workspace root:

```bash
just setup-env
source .env
just check-tools
```

`setup-env` creates `tmp/` and `uv-cache/`, writes their absolute paths as exported
`TMPDIR` and `UV_CACHE_DIR` in `.env`, and preserves unrelated settings. Re-run it
after moving the workspace. Root `just` automatically loads `.env`, overriding
inherited values. Source it from the workspace root in each new shell before
running commands directly; its paths remain valid after `cd`.

`.env`, `tmp/`, `uv-cache/`, and `old-wt/` stay untracked. If uv reports cache
permission errors, check that `UV_CACHE_DIR` points to the writable workspace
cache; do not use `sudo uv`. Cargo, npm and model caches retain their own settings.

## Build

Run long builds in a named tmux session, with a log:

```bash
tmux new-session -s vibesim-build
# Inside tmux, from the workspace root:
source .env
set -o pipefail
just build 2>&1 | tee "$TMPDIR/vibesim-build.log"
```

`just build` checks tools and recursive submodule revisions, builds the Analyzer
with locked Cargo dependencies, syncs the Agent's frozen uv environment, and builds
both the UI and introduction website. Individual targets are `build-analyzer`,
`sync-agent`, `build-ui`, and `build-intro`.

The Analyzer and web stack do not require CUDA or the full simulator Python
environment. For optional host simulation/profiling, use `just sync-simulator`;
it delegates to VibeSim's two-stage DeepGEMM setup. Bare `uv sync` is insufficient
for that cold setup. The introduction site uses the standard `npm run build`;
its separate `build:single` export currently fails on external asset references.

## Directory layout

```text
VibeSimWorkspace/
  VibeSim/                 Simulator, Analyzer and launcher source
  VibeSimAgent/
    vibesim_agent/         API, services, domain, providers, runtime and storage
    tools/                 Offline migration and deployment tools
    examples/providers.yaml  Shared configuration template
    providers.yaml         Private connections, ignored and loaded by default
  VibeSimUI/app/           Shared browser application
  vibesim-intro/           Introduction website
  agent-workspaces/        Default durable state, outside component repositories
  scripts/                Shared setup and service entry points
  .env                    Private workspace paths and service settings
  tmp/                    Disposable build/test/service logs
```

`VIBESIM_AGENT_WORKSPACES_ROOT` can select a different durable state directory,
including a verified migration target. Each workspace stores a descriptor and
SQLite database; conversations share its repository and logs but have separate
role/provider homes and runtime containers. `w_main` references `VibeSim/`
directly. Keep state and credentials out of commits and out of disposable `tmp/`.
The Agent repository has no separate browser frontend; both conversations and
analysis use VibeSimUI. See [Agent architecture](VibeSimAgent/doc/architecture.md)
for ownership and [named providers](VibeSimAgent/doc/providers.md) for multiple
accounts and endpoints.

## Run the application

Run the commands below from the workspace root after `just build`.
They start a local development UI, one Agent backend and one Analyzer in named
tmux sessions. The service commands require the new `vibesim_agent` package;
old Agent checkouts do not support these entry points.
On a shared server, assign three ports per user and check them before startup.
Use `60030 + 3 × UID` as the default base; spacing by three prevents adjacent
UIDs from sharing backend/Analyzer ports. For example, UID 1003 uses 63039–63041.
Override `VIBESIM_PORT_BASE` for a second workspace or if the default is occupied
or outside the valid port range; never stop another user's service to free a port.

Set `VIBESIM_PORT_BASE` in the ignored `.env` to override it, then run
`just check-ports`. This checks all three ports without stopping any service.
Startup also rejects occupied ports and disables Vite's automatic port switching.

### Build the Docker runner image

Agent conversations also need Docker daemon access and an authenticated host
Codex or Claude installation. Codex needs readable `$HOME/.codex/auth.json`;
valid CLI authentication does not require a separate `OPENAI_API_KEY`.
Keep credentials outside the repository. For the Codex runner:

```bash
source .env
just check-docker
just check-agent-auth
export VIBESIM_RUNNER_IMAGE="vibesim-agent-runner:${USER}"
just build-runner-image 2>&1 | tee "$TMPDIR/runner-image-build.log"
```

Run that build inside the build tmux session with `set -o pipefail`. It invokes
the selected Agent checkout's `scripts/build-runner-image.sh`,
prewarms Python dependencies, compiles the Cargo seed, and runs non-GPU acceptance.
Keep the same user-specific image tag when starting the backend: it records the
build account's UID/GID and home, so another user's image is unsuitable. GPU work requires
the NVIDIA driver and Container Toolkit; verify with `just check-gpu` using
physical-device access. CPU-only runners can use `export VIBESIM_RUNNER_GPUS=`.
See the [Agent setup](VibeSimAgent/README.md) for authentication, runner
configuration, Claude support and deployment details.

Optional configuration, exported before startup:

| Variable | Purpose |
| --- | --- |
| `VIBESIM_AGENT_API_TOKEN` | Bearer token for the Agent tools API. This is not an authentication layer for the entire browser application. |
| `OPENROUTER_API_KEY` | Optional automatic conversation titles. |
| `HF_HOME` | Existing readable model cache, mounted read-only at `/model` in runners. |
| `VIBESIM_AGENT_PROVIDERS_FILE` | Optional absolute path override for the required provider YAML; otherwise Agent loads `providers.yaml` from its source checkout. |
| `VIBESIM_AGENT_WORKSPACES_ROOT` | Absolute durable state directory; defaults to this workspace's `agent-workspaces/`. |
| `VIBESIM_AGENT_MAIN_DIR` | Absolute main simulator checkout; defaults to `VIBESIM_SIM_DIR`. |
| `VIBESIM_AGENT_DIR` | Agent source checkout; defaults to `VibeSimAgent/` under this workspace. |
| `VIBESIM_SIM_DIR` | Simulator/Analyzer source checkout; defaults to `VibeSim/`. |
| `VIBESIM_UI_DIR` | UI application directory; defaults to `VibeSimUI/app/`. |
| `VIBESIM_ANALYZER_BIN` | Absolute built Analyzer executable; defaults to `VIBESIM_SIM_DIR/target/release/analyze`. |
| `VIBESIM_UI_ALLOWED_HOSTS` | Explicit hostname allowlist when accessing Vite remotely. |

All directory overrides must be absolute. Individual `build-analyzer`, `sync-agent`
and `build-ui` recipes honor the selected component directories. `just build`
still checks the root repository's recorded submodule revisions first.

For multiple Codex/Claude accounts, put the local connection configuration at
`VibeSimAgent/providers.yaml`, or `providers.yaml` in the selected Agent checkout
when `VIBESIM_AGENT_DIR` is overridden. Agent discovers that file by default;
it is ignored by the Agent repository. `VIBESIM_AGENT_PROVIDERS_FILE` can select
a different absolute path. Start from the selected Agent checkout's
`examples/providers.yaml`. Tokens may use environment-variable references or an
inline `{value: "..."}` only in the ignored private file; keep that file mode 600
and never put tokens in the tracked example. Profiles retain separate
conversation/role session homes. See
[provider configuration](VibeSimAgent/doc/providers.md).

Legacy `CODEX_*`, `VIBESIM_WORKSPACES_ROOT`, and `VIBESIM_API_TOKEN` configuration
must be converted to the new Agent settings; the service rejects retired keys.
Use `uv run --frozen python -m vibesim_agent env-reference` from the selected
Agent directory for the complete supported list. `just check-agent-auth` is only
a convenience check for the default host Codex account, not a prerequisite for a
Claude-only YAML configuration.

### Initialize state

For a new installation, choose a new `VIBESIM_AGENT_WORKSPACES_ROOT` and run:

```bash
just agent-init
```

Initialization refuses any existing directory, including an empty one. Startup
never creates or migrates a state directory implicitly. For an existing legacy
installation, follow the Agent's [migration guide](VibeSimAgent/doc/migration-v1.md)
and select the validated target directory before using these ordinary service
commands. Agent and Analyzer read the same selected state's `registry.json`.

### Start the local UI and APIs

From the workspace root, after building components, preparing the runner image,
and initializing or migrating state:

```bash
just start
just services-status
just smoke-local
```

`just restart` gracefully stops the backend before restarting the three services
with the same configuration. Use `just services-status` and `just smoke-local`
to inspect them. Local service settings stay in the existing ignored `.env`;
provider connections stay in `VibeSimAgent/providers.yaml`.

`just start` runs `scripts/start-services.sh`: it checks ports and the runner
image, starts the backend, waits for its registry/API, then starts Analyzer and
one Vite development UI with `npm run dev`. The backend runs
`uv run --frozen python -m vibesim_agent serve` without rebuilding an image.
Services run in workspace-specific tmux sessions; logs are in
`tmp/{backend,analyzer,ui}.log`. It refuses to replace existing sessions.
Open the printed UI address, substituting your server hostname. `just stop`
stops only these three workspace sessions; inspect logs before restarting after
a failure.

APIs bind to the Docker bridge gateway so runners can reach them via
`host.docker.internal`; UI proxies and runner URLs use the selected ports.
`VIBESIM_API_BIND` overrides that address, which must remain runner-reachable.
`VIBESIM_UI_HOST` defaults to `0.0.0.0`; set a hostname allowlist with
`VIBESIM_UI_ALLOWED_HOSTS` for remote hostname access. Put shared configuration in
`.env` before starting services. A containerized backend needs additional host-path
translation or Docker-in-Docker; these scripts run on the host.

For production, serve the `npm run build` output (`dist/`) from a static web
server and reverse-proxy `/api/agent/v1` to Agent and `/api/analyzer/v1` to
Analyzer. Preserve streaming responses on Agent routes. Apply access control at
that deployment boundary. `just start` is a local development stack; neither
Vite's development server nor `vite preview` is the production static server.

## Verify

```bash
just smoke-local  # Or: just smoke http://127.0.0.1:<port>
```

This requires HTTP 200 through Vite for `/`, `/api/agent/v1/workspaces`,
`/api/analyzer/v1/runs`, `/api/analyzer/v1/predictions`,
`/api/analyzer/v1/kernel-profiles`, and `/api/analyzer/v1/kernel-measurements`.
It checks both proxy paths and fails on the first unsuccessful response.

To repeat runner acceptance independently:

```bash
(cd VibeSimAgent && ./scripts/test-runner-image.sh build)
# Optional, with GPU access and a warm kernel catalog:
(cd VibeSimAgent && ./scripts/test-runner-image.sh timing)
```

The final end-to-end gate is a real Agent conversation in workspace `w_main`.
Refresh while the turn runs; verify SSE reconnects, the answer persists, and any
managed simulation/timing/profile result appears in Page 0 and opens in Analyzer.

## Validation

Run `python3 scripts/test-services.py` after sourcing `.env` to check the current
service command wiring in a private temporary workspace. It checks explicit
initialization, selected paths and state, both proxy targets, versioned smoke
routes, and startup/stop failure handling with stubbed external commands. It
does not start real services or make provider calls.

The September 9, 2026 audit used a clean Ubuntu 24.04 container and exported pinned
sources without host environments, caches or credentials. It is not evidence of
an authenticated GitHub clone or complete Docker/GPU deployment.

| Check | Current audit result |
| --- | --- |
| Analyzer cold release build | Passed in 4m35s after installing missing `mold`. |
| Agent frozen uv sync | Passed; CPython 3.12.14, 41 packages. |
| UI production build | Passed. |
| Three-service smoke | All six routes returned HTTP 200 as UID 1000; no host ports published. |
| Agent tests | Candidate fixes: 176/176 passed without host model catalogs; explicit capability fixtures cover fast selection and missing-catalog fallback. |
| UI tests | Candidate cleanup: 985/985 passed, plus TypeScript checks; five decorative tests removed, behavioral assertions retained. |
| Introduction single-file export | Failed on external assets; root recipe now uses the standard website build. |
| Full aggregate build with revised recipes | Passed as non-root, including environment/submodule checks and all six smoke routes. |
| Extracted service scripts | Start/status/smoke, occupied-port and duplicate-start rejection, and scoped stop passed; Docker info/image/network checks were stubbed. |
| Runner image, GPU and real conversation | Not tested in this container; no Docker daemon/socket, GPU or credentials supplied. |

Audit tools: Node 22.14.0/npm 10.9.2, uv 0.12.12, Rust 1.98.1, just 1.58.0,
mold 2.30.0, protoc 3.21.12 and system Python 3.12.3. UI npm audit reported seven
moderate and five high findings; the intro reported none. No dependency upgrades
were made as part of setup verification.

Full earlier results, dates and limitations remain in
[reproduce-validation.md](reproduce-validation.md). Historical runner/GPU results
do not establish that the current revisions pass those gates.
