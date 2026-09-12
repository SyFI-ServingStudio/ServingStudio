<p align="center">
  <img src="branding/servingstudio-logo.svg" alt="ServingStudio" width="280">
</p>

This repository pins the four repositories that form the ServingStudio application
and introduction site, and keeps the reproducible build/deployment entry points
in one small place. It intentionally contains no product source code.

## Clone

```bash
git clone --recurse-submodules \
  https://github.com/SyFI-ServingStudio/ServingStudio.git servingstudio
cd servingstudio
just setup-env
just check-tools
just build
```

For an existing clone:

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

Normal updates remain reproducible because every workspace commit records an
exact commit for each submodule. The branch declarations in `.gitmodules` are
used only when intentionally advancing with `git submodule update --remote`.

See [reproduce.md](reproduce.md) for prerequisites, credentials, first build,
service startup, and smoke checks.

The local service entry points use the new `vibesim_agent` package, explicit
`just agent-init` for new state, and versioned Agent/Analyzer APIs. Configure
`ServingStudioAgent/providers.yaml` (locally ignored) and durable state before `just start`;
Agent discovers the YAML in its selected checkout by default. Existing legacy state must
be migrated first. `just start` runs a Vite development UI. Production static
hosting and proxy requirements are documented in `reproduce.md`.

## Components

| Path | Repository | Tracking branch |
| --- | --- | --- |
| `ServingStudioSim/` | Simulator and ServingStudio Analyzer | `main` |
| `ServingStudioAgent/` | Agent/conversation backend | `main` |
| `ServingStudioUI/` | User-facing Analyzer UI | `main` |
| `ServingStudioIntro/` | ServingStudio introduction website | `main` |

## Agent Capabilities

Agent supports named Codex and Claude connections, each with its own account or
endpoint. A conversation can use a single assistant or an orchestrator and
implementer, with separate role sessions. Turns and streamed events are durable,
so reconnecting restores progress; cancellation and restart recovery retain
conversation history. Managed jobs link simulation and profiling results to
Analyzer resources.

The browser UI lives in `ServingStudioUI/app/`; Agent no longer bundles its own frontend.
`ServingStudioAgent/vibesim_agent/` separates API routes, services, domain objects,
providers, Docker runtime and storage. Private `providers.yaml` stays in the
Agent checkout and is ignored by Git. Workspace data lives in a separate state
directory; `w_main` references the editable `ServingStudioSim/` checkout.

See [Agent architecture](ServingStudioAgent/doc/architecture.md),
[provider configuration](ServingStudioAgent/doc/providers.md), and the
[directory layout](reproduce.md#directory-layout).
