# VibeSim Workspace

This repository pins the four repositories that form the VibeSim application
and introduction site, and keeps the reproducible build/deployment entry points
in one small place. It intentionally contains no product source code.

## Clone

```bash
git clone --recurse-submodules \
  https://github.com/SyFI-VibeSim/VibeSimWorkspace.git vibesim-workspace
cd vibesim-workspace
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

## Components

| Path | Repository | Tracking branch |
| --- | --- | --- |
| `VibeSim/` | VibeSim simulator and Analyzer | `master` |
| `VibeSimAgent/` | Agent/conversation backend | `agent-http-api` |
| `VibeSimUI/` | User-facing Analyzer UI | `main` |
| `vibesim-intro/` | Standalone VibeSim introduction site | `main` |
