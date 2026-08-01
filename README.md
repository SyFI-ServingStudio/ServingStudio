# VibeSim Workspace

This repository pins the three repositories that form the VibeSim application
and keeps the reproducible build/deployment entry points in one small place.
It intentionally contains no product source code.

## Clone

```bash
git clone https://github.com/serendipity-zk/VibeSimWorkspace.git vibesim-workspace
cd vibesim-workspace
git submodule update --init
just check-tools
just build
```

For an existing clone:

```bash
git pull --ff-only
git submodule sync
git submodule update --init
```

Normal updates remain reproducible because every workspace commit records an
exact commit for each submodule. The branch declarations in `.gitmodules` are
used only when intentionally advancing with `git submodule update --remote`.
Nested VibeSim alignment dependencies are intentionally excluded from the
default deployment; initialize them only when running alignment workflows:

```bash
just init-alignment
```

See [reproduce.md](reproduce.md) for prerequisites, credentials, first build,
service startup, and smoke checks.

## Components

| Path | Repository | Tracking branch |
| --- | --- | --- |
| `main/` | VibeSim simulator and Analyzer | `master` |
| `user-facing-ui/` | Agent/conversation backend | `agent-http-api` |
| `viz-ui/` | User-facing Analyzer UI | `main` |
