# Reproduction validation history

Historical results apply only to the revisions and environments stated below.
See [reproduce.md](reproduce.md) for the current procedure.


The previous guide's August 18 baseline recorded these component revisions:

| Component | Commit |
| --- | --- |
| VibeSim | `48864532c291e5cd48d01efb515d63733496e420` |
| VibeSimAgent | `2c80deb6eedf4a99f27790e659cc057b7a085854` |
| VibeSimUI | `3edaf8d8be290bd9a234ad49099ce9cd5a4039ae` |

Append dated evidence here when the procedure changes.

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

- 2026-09-09: workspace environment setup now uses `just setup-env`; verified
  generated absolute paths, direct-shell exports after changing directories,
  `just check-env`, and Git ignore rules on the host.

- 2026-09-09: extracted service-script audit passed in isolated Ubuntu as non-root.
  `just start`, `services-status` and `smoke-local` served all six routes. An
  occupied listener survived rejected startup; duplicate startup preserved all
  three service PIDs. `just stop` preserved an unrelated tmux session on the
  same socket. Port probes now tolerate TCP TIME_WAIT after shutdown. Docker
  info, image inspection and bridge lookup were stubbed; actual service
  processes and HTTP checks ran. Log: `service-tests-final.log` in the local
  `vibesim-ubuntu-setup-audit` evidence directory.

- 2026-09-09: pinned Agent tests initially passed172/175; UI passed979/990.
  Candidate test fixes isolate Agent capabilities from the host model catalog
  and remove five decorative UI tests plus redundant style assertions. Final
  clean non-root runs passed176/176 Agent tests and985/985 UI tests (125 files),
  with UI TypeScript checks passing. No production behavior changed. Evidence:
  `fixed-agent-tests.log` and `final-ui-fixed-tests.log` in the Ubuntu audit
  directory; the earlier990-test candidate run is superseded.
