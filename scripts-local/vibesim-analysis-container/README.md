# VibeSim analysis container (case: vllm-28103)

A self-contained **before-state diagnosis oracle**. It serves the VibeSim roofline
analysis API (the 5 agent-facing verbs) so an optimizer agent — running in a separate
Codex container — can locate and classify the bottleneck before editing real
sglang/vLLM code. It ships the docs + a Codex skill so you can teach the agent.

- **No GPU, no Rust toolchain, no PyO3.** Only the prebuilt `analyze` binary (257M,
  glibc-only) + `profile.db` + a baked before-state prediction.
- **Before-only / answer-key clean.** Source is `git archive defcc63`; the after arch
  recipe, after work-map, and `*_plan`/`*_progress` files are purged. `compare`
  (the evaluator verb) is not exposed over HTTP.
- **Role:** diagnosis only. It analyzes the *before* cost model; it does NOT evaluate
  the agent's code fix. The real workload run in the sglang/vLLM container is the
  ground truth.

## Build

```bash
./build_context.sh                       # stage ./context/ (source@defcc63 + binary + baked pred + docs)
docker build -t vibesim-analysis:vllm-28103 .
```

`build_context.sh` env knobs: `SRC_REPO`, `API_PKG`, `BASELINE` (defcc63),
`BAKED_SRC` (w_5609772a9151), `BAKED_RUN` (`api_v0_before` — the genuine
before-state prediction; do NOT use `eval_before_run`, that tree is on the after
commit), `OUT`. The image installs `python3-numpy` + an offline `uv` shim so the
`analyze` binary's R6/R7 labeler (`uv run python -m model.work.floors`) resolves to
`python3 -m ...` against system numpy — no network, no 9GB torch venv.

## Run

```bash
# HTTP API (default) on :8799
docker run --rm -p 8799:8799 vibesim-analysis:vllm-28103
curl -s localhost:8799/healthz
curl -s 'localhost:8799/api/v1/workspace-info'
curl -s 'localhost:8799/api/v1/optimality?prediction=.:before&scope=iter'

# one-shot CLI
docker run --rm vibesim-analysis:vllm-28103 cli optimality --prediction .:before --scope iter
# shell
docker run --rm -it vibesim-analysis:vllm-28103 bash
```

## Verbs (HTTP `/api/v1/<verb>`, args as query params; CLI `cli <verb> ...`)

| Verb | Purpose |
|---|---|
| `workspace-info` | action space: configs, GPU, profile.db coverage. Call first. |
| `simulate` | returns the baked before prediction `.:before` (idempotent, no GPU). |
| `analyze` | time breakdown: `level=operator\|run_summary\|iteration`. |
| `optimality` | the bottleneck finder: R0/R5/R6/R7 ladder per node, ranked by headroom. |
| `kernels` | per-leaf drill-down + `has_cached_alternative`. |

`compare` is evaluator-only and intentionally unrouted.

## The reading rule

- `R6 ≈ R5` → necessary work; gap is batching/launch. Kernel swaps won't help.
- `R6 ≈ 0` with big `R0` → unnecessary work → **the target** (28103: the qk_norm copies).
- `R0/R5` large, `R6` nonzero → inefficient necessary work → kernel-efficiency fix.

## Teaching the agent

`skill/SKILL.md` is a ready Codex skill. Point the agent at `$VIBESIM_API`
(e.g. `http://vibesim:8799`) and mount/copy the skill. The worked example and real
outputs are in `/opt/vibesim/vibesim-api/` inside the image.

## Architecture context

Three containers in the intended loop:
1. **sglang/vLLM repro container** (GPU) — real before-state code; agent edits + runs workload here.
2. **this VibeSim analysis container** (CPU) — diagnosis oracle; agent queries over HTTP.
3. **Codex agent container** — reads diagnosis → edits (1) → re-runs → loops → judge decides.
