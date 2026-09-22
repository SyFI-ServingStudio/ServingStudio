---
name: vibesim-analysis
description: Diagnose the single largest fixable bottleneck in an ML-serving workload using the VibeSim roofline analysis API (before-state oracle). Use before proposing a performance fix.
---

# vibesim-analysis

You have a **VibeSim analysis oracle** for the before-state of the workload you are
optimizing. It tells you *where* the bottleneck is and *what kind* it is, so you do
not guess. It analyzes a cost model of the before state — it does **not** see or run
your code changes. You validate your actual fix by re-running the real workload; you
use this API only to **locate and classify** the bottleneck first.

The API is cache-only, deterministic, and uses **zero GPU**.

## How to reach it

Two interchangeable transports (same 5 verbs, same JSON):

- **HTTP** (from another container): `GET/POST $VIBESIM_API/api/v1/<verb>`, where
  `$VIBESIM_API` is the analysis container's URL (e.g. `http://vibesim:8799`).
- **CLI** (inside the analysis container): `python3 /opt/vibesim/app/vibesim_api.py <verb> ...`

Examples below use CLI form; the HTTP path takes the same args as query params
(`?prediction=.:before&level=operator`). The prediction handle is `.:before`.

## The verbs

1. `workspace-info` — what the workspace offers (configs, GPU, profile.db coverage).
   **Call first** to learn the action space.
2. `simulate --config eval-configs/before.json` — returns the baked before-state
   prediction handle `.:before` (idempotent; no GPU).
3. `analyze --prediction .:before --level operator|run_summary|iteration` — where
   time goes. **Time share alone is misleading** — it does not tell you what is fixable.
4. `optimality --prediction .:before --scope iter` — **the bottleneck finder.** One
   row per cost-tree node, full R0/R5/R6/R7 ladder, sorted by headroom `R0−R5`, plus
   `necessary_share_r6_over_r0`.
5. `kernels --prediction .:before --kernel-set nameA,nameB` — per-leaf drill-down:
   kind, backend, shape, ladder, and whether the profile.db holds an alternative
   implementation (`has_cached_alternative`).

## The reading rule (this is the whole game)

For each node in the `optimality` output:

- **big `R0−R5`, but `R6 ≈ R5` (necessary_share ≈ 1)** → work is necessary; the gap
  is batching / launch / scheduling overhead. **A kernel change will not move it.**
- **big `R0`, `R6 ≈ 0` (necessary_share ≈ 0)** → the operator does work the model
  semantics do not require at all → **unnecessary-work bottleneck. This is the target.**
- **`R0/R5` large, `R6` nonzero** → inefficient implementation of necessary work →
  kernel-efficiency bottleneck (swap/fuse the backend, if `has_cached_alternative`).

Drill by re-scoping to a node's `path` (`optimality --scope iter/1/0/1`). Never
construct paths by hand — read them from the report rows.

## Method

1. `workspace-info`, then `simulate` (→ `.:before`).
2. `analyze --level operator` for orientation; `optimality --scope iter` to rank
   *fixable* headroom.
3. Find the node with large `R0` and `necessary_share ≈ 0`. Drill into its subtree
   and `kernels` its leaves to see exactly which redundant sub-operations it emits.
4. That is your bottleneck. Apply the fix to the real serving code (in the
   sglang/vllm container), then re-run the workload to confirm it is faster.

## Do not

- Do not call `compare` — it is evaluator-only and is not exposed here.
- Do not treat time share (`analyze`) as the answer; always confirm with the ladder.
- Do not try to make VibeSim re-simulate your code fix — it models the before state
  only. Ground-truth validation is the real workload run.
