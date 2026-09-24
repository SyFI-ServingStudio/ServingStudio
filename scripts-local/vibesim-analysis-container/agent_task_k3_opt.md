# Task: minimize $METRIC

You are optimizing a **real $FRAMEWORK checkout** at `$CHECKOUT` (the Python package is
`$CHECKOUT/python/sglang`, imported directly from that tree), running on one NVIDIA **B200**
(`CUDA_VISIBLE_DEVICES=0`).

- **Model:** $MODEL_NAME.
- **Workload:** $WORKLOAD.
- **Objective:** **minimize $METRIC**. Change the real source so the layer's decode step runs
  faster under CUDA-graph replay, and prove the change **does not alter the layer's numerical
  output or its post-step state**.

You are NOT told which operator, kernel, or code path to change, and NOT given a target
number. You must **derive** what to optimize from measurement and analysis — not from reading
code alone. Anything the layer executes is in scope (attention, MoE routing/dispatch/
activation-quant, projections, norms, the layer's dataflow, Triton and JIT-CUDA kernels under
`python/sglang/kernels`), as long as the checked numerics stay within tolerance.

## The rule of the loop: measure, simulate, THEN analyze
Whenever you obtain a new profile of the layer (the initial one, or one after an edit), route
it through VibeSim before drawing conclusions: build/fetch the prediction, then run the
analysis verbs. Never decide from raw numbers by eye.

## VibeSim roofline analysis API (read-only, no GPU) — `$ORACLE_URL/api/v1/<verb>`
Query args as URL params. Verbs:
- `workspace-info` — the action space and **how to read the analysis outputs** (what each
  rung and ratio means). **Call this first.**
- `simulate` — build/fetch the timing prediction of this layer/workload
  (`prediction=.:before` is the initial one).
- `analyze?level=operator|run_summary|iteration` — time breakdown of a prediction.
- `optimality?prediction=<id>&scope=iter` — per-node roofline ladder (R0/R5/R6/R7) plus a
  `necessary_share` per node. Use it to decide which node is the best candidate to change.
- `kernels?prediction=<id>` — per-leaf drill-down, incl. `has_cached_alternative`.

Once analysis points you at a node, map it to the **source** under `$CHECKOUT/python/sglang`
that produces it, then confirm with the profiler's per-kernel table (below) that the kernels
you expect are the ones actually launched.

## Measurement tool — `/tmp/kimi_single_layer_decode.py`
Builds exactly ONE decoder layer of this model in the production per-GPU shape with seeded
weights/state, drives real decode steps through the framework's own attention backends and
MoE kernels, captures the step in a CUDA graph and times replays. Always pass the fixed
workload flags `$DRIVER_ARGS`.

```bash
cd /tmp
ARGS="$DRIVER_ARGS"
# 1) latency + per-kernel table (what actually launches, how long each kernel takes):
python3 /tmp/kimi_single_layer_decode.py --point "$POINTS" $ARGS --split \
    --profile-kernels /workspace/opt_run/iter_NN/profile.json --json-out /workspace/opt_run/iter_NN/run.json

# 2) golden BEFORE you edit (reference output + post-step state your change must preserve):
python3 /tmp/kimi_single_layer_decode.py --point "$POINTS" $ARGS --capture /tmp/golden.pt

# 3) after editing, replay: speed AND correctness (per point: `JSON {...}` and `CHECK {...}`):
python3 /tmp/kimi_single_layer_decode.py --point "$POINTS" $ARGS --replay /tmp/golden.pt
```
- Every point prints a `JSON {"latency_us": ..., "latency_mode": "graph", ...}` line; the
  judged number is `latency_us` in graph mode. `--split` adds eager attn/moe/norms times.
- **Measure before and after with the IDENTICAL command** (same flags, same points, same
  `--iters/--warmup`), and treat a run that also carries `--split` or `--profile-kernels` as a
  *diagnostic*, not a timing reference: those extra phases perturb the graph-replay number by
  ~10%. The judge re-measures your tree and the tree you started from with the plain fixed
  flags, 5 repetitions each, and credits any improvement on the first point that clears 3 sigma
  (at least 0.5%) without regressing the others — so a before/after pair taken with different
  flags will mislead you about your own progress.
- **This is a continuing optimization.** The tree you start from may already contain accepted
  changes from earlier rounds (diff it against `original/` copies you keep, or read it): keep
  them working, build on them, and look for the *next* improvement. There is no target number —
  every verified gain counts, and the judge's correctness reference is always the ORIGINAL
  model's output, so accuracy cannot be traded away across rounds.
- `CHECK {... "pass": ...}`: output and post-step state must match the golden
  (`max_rel_err <= 0.02`, no NaN). **A change that speeds things up but alters the output or
  state FAILS**, on every point in `$POINTS` (the first point is the one scored for speed; the
  others must not regress).
- Do NOT edit the driver or pass different workload flags: the judge re-measures with its own
  copy and the fixed flags on a pristine container; only your edits under
  `$CHECKOUT/python/sglang` carry over.

## Reload after every edit (~1 min)
The package is imported from the source tree, so edits take effect in the next process, but
stale bytecode and a layer that no longer builds are the two silent failure modes. After each
edit run:
```bash
$REBUILD_CMD
```
It clears `__pycache__` and runs a one-point smoke that must print `LAYER_SMOKE_OK`. If it
does not, your change is NOT valid no matter how good an isolated number looks; fix it first.
The smoke covers ONE small point only: before you finish, re-run the replay at **every** point in
`$POINTS` — the judge measures all of them, and a kernel that runs at B=1 but rejects the B=128 or
long-context shape (dtype/layout guards inside the kernels are common) is scored as a hard FAIL.

## Organize your work by iteration
Create `/workspace/opt_run/iter_00/`, `iter_01/`, … For each iteration write, in that folder:
- `profile.json` — the per-kernel table + latencies you measured this iteration.
- `analysis.md` — the VibeSim prediction id you built + what the analysis showed (which node,
  its ladder/`necessary_share`, and why you did or did not target it).
- `hypothesis.md` — what you will change and why the output/state stay identical.
- `diff.patch` — `diff -ru` of the source you changed this iteration (against a copy you
  keep of the original files).
- `result.json` — post-edit `JSON` lines + the replay `CHECK` results.
Keep a top-level `/workspace/opt_run/log.md` summarizing each iteration in one line.

## Loop until done
1. Profile current code → build a VibeSim prediction of it → run `optimality`/`analyze`.
2. Pick the node most worth changing; map it to source; capture a golden.
3. Form a correctness-preserving hypothesis, edit the real source, reload (smoke).
4. Replay: confirm `latency_us` dropped on the first point AND `CHECK pass:true` on all points.
5. Re-profile, re-simulate, re-analyze: has the bottleneck moved, or is the remaining time on
   the top node no longer improvable by a code change? If there is still headroom and you have
   iterations left, go to 2. If not, you are done.

## Report
End with: the node(s) you changed, the source files you edited, and your per-iteration
before→after `latency_us` per point with the final CHECK results.
