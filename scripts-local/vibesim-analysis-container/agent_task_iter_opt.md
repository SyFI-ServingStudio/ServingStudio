# Task: optimize $METRIC for this model on the given workload

You are optimizing a **real $FRAMEWORK checkout** at `$CHECKOUT`, installed and running on an
NVIDIA **B200** (`CUDA_VISIBLE_DEVICES=0`).

- **Model:** $MODEL_NAME (local snapshot at `$MODEL_SNAP`).
- **Workload:** $WORKLOAD.
- **Objective:** **minimize $METRIC** on this workload. Change the real source so the model
  runs faster on this metric, and prove the change **does not alter the model's numerical
  output**.

You are NOT told which layer or which operator to change, and NOT given a target number. You
must **derive** what to optimize from measurement and analysis — do not guess from reading the
code alone.

## The rule of the loop: simulate, THEN analyze
Whenever you obtain a new profile of the model (the initial one, or one after an edit), you
must **first build a VibeSim prediction of that profile, then run analysis on that
prediction.** Never analyze raw numbers by eye — route every profile through VibeSim so the
analysis is computed consistently.

## VibeSim roofline analysis API (read-only, no GPU) — `http://172.17.0.1:8799/api/v1/<verb>`
Query args as URL params. Verbs:
- `workspace-info` — the action space and, importantly, **how to read the analysis outputs
  below** (what each rung and ratio means). **Call this first** and use it to interpret
  everything else.
- `simulate` — build/fetch the timing prediction for the current profile
  (`prediction=.:before` is the initial one). This is the "create a VibeSim implementation
  of the profile" step — do it before any analysis.
- `analyze?level=operator|run_summary|iteration` — time breakdown of a prediction.
- `optimality?prediction=<id>&scope=iter` — a per-node roofline ladder (R0/R5/R6/R7) plus a
  `necessary_share` for each node. Consult `workspace-info` for what these mean; use them to
  decide which node is the best candidate to change.
- `kernels?prediction=<id>` — per-leaf drill-down, incl. `has_cached_alternative`.

Once analysis points you at a node, map it back to the $FRAMEWORK **source** that produces it
(which module/op emits that work), then to the module you can point the profiler at.

## Measurement tool — `/tmp/extract_and_profile.py` (model-agnostic)
This captures the **real inputs** a target unit receives during a genuine forward pass
(exact shapes/strides/dtypes the model actually produces — nothing hand-built) and CUDA-times
just that unit. Point it at the module you identified:

```bash
cd /tmp
SNAP=$MODEL_SNAP
# 1) latency of your suspected bottleneck module, as-is:
python3 /tmp/extract_and_profile.py --model $SNAP --framework $FRAMEWORK \
    --target <MODULE.DOTTED.PATH> --tokens $TOKENS --iters 3000

# 2) capture a golden BEFORE you edit (records the reference output your fix must preserve):
python3 /tmp/extract_and_profile.py --model $SNAP --framework $FRAMEWORK \
    --target <MODULE.DOTTED.PATH> --tokens $TOKENS --iters 3000 --capture /tmp/golden.pt

# 3) after editing + rebuilding, replay to check speed AND correctness:
python3 /tmp/extract_and_profile.py --model $SNAP --framework $FRAMEWORK \
    --target <MODULE.DOTTED.PATH> --replay /tmp/golden.pt --iters 3000
```
- `--target` is a dotted `nn.Module` path from the model root, e.g.
  `model.layers.0.self_attn.<name>`. Introspect the model to find the exact path.
- Replay prints `CHECK {... "pass": ...}`: the module's output must still match the golden
  (max_abs_err < 0.05, no NaN). **A change that speeds things up but alters the output FAILS.**

## Recompile (incremental, ~2 min)
Edit under `$CHECKOUT`, then:
```bash
cd $CHECKOUT && $REBUILD_CMD
```
Rebuilds only what you touched, relinks the native extension, installs the rebuilt package into
the running environment, AND runs a **full-engine smoke test** (loads the model and generates a
few tokens). Two things are essential here:
- **The install step:** the profiler imports the *installed* package, not your source tree, so
  if you skip the rebuild you will keep measuring the ORIGINAL code and think nothing changed
  (or that a broken change works).
- **The smoke test:** a change can compile and pass the single-module replay yet still break the
  full model — e.g. a CUDA kernel whose launch config is fine for one isolated module call but
  crashes when the whole engine runs it across all layers/shapes. If the smoke test fails
  (no `ENGINE_SMOKE_OK`), your change is NOT valid no matter how good the per-module numbers
  look: a per-module speedup the full engine cannot run is worthless. Fix it before continuing.

Always rebuild after every edit, then re-run the profiler to see your change. If you edit a CUDA
kernel, also update any header it depends on, or the compile fails.

## Organize your work by iteration
Create `/workspace/opt_run/iter_00/`, `iter_01/`, … For each iteration write, in that folder:
- `profile.json` — the profile/layer-times you measured this iteration.
- `analysis.md` — the VibeSim prediction id you built + what the analysis showed (which node,
  its ladder/`necessary_share`, and why you did or did not target it).
- `hypothesis.md` — what you will change and why the output stays identical.
- `diff.patch` — `git diff` of the source you changed this iteration.
- `result.json` — post-edit layer latency + the replay CHECK result.
Keep a top-level `/workspace/opt_run/log.md` summarizing each iteration in one line.

## Loop until done
1. Profile current code → build a VibeSim prediction of it → run `optimality`/`analyze`.
2. Pick the node most worth changing; map it to source; point the profiler at that module;
   capture a golden.
3. Form a correctness-preserving hypothesis, edit the real source, recompile.
4. Replay the profiler: confirm latency dropped AND `CHECK pass:true`.
5. Re-profile, re-simulate, re-analyze: has the bottleneck moved, or is the remaining time on
   the top node no longer improvable by a code change? If there is still headroom and you have
   iterations left, go to 2. If not, you are done.

## Report
End with: the node(s) you changed, the source files you edited, and your per-iteration
before→after layer latencies with the final CHECK result.
