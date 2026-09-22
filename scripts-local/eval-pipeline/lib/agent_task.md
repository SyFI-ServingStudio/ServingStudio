You are optimizing an ML-serving **simulation** in VibeSim (a discrete-event
simulator; Rust core + Python launcher). A baseline ("before") timing-prediction
config is provided at `eval-configs/before.json` (with `eval-configs/cases.json`).
It models **Qwen3-0.6B dense inference on the vLLM path** on a single GPU.

## Your goal
Find and eliminate the single largest source of **unnecessary / redundant work**
in this model — work the simulated model performs that its *necessary-work*
roofline does not require — then demonstrate the improvement by re-running the
prediction. This mirrors the kind of optimization a real framework PR would make.

## Hard constraints
- **Strictly cache-only / NO GPU.** Re-prediction must read the warm
  `profiling/profile.db`; a cache miss must ERROR, never launch a profile. Never
  pass `--profile`. Never run anything that would occupy a GPU.
- **Do not change what the model computes semantically.** The necessary-work
  floor (the R6/R7 rungs) must stay the same — you are removing redundant
  overhead, not reducing the actual math.
- Work only inside this repository.

## Suggested method
1. Run the baseline prediction (GPU-free):
   `uv run python -m launcher timing-predict eval-configs/before.json`
   (writes to `logs/eval_before_run/`).
2. Rank operators with the optimality ladder. Useful commands:
   - `uv run cargo build --release -p analyzer --bin analyze` (if needed), then
   - `./target/release/analyze run logs/eval_before_run optimality`
     → `reports/optimality_report.json` (per-operator ladders, worst kernels).
   - `./target/release/analyze gen-iter-breakdown logs/eval_before_run`
     → renders the labeled cost tree (`reports/iter_breakdown.ans`).
   - `./target/release/analyze optimality-scoped logs/eval_before_run --label "<node label>"`
     → per-subtree R0 (measured) / R5 (hardware limit) / R6 (segmented
       necessary) / R7 (fused necessary), and the subtree's leaf list.
3. Find the operator whose **measured cost (R0/R5) is large but whose
   necessary-work (R6/R7) is ~zero** — i.e. it does real hardware work that the
   model's necessary-work definition does not account for. That gap is redundant
   overhead. Inspect the cost-tree leaves under that operator to see exactly what
   redundant sub-operations it performs.
4. Implement a fix in the **simulator model** (Rust arch/worklet code) so the
   redundant sub-operations are no longer emitted, while the necessary
   computation is unchanged. Rebuild happens automatically via the launcher.
5. Re-run `uv run python -m launcher timing-predict eval-configs/before.json` and
   confirm the redundant leaves are gone and the operator's R0/R5 dropped, with
   R6/R7 unchanged.

## Report back (required)
- **Diagnosis:** which operator, which redundant sub-operations, and *why* they
  are redundant (what necessary work they were not contributing to).
- **Fix:** exactly what you changed (file + the edit) and why it removes the
  redundant work without changing the necessary computation.
- **Evidence:** the before/after R0/R5 for that operator's scope and the leaf
  lists, showing the redundant leaves are gone.
