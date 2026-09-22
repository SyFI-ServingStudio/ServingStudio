You are optimizing an ML-serving **simulation** in VibeSim (a discrete-event
simulator; Rust core + Python launcher). A baseline ("before") timing-prediction
config is provided at `eval-configs/before.json` (with `eval-configs/cases.json`).
It models **Qwen2.5-7B dense inference on the SGLang path with speculative
decoding** on a single NVIDIA B200.

## Your goal
Find the operator whose **measured cost most exceeds its hardware limit** —
where the model spends far more simulated time than the hardware roofline for
its actual compute/bytes requires (typically launch/dispatch overhead from an
inefficient eager implementation) — and make the simulated model execute it
near its hardware limit, then demonstrate the improvement by re-running the
prediction. This mirrors the kind of kernel-efficiency optimization a real
framework PR would make.

## Hard constraints
- **Strictly cache-only / NO GPU.** Re-prediction must read the warm
  `profiling/profile.db`; never profile, never occupy a GPU.
- **Do not change what the model computes semantically.** The necessary-work
  floors (R6/R7) and the hardware limit (R5) of the operator must stay the
  same — you are removing implementation inefficiency, not changing the math.
- Work only inside this repository.

## Suggested method
1. Run the baseline prediction (GPU-free):
   `uv run python -m launcher timing-predict eval-configs/before.json`
   (writes to `logs/eval_before_run/`).
2. Build the analyzer if needed (`uv run cargo build --release -p analyzer --bin analyze`),
   then rank operators:
   - `./target/release/analyze run logs/eval_before_run optimality`
     → `reports/optimality_report.json` (per-operator ladders).
   - `./target/release/analyze gen-iter-breakdown logs/eval_before_run`
     → the labeled cost tree (`reports/iter_breakdown.ans`).
3. For each operator compare **R0 (measured) against R5 (hardware limit)**.
   Look for the operator with the largest R0/R5 ratio combined with a
   non-trivial absolute gap — hardware time far above what its compute and
   memory traffic justify. Inspect how that operator is implemented in the
   simulator model (arch/worklet/backend selection).
4. Implement a fix in the **simulator model** (Rust arch/worklet code, e.g. a
   more efficient backend/implementation for that operator) so its measured
   cost approaches the hardware limit, with semantics unchanged. Rebuild
   happens automatically via the launcher.
5. Re-run `uv run python -m launcher timing-predict eval-configs/before.json`
   and confirm the operator's R0 dropped substantially while R5/R6/R7 and the
   cost-tree structure are unchanged.

## Report back (required)
- **Diagnosis:** which operator, how far above its hardware limit it ran, and
  *why* (what the inefficient implementation was doing).
- **Fix:** exactly what you changed (file + the edit) and why it reaches near
  the hardware limit without changing semantics.
- **Evidence:** before/after R0 for that operator's scope with R5/R6 shown
  unchanged.
