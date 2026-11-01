# Kimi-K3 loop: was any accepted optimization caused by VibeSim?

Audit date 2026-10-15. Method: for every accepted round, read the agent transcript (`trial_<k>_agent.log`, tool-call
order) and the `trial_<k>_opt_run/` write-ups, then classify VibeSim's role:

- **CAUSAL**: VibeSim's output was read before the target or fix was chosen, and the choice followed from it.
- **CONFIRMATORY**: VibeSim agreed, but other evidence had already identified the target.
- **NOT USED**: the target came from elsewhere; any VibeSim citation is absent or was written after the fact.
- **MISLEADING**: VibeSim pointed somewhere wrong and cost the agent iterations.

Caveat: the transcripts have empty thinking blocks, so causality is inferred from tool-call order, visible text and
file timestamps.

## Answer

**None of the ~48 accepted levers was clearly caused by VibeSim.** Three Codex decode levers are partial cases, and
even there the profiler picked the target:

- **MLA trial 8 (T1, cute-dsl → trtllm-gen MLA decode):**
  - The target came from the profiler.
  - The *fix* came from VibeSim's `profiled_alternatives: [cutedsl, trtllm]` list, read before any source reading.
  - VibeSim's own ranking pointed at the MoE.
- **MLA r21 (front GEMM → CuTe TGV) and r23 (latent_up/shared_down → bf16 TGV):** VibeSim's ranked list named the
  next node to examine after the MoE and attention were exhausted. The fixes came from source reading and from
  reusing a pattern already in the tree.

## Per campaign

All directories are under `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/`.

| campaign / cases | accepted levers | causal | confirmatory | not used | misleading |
|---|---|---|---|---|---|
| Codex decode, MLA + KDA (campaigns 2–3) | 11 | 0 (3 partial: MLA T1 fix choice, r21, r23) | ~4 | ~3 | 1 (KDA r22) |
| Claude b512 + verify (`iter_opt_eval_k3_{kda,mla}_{b512,verify}_claude`) | 15 in 11 rounds | 0 | 5 (kda_b512 r1 ×2, r2; mla_b512 r1; kda_verify r3) | 9 | 1 (kda_verify r2) |
| Claude shapes (`…_{kda,mla}_shapes_claude`) | 9 | 0 | 3 (KDA r1, r2; MLA r4) | 6 (KDA r3–r5, MLA r1/PR C, r2, r5) | 0 |
| Claude prefill + mixed (`…_mla_prefill_claude`, `…_kda_prefill_claude`, `…_mla_mixed_claude`) | 7 | 0 | – | 7 (PR A's VibeSim hint came from a before-model that already assumed the fix) | 0 |
| Claude long-context prefill (`…_lcprefill_claude`) | 6 | 0 | – | 6 | 0 |

Per-round evidence for the three PR candidates is in `patches/sglang/pr_candidates/PR_{A,B,C}_*.md`.

## What actually found the targets

- **nsys graph timelines:** serial placement, idle HBM windows, single-CTA kernels and stream contention. This decided
  every shapes round and most b512/verify rounds. Five of the nine shapes levers are scheduling or overlap changes,
  which a sum-of-kernels model cannot represent.
- **Kernel names in the profiler table:** `HVPerCta256` (PR C), the trtllm-gen kernel still running where cute-dsl was
  expected (PR B T23), and the tile-8 MoE cubin (T30).
- **ncu / SASS:** T15, T19, T31.
- **Microbenchmarks against a hand-computed bandwidth floor:** T3, T35, PR C's split sweep, PR B's accuracy sweep.
- **The CPU profiler:** kda_verify r2, where the step was host-bound.
- **Warm-start history:** T30 hint, T36, T37, T43 (port of T9), T45 carveout.

## Why VibeSim didn't drive anything

1. **Wrong #1 node every round.** `mxfp4_fused_moe` ranked first in every round (R0/R5 17–44×). Its R5 (8.6–9.45 µs per
   case) is far below the real weight-read floor (~230 µs). Apart from the one autotune (T30), whose evidence came
   from the profiler, it never produced an accepted change. A sweep of all 1,736 MoE tactics found nothing more.
2. **Static before-state.** The baked prediction cannot re-simulate, so from round 2 on it describes the pristine
   tree. After T1, a stale `has_cached_alternative` cost MLA rounds 20, 21, 24 and 25 one iteration each.
3. **Internally inconsistent ladder:**
   - KDA R6/R7 are null for every node.
   - MLA attention has R6 > R0, which agents misread as "bandwidth-bound".
   - The fusion bucket reads 0, yet T20, T37 and T45 were fusions.
4. **Missing nodes and workloads:**
   - The attn_res kernels (KDA r3, MLA r2) and any prefetch don't exist in VibeSim's graph.
   - The verify cases are served a decode prediction, which has no verify node and is blind to host time.
5. **Broken verbs:**
   - `kernels` errors with "requires kernel_set".
   - The `prediction_id` that `simulate` returns doesn't resolve; every trial spent 3–8 calls on the handle.
6. **Citations written after the fact:**
   - The prompt requires each `analysis.md` to cite VibeSim, so most citations were written after the edit was
     measured.
   - In mla_b512 r3 (T20) and mla_verify r3, the cited ladder was fetched only after the code change.
   - T12's "R0/R5 1.3–1.7" attribution is post hoc; o_proj's real value is 1.77.

## Corrections to earlier write-ups

- **"MLA trial 8 — one VibeSim-guided iteration"** is half true: VibeSim supplied the alternative kernel, but not the
  target.
- **The KDA stacked seed's `num_warps` lever** isn't in the stacked patch.
- **The kda_verify recurrence kernel (T35)** landed in round 3, not round 2.
- **`vibesim_usage_audit.py` undercounts calls:** it only matches the literal `api/v1/<verb>` form and misses calls
  made through a `$B/<verb>` URL variable.

## What would make VibeSim causal next time

- Re-simulate from the agent's current tree each iteration (live prediction), instead of a baked before-state.
- Fix the MXFP4 MoE R5 floor, and make KDA R6/R7 non-null.
- Model the critical path and stream overlap, not just the kernel sum.
- Add verify and attn_res nodes.
- Make `kernels` and `prediction_id` work.
- Drop the prompt rule that forces a VibeSim citation, so that citations measure real use.
