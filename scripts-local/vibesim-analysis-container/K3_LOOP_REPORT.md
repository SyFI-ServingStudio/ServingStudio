# Kimi-K3 decoder-layer optimization loop — results (C4 report, DRAFT 2026-09-23)

Open-ended case of the 3-container optimization loop (eval-harness sglang container with
`kimi_single_layer_decode.py` as extractor · VibeSim K3 oracle · Codex `gpt-5.6-luna@max` agent).
Optimization unit = one `KimiK3DecoderLayer` (KDA or MLA variant) in the sglang cookbook B200 shape
(attention TP8 rank: 12 heads; EP8 rank: 112 MXFP4 experts; fp8 KV; bf16 KDA state). Metric =
CUDA-graph replay time of the decode step; correctness = output + post-step state bit-exact-ish
(`max_rel_err <= 0.02`) vs a seeded golden from the pristine tree. Gate: primary point improves by
`max(5%, 3σ)` ∧ no secondary regression ∧ all CHECKs pass. Judge: `judge_k3.py` (5 reps per point).

Campaign complete 2026-09-23 22:16: 8 judged trials (KDA 4–7, MLA 4–7), KDA 7 re-judged (v2),
B8 rows filled, oracles re-baked on the post-B8 predictions.

> **Sign convention in this report:** percentages written next to a `before → after` arrow follow the arrow; standalone percentages in the campaign-3/4 round tables and in `rounds.jsonl` are the judge's `improvement` = (before − after)/before, i.e. **positive = faster**. The clean write-up `K3_PIPELINE_AND_RESULTS.md` uses latency change (negative = faster) throughout.

## Pipeline controls (trials 1–3 of each case, Milestone 1 C1)

| case | control | expected | result |
|---|---|---|---|
| KDA / MLA | null (pristine tree, agent budget 1 s) | ≈0% improvement, CHECK pass | −0.01% / −0.12%, pass |
| KDA / MLA | planted slowdown (`.contiguous()` in MoE) | FAIL latency | −17.7% / −15.5% (slower), pass, FAIL ✓ |
| KDA / MLA | planted numerics (`latent *= 0.9`) | FAIL correctness | rel_err 0.067, FAIL ✓ |

## Agent trials (luna@max; 4 = 40–45 min budget via slurm, 6–7 = 30 min direct on GPU 3)

Baselines (judge medians, σ ≤ 0.8 µs): KDA 558.6 / 312.8 / 181.7 µs at B=128/32/1 (L=8k);
MLA 681.3 (128×8k) / 345.5 (1×1M) / 358.8 (16×64k) µs.

| trial | verdict | primary Δ | secondaries Δ | CHECK | what the agent changed (files) | VibeSim node that led there |
|---|---|---|---|---|---|---|
| KDA 4 | FAIL (gate) | −1.5% (550.3) | −1.3%, 0.0% | pass, rel 0.0 | shared experts on the alt stream during routed MoE (`kimi_k3.py`, 25 lines) | `moe.mxfp4_fused_moe` rank 1 (necessary_share 0.43) |
| KDA 5 | FAIL (crash) | — | — | — | forced fused KDA decode kernel for bf16 state by loosening `covered()` (`kda_fused_decode.py`, `kimi_k3.py`, 57 lines); kernel's dtype guard rejects bf16 at B=128 | `attention.kda_recurrent_decode` (unfused chain) |
| KDA 6 | FAIL (gate) | **−2.9%** (542.1) | −2.6%, −0.6% | pass, rel 0.0 | shared-expert `down` on the side stream vs routed MXFP4 BMMs + `lower_bound` gate takes fast path for B≥8 (`kimi_k3.py`, `kda_backend.py`, 47 lines); rejected bf16-SiTU (no B200 tactic) and PDL-off (regressed) | `moe.mxfp4_fused_moe` rank 1 |
| KDA 7 | v1 FAIL (judge artifact); **v2 FAIL (gate)** | −1.1% (552.3) | −2.6%, −3.4% | pass, rel 0.017 (not bit-exact) | **templated the JIT CUDA kernel `kda_fused_decode.cuh` for a bf16 state** + `.py` gate (39 `.py` lines + `.cuh`); v1 judge dropped the `.cuh` (`.py`-only diff) → pristine kernel rejected bf16; v2 with the `.cuh`: the port works, fused decode on bf16 state is worth 1–3% here. Also rejected FlashInfer `recurrent_kda` (556→591 µs) and PDL-off | `attention.kda_recurrent_decode`, then `moe.merged_front` (R0 120 → R6 30 µs) |
| MLA 4 | FAIL (gate) | −2.1% (667.1) | **−11.3% @1×1M** (306.5), −0.3% | pass, rel 0.0 | non-DCP decode `cute-dsl` → `trtllm-gen` (`cutedsl_mla_backend.py`) + shared/routed overlap for ≥64 tokens (`kimi_k3.py`), 35 lines; 3 null hypotheses killed (fused-token cap, TRT tactic bucket, CUTLASS backend) | `moe.mxfp4_fused_moe` rank 1; attention leaf for the 1M point |
| MLA 5 | FAIL (gate) | −2.1% (667.3) | **−14.5% @1×1M** (295.3), −0.9% | pass, rel 0.0 | `backend = cute-dsl if dcp_enabled else trtllm-gen`; shared GEMM on side stream joined by event before the final add — **gated on `tp_size == 1`** (harness-only, would not fire in production TP8); `cutedsl_bf16_gemm.py` tweak; 60 lines. Thin artifacts (no analysis.md/diff.patch) | same |
| MLA 6 | FAIL (null) | +0.1% (680.4) | 0.0%, 0.0% | pass | TRT-LLM MoE tactic ceiling 128→256 in `moe_runner/flashinfer_trtllm.py` (39 lines) — no kernel change, no effect; agent analyzed the KDA prediction (oracle wiring bug, finding 6) | `mxfp4_fused_moe` (KDA prediction) |
| MLA 7 | FAIL (gate) | −1.2% (673.2) | +0.6%, +0.3% (within noise) | pass, rel 0.0 | shared/routed alt-stream overlap (`kimi_k3.py`) + host-side `is_var_seq` uniform-length detection in `trtllm_mla_backend.py` aimed at a CuTeDSL persistent kernel (no effect); 69 lines. First MLA trial on the correct oracle (8802): `unified.mla.moe.mxfp4_fused_moe` R0 527 µs, necessary_share 0.43 → same MoE-first conclusion. Did not find the cute-dsl→trtllm-gen switch | `moe.mxfp4_fused_moe` rank 1 |

PASS: 0 / 8 judged. Bit-exact improvements found in 5 of 8; the two crashes were both attempts to
run the fused KDA kernel on the bf16 state (trial 7 did the real kernel work, judged v2).

## Findings

1. **At the production point (B=128, L=8k) both layers are MoE-bound** (KDA: MoE leaf 54% of the
   graph, R6 necessary_share 0.43; MLA similar) and the routed MXFP4 expert GEMM is a closed
   TRT-LLM cubin (`trtllm_fp4_block_scale_moe`). Every agent correctly identified it first via
   `optimality`, then found only overlap wins around it (1.5–3%). The 5% gate is above what
   Python-level edits can recover there.
2. **The MLA attention backend choice is worth −11…−14% at long context.** sglang's default
   `cutedsl_mla` path for the non-DCP decode is slower than `trtllm-gen` at 1×1M with identical
   numerics (rel_err 0.0). This is a real, portable finding (surfaces as a `cutedsl_mla_backend.py`
   one-liner), but it is a secondary point under our gate.
3. **bf16 KDA state (cookbook `--mamba-ssm-dtype bfloat16`) disables sglang's fused KDA decode
   kernel** (fp32-only); agents keep rediscovering this (trials 5, 6, 7). Trial 7's kernel port is the
   right kind of fix — verdict v2 pending.
4. Harness overfits observed: MLA 5 gated its overlap on `tp_size == 1`; agents' before/after
   comparisons drifted when they mixed `--split/--profile-kernels` runs with plain ones (prompt now
   mandates identical flags and a full-point replay before finishing).
5. Judge corrections made during the campaign (all evidence preserved as `_v1`): driver snapshotted
   per case key (an edit between baseline and replay had produced a false rel_err 1.79); JIT CUDA
   sources (`.cu/.cuh/.h`) now part of the judged diff (trial 7).
6. **Oracle wiring bug in the MLA case (found during MLA 6):** `issue_k3_mla.json` still pointed at
   the KDA oracle (8801). That image carries both baked predictions, and the agents in MLA 4 and 5
   discovered and used `prediction=k3_mla:before` themselves (trial 4: 1938 `unified.mla` hits vs 37
   `unified.kda`); MLA 6 analyzed the KDA prediction only. Config fixed to 8802 (whose default
   `.:before` is `k3_mla`); MLA trial 7 re-run with the correct oracle is queued after the fill.

## VibeSim side (Milestone 2)

- Branch `kimi-k3` (VibeSim `main/`): kinds `kda_recurrent_decode`, `kda_fused_decode`,
  `mla_decode_attention`, `mxfp4_fused_moe` + sglang backends on existing kinds; `sglang_k3_env`;
  arch `kimi_k3_sglang` (KDA/MLA/MoE worklets, state-dtype chain, TP8/EP8/PP2 + rank-1 presets);
  model.work label + location maps; scoped R6/R7; alignment pack (`presets/alignment/kimi_k3_single_layer/`,
  `doc/alignment/kimi_k3_single_layer.md`).
- Rank-1 predictions vs measured graph time: KDA 161/303/542 vs 169/360/623 µs (B=1/32/128, older driver
  baseline); MLA 585/331/289 vs 688/352/342 µs (128×8k, 16×64k, 1×1M). Alignment (duration-weighted):
  KDA −1.9% (97% coverage), MLA −13.7% (98%).
- B8 (07d5ee16): `qkvbfg_a_proj` split into wide QKVG GEMM + side-stream `[f_a|β]` GEMV under
  `CostNode::Max` (was +198%); MLA cache-append runner storage fixed; 136 new B200 rows filled
  (21:41). Post-B8 rank-1 predictions vs measured graph time: KDA 141/277/495 vs 179.6/310.7/556.5 µs
  (B=1/32/128 → −21/−11/−11%); MLA 569/274/317 vs 680.4/345.5/357.9 µs (128×8k, 1×1M, 16×64k →
  −16/−21/−11%). The qkvbfg leaf is now correct (~21 µs), so the remaining under-prediction is the
  `mxfp4_fused_moe` row (−10.8%: the runner times one call with a uniform expert distribution, the
  driver's seeded routing is skewed) plus ~17 µs of unmapped `attn_res_fused_tma` + CUDA-graph glue.
  Analyzer caveat: `Max` hides non-critical leaves' time in the ladder.
- B9 (b088a6eb): root cause of the MoE row: the runner profiled 896 global experts, under which only 258 of
  the 2048 assignments hit the 112 local experts (≈1/8 of the layer's expert work). Added a
  `routing_experts` dimension (112 rank-1 / 896 EP8); 68 new rows filled per preset. Post-B9 alignment
  (CPU rerun): KDA 553.1 measured vs 679.4 simulated (**+22.8%**, mxfp4 leaf +37%), MLA 677.5 vs 752.9
  (**+11.1%**, mxfp4 +25%) — the 112-expert runner call now *over*-prices the layer's launch (layer's
  MoE kernels sum to ≈418 µs at B=128: two `bmm_*MxE2m1*` 265+135 µs + finalize 10 + routing 8). Also
  confirmed: `mla_cache_append`/`output_gate`/`kv_a_layernorm` read −100% under the `Max` attribution.
  → B10 (Codex, GPU 3 pinned by UUID): A/B the runner against the layer's real call (routing histogram,
  tactic/autotune, input format, PDL/graph) and fix the Max attribution. Oracles stay on the post-B8 image
  until B10 lands.
- B10 (ef202ca2, GPU 3 microbench; final rows B=32 178.1 µs vs 176.2 reference (+1.1%), B=128 382.6 vs ≈418
  (−8.3%); tests 1091 Rust / 3478 Py green): the gap was the **routing distribution**. The driver's real top-k
  histogram is collapsed (≈16–20 experts get 119–128 tokens each, most experts 0 — decode inputs are
  `randn×0.02` plus a shared-mean planted state, so all tokens look alike after the pre-MoE norm), and
  `--experts 112` makes all 2048 top-16 assignments local (a real EP8 rank gets ≈256). Runner with the routed
  API + actual histogram: 390.5 µs @B=128, 180.9 @B=32 (layer ≈418/384); balanced routing ≈516. With the
  histogram-aware row + shared `Max` attribution, alignment is KDA **−3.3%** (553.1 vs 535.0 µs; MoE leaf
  within 8%) and MLA **−10.2%** (677.5 vs 608.5; MoE −9.2%, `mla_decode_attention` −9.2%, small attention
  leaves still noisy). **Harness caveat for every trial above:** the judged MoE workload is collapsed-routing
  and 8× a production rank's assignment count; fixing the driver (unit-scale inputs, EP8 shard routing over
  896 experts) requires a re-baseline and re-run — pending the user's decision.
- Branch profile DB `kimi_single_layer/k3_branch_profile.db`; merging into the shared
  `profiling/profile.db` awaits the user's explicit OK.

## Realistic-routing re-baseline (2026-09-24, GPU 7)

Driver flags `--local-topk 2 --hidden-scale 1.0` (commit e4a3c25; now in both case configs, 9cc2670):
top-2 over the 112 local experts emulates the rank's share of K3's global top-16 (256 local rows at
B=128 instead of 2048), unit-scale inputs stop the routing collapse.

| layer | point | old flags (µs) | realistic (µs) | MoE kernels (realistic) |
|---|---|---|---|---|
| KDA | 128×8k | 556.5 | **252.4** | 85 of 306 kernel-µs (28%; was ≈54%) |
| KDA | 32×8k / 1×8k | 310.7 / 179.6 | 174.6 / 139.7 | 44 / 24 µs |
| MLA | 128×8k | 680.4 | **352.6** | 96 of 405 kernel-µs |
| MLA | 1×1M / 16×64k | 345.5 / 357.9 | 306.7 / 264.7 | 25 / 46 µs |

**Correction (02:30):** the table above was still routing-collapsed (8 of 112 experts active). The actual
root cause was the parameter init: every parameter, *including the RMSNorm gains*, was `N(0, 0.02)`, so
normalized activations were ×0.02, router logits ~0.03 wide, sigmoid ≈ 0.5 for every token, and the
`e_score_correction_bias` alone ranked the experts (`--hidden-scale` was a red herring: 0.02 → 75 active,
1.0 → 8, 20 → 91). With norm gains `1 + 0.02·randn` (commit 80786a0) the top-2 routing is Poisson-like:
**85/112 experts active at B=128 (max 11), 44 at B=32.**

| layer | point | old flags (µs) | realistic v2 (µs) | MoE kernels (v2) |
|---|---|---|---|---|
| KDA | 128×8k | 556.5 | **402.9** | 236 of 457 kernel-µs (52%) |
| KDA | 32×8k / 1×8k | 310.7 / 179.6 | 262.7 / 139.7 | 135 / 25 µs |
| MLA | 128×8k | 680.4 | **493.0** | 237 of 545 kernel-µs (43%) |
| MLA | 1×1M / 16×64k | 345.5 / 357.9 | 305.7 / 303.6 | 25 / 85 µs |

So at B=128 the MoE is weight-bandwidth-bound over ~85 active experts (≈1.4 GB of MXFP4 weights per step)
and still the largest single leaf, but no longer 8× inflated; at 1×1M attention dominates. The trial campaign
is re-run on these baselines with the MLA primary moved to 1×1M.

## Campaign 2 — realistic workload (v2 driver, oracles on B11 predictions, GPU 7, 45-min agents)

Baselines (judge, 5 reps): MLA 304.6 (1×1M, primary) / 493.0 (128×8k) / 302.5 (16×64k) µs;
KDA 402.9 / 262.7 / 139.7 µs (B=128/32/1).

| trial | verdict | primary Δ | secondaries Δ | CHECK | change | VibeSim path |
|---|---|---|---|---|---|---|
| **MLA 8** | **PASS** | **−12.1%** (304.6 → 267.7 @1×1M) | +1.6%, +0.7% | pass (rel 0.0 / 0.014 / 0.005), state_ok | 15-line diff, `cutedsl_mla_backend.py`: non-DCP decode routed to TRT-LLM MLA | one iteration: `kernels` flagged a cached alternative on the MLA decode leaf → agent applied it, re-verified, stopped |
| KDA 8 | FAIL (gate) | −3.1% (403.0 → 390.7 @128×8k) | −4.7% @32, −5.9% @1 | pass, rel 0.0 at all points | 45-line diff, `kimi_k3.py`: shared `down` GEMM (7168×6144, m≤128) routed to the CuteDSL bf16 GEMM + shared branch on the alt stream; rejected fused MXFP4 router (+6 µs) and KDA TMA threshold (neutral) | `mxfp4_fused_moe` rank 1 (no alternative) → `moe.shared_down` (50 µs vs R6 16.6) → `kda_recurrent_decode` (42.5 vs R5 8.2) |
| KDA 9 | FAIL (gate) | −0.3% (403.0 → 401.9) | −2.3% @32, 0.0% @1 | pass (rel 0.013 / 0.007 / 0.0) | 589-line CUDA port of the fused KDA decode JIT kernel to bf16 state (`kda_fused_decode.cuh` + `.py`) — judged properly this time; two earlier iterations failed CHECK and were rolled back | `mxfp4_fused_moe` → `kda_recurrent_decode` (42.5 µs vs R5 8.2) |
| KDA 10 | FAIL (gate, by 0.4 pt) | **−4.6%** (403.0 → 384.6 @128×8k) | −5.4% @32, 0.0% @1 | pass, rel 0.0 at all points | 50-line diff: packed KDA decode fast path kept for K3's lower-bounded gate (`kda_backend.py`) + shared/routed alt-stream overlap (`kimi_k3.py`); rejected bf16 fused kernel (483 µs), bf16-activation dispatch, route-quant cap, PDL-off | `mxfp4_fused_moe` → KDA packed-dispatch guard |
| KDA 11 | FAIL (gate) | −4.4% (403.0 → 385.5) | −5.05% @32, −5.9% @1 | pass, rel 0.0 at all points | 62-line diff over 16 iterations: packed decode fast path (`kda_backend.py`) + Triton `fused_recurrent_kda_packed_decode` launch tuning (`fla/fused_recurrent.py`) + shared/routed overlap (`kimi_k3.py`) | `mxfp4_fused_moe` → KDA packed-dispatch guard → `kda_recurrent_decode` |

**Campaign 2 result: MLA PASS (−12.1% @1×1M); KDA best −4.6% @B=128 (4 trials, all bit-exact, gate 5% not
met).** KDA levers found are independent and bit-exact: packed decode fast path (≈−2%), shared-`down` GEMM →
CuteDSL bf16 (≈−3%), Triton recurrent-kernel tuning + overlap (≈−1.5…−2.5%); no single 45–60-min run combined
the CuteDSL GEMM with the others. **Operator-stacked verification** (trial 11's tree + trial 8's CuteDSL GEMM
hunk, judged identically, `iter_opt_eval_k3_kda/stacked_verdict.json`, not an agent result): 403.0 → 383.7 µs
(**−4.8%**), −5.4% @32, −6.0% @1, bit-exact — the levers do not add linearly (the shared GEMM already runs on
the side stream, so making it faster barely moves the critical path). With Python-level edits the KDA layer's
ceiling at B=128 is ≈4.8% under this workload; the remaining 52% is the weight-bandwidth-bound MXFP4 MoE cubin.

## Campaign 3 — continuous loop (user decision 2026-09-24: no fixed gate)

Each round seeds the agent with the current `best_tree`, the judge measures the round's baseline from
that tree and accepts any primary gain ≥ max(3σ, 0.5%) with no secondary regression; correctness is
always checked against the **pristine** goldens (accuracy cannot drift). Seeds: MLA = trial 8's judged
tree (267.7 µs @1×1M), KDA = the verified stacked tree (383.7 µs @B=128). GPU 7 is shared with a heavy
co-tenant during this campaign (±3% run-to-run noise; the 3σ requirement absorbs part of it).

| round | verdict | primary | secondaries | numerics | new change this round | note |
|---|---|---|---|---|---|---|
| MLA 20 | PASS | +0.75% (267.7 → 265.7 @1×1M) | 0.0%, +0.65% | rel ≤ 0.014 | `route_quant_fused` JIT specialized for the 112-expert/top-2 shape (+ `route_radix`, `topk.py`) | **harness-specific**: production routes top-16 over 896 experts and already uses the fused kernel |
| MLA 21 | PASS | +1.5% (267.8 → 263.7 @1×1M) | +3.2% @128×8k, +0.7% @16×64k | rel ≤ 0.014 | fp32-output front GEMM (weights 15984×7168 / 6016×7168, m ≤ 16) routed to the CuTe DSL `cutedsl_bf16_gemm_out` instead of cuBLAS (`kimi_k3.py`, ~27 lines) | production-relevant kernel selection for small decode batches; GPU 7 exclusive during this round (your areal run had ended) |
| MLA 27 | FAIL (null) | 0.0% (253.5 → 253.5) | 0.0%, +0.7% | pass | a fused finalize+shared-expert JIT kernel (`moe_finalize_fuse_shared`, regressed to 264 µs, reverted) and a `flashinfer_trtllm.py` tweak (no effect) | second consecutive null → MLA plateau; **loop stopped here** |
| MLA 26 | FAIL (null) | 0.0% (253.4 → 253.4) | +0.4% @128×8k, 0.0% | pass | `cutedsl_bf16_gemm.py` tweak with no effect on the primary point | first null MLA round since 22 |
| KDA 25 | FAIL (null) | +0.3% (379.5 → 378.4; needed 0.56%) | +0.8% @32, +1.6% @1 | pass (rel ≤ 0.013) | small `kimi_k3.py` edit; secondaries improved but the primary point didn't clear the floor | KDA plateau confirmed (rounds 23–25) |
| MLA 25 | PASS | +0.8% (255.4 → 253.5 @1×1M) | +0.4%, +0.4% | rel ≤ 0.014 | `trtllm_mla_backend.py`: forces `is_var_seq=False` (fixed-length scheduling) for the K3 fp8 MLA layout; 16-warp CTA for the B=1 `set_mla_kv_concat_q` grid; FP16-softmax attempt rejected (SM107-only) | **HARNESS OVERFIT — not production-safe as written**: the driver gives every request the same context length, so fixed-length scheduling passes CHECK here, but a real decode batch has mixed lengths. Must be gated on actual uniform `seq_lens` (or dropped) before any upstreaming; the harness needs a mixed-length point to catch this class |
| KDA 24 | FAIL (null) | 0.0% (379.3 → 379.3) | 0.0%, 0.0% | pass | tree returned to the seed after several rejected experiments | KDA plateau: rounds 23–24 null, 21 failed correctness — the remaining 52% is the MXFP4 MoE cubin |
| MLA 24 | PASS | +2.4% (261.6 → 255.4 @1×1M) | +2.2% @128×8k, +3.4% @16×64k | rel ≤ 0.014 | shared/routed alt-stream overlap in `KimiK3MoE._forward_fused` (`kimi_k3.py`) — the MLA best tree had not carried this lever yet | MLA cumulative 304.6 → 255.4 µs (−16.2% vs pristine) |
| KDA 23 | FAIL (null) | +0.3% (379.4 → 378.4; below the 0.5% floor) | 0.0%, +0.1% | pass | five ideas rejected cleanly (bf16-input MoE dispatch, TMA 4→3 stages, PDL-off, 896/top-2 route+quant specialization, …) | `best_tree` unchanged |
| MLA 23 | PASS | +1.5% (265.7 → 261.6 @1×1M) | +0.2%, +0.4% | rel ≤ 0.014 | `latent_up` (7168×3584) and `shared_down` (7168×6144) GEMMs routed to the BF16 TGV kernel (`kimi_k3.py`); a `merged_front` tactic experiment (296 µs) and a `q_b_proj` TGV (neutral) reverted | MLA cumulative 304.6 → 261.6 µs (−14.1% vs pristine) |
| **KDA 22** | **PASS** | **+1.3%** (384.4 → 379.4 @128×8k) | +2.5% @32, +3.1% @1 | rel 0.013 / 0.007 / 0.0 | bf16-state port of the fused KDA decode JIT kernel (`kda_fused_decode.cuh` + `.py`) | the lever trials 7/9/10 kept attempting finally pays off on top of the stacked tree; KDA cumulative 403.0 → 379.4 µs (−5.9% vs pristine) |
| MLA 22 | FAIL (null) | 0.0% (263.7 → 263.6) | 0.0%, +0.1% | pass | edits in `cutedsl_bf16_gemm.py`, `cutedsl_mla_backend.py`, `deepseek_v2.py` with no measurable effect | required gain 7.2% (3σ under the co-tenant); `best_tree` unchanged |
| KDA 21 | FAIL (correctness) | −2.6% (394.7 → 384.6, shared GPU) | −0.9% @32, 0.0% @1 | **rel 0.42 @32, 0.06 @1** | left a `route_quant_fused` 112/top-2 JIT specialization + `topk.py` change that breaks numerics at small batch; required gain was 7% (3σ under the busy co-tenant) | rejected on both gates; `best_tree` unchanged |
| KDA 20 | FAIL (infra) | — | — | — | agent ended near its seed (+15 lines `kimi_k3.py`); tried bf16-input MoE dispatch (383 µs, no gain), PDL-off, a variant that failed CHECK (rel 0.37) | judge's replay hit **CUDA OOM** (17 MB free): the co-tenant sglang scheduler grew to 167 GB on GPU 7. Not an agent result. Loop paused 20:25; auto-resumes (`resume_k3_loop_when_free.sh`) when an authorized GPU has room |

## Accuracy hardening (2026-09-24 16:05–16:27)

Both final best trees re-judged against **fresh pristine goldens under seeds 1 and 2** (new weights,
state, inputs; 5-rep timings vs the pristine tree at every point):

| tree | seed | CHECK (max_rel_err per point) | speedup vs pristine |
|---|---|---|---|
| MLA best | 1 | 0.006 / 0.008 / 0.004 | −16.8% @1×1M, −3.5% @128×8k, −4.7% @16×64k |
| MLA best | 2 | 0.010 / 0.010 / 0.005 | −16.8%, −3.5%, −5.0% |
| KDA best | 1 | 0.009 / 0.0 / 0.0 | −5.7% @128, −7.3% @32, −8.9% @1 |
| KDA best | 2 | 0.011 / 0.011 / 0.0 | −6.1%, −7.1%, −8.9% |

All PASS; the speedups reproduce under new seeds. Caveat stands: this is layer-level kernel equivalence
(≤ 2% relative error on bf16 outputs + exact post-step state), not an end-to-end model-quality eval.
The MLA case now also carries a **mixed-context-length** point (`16,65536,mix`: 64k/48k/32k/16k) so
uniform-length shortcuts fail CHECK if they are wrong. **Result for round 25's `is_var_seq=False`:** the
round-25 tree **passes** the mixed-length point (rel 0.0046, and −7.4% vs pristine there: 293.6 → 271.8 µs).
Reading FlashInfer (`mla/_core.py`): `is_var_seq=False` maps to `is_persistent=True`, i.e. it selects the
persistent TRT-LLM MLA kernel schedule; per-request `seq_lens` are still honoured, so it is a scheduling
choice, not a uniform-length assumption. The "harness overfit" flag on round 25 is therefore downgraded
to "agent's stated rationale was harness-specific, but the change is correct on mixed lengths" and the
MLA best tree stays at 253.5 µs.

## Campaign 4 — new input dimensions with warm start (2026-09-24 17:00–)

User direction: extend the loop to (a) large decode batches (B = 256/512, the DP-attention regime) and
(b) chunked prefill (popular chunk sizes), warm-starting from the accepted decode trees and giving the
agents the optimization history to retrieve.

**Warm start.** `k3_opt_history/` (built by `build_opt_history.py`): curated `TECHNIQUES.md` (accepted
levers + dead ends with reasons), `INDEX.md`/`history.json` over all 33 judged trials, and per trial the
judged patch, the round's incremental diff and every hypothesis/analysis the agent wrote. Mounted read-only
at `/workspace/opt_history` in every agent container; the task prompt asks for it to be read first and cited.

**Large-batch decode cases** (`issue_k3_{kda,mla}_b512.json`; KDA primary 512×8k, secondary 256×8k, 128×8k;
MLA primary 512×8k, secondary 256×8k, 16×64k mixed). New VibeSim presets `*_b512`; the MLA oracle initially
predicted 16 ms for the attention leaf at B=512 because the MLA worklet fed the group's total KV as one
request's context (Codex B12a fixed it, `kimi-k3-arch` 83847486: 1025/661/251 µs predicted vs 992/661/294
measured). Oracles on 8803 (KDA) / 8804 (MLA).

Transfer check (decode best trees judged on the new points against the pristine tree, GPU 7):

| case | point | pristine µs | best tree µs | Δ | rel err |
|---|---|---|---|---|---|
| MLA | 512×8k | 991.8 | 978.4 | −1.4% | 0.012 |
| MLA | 256×8k | 661.0 | 638.5 | −3.4% | 0.013 |
| MLA | 16×64k mixed | 294.4 | 272.7 | −7.3% | 0.005 |
| KDA | 512×8k | 724.5 | 677.4 | −6.5% | 0.017 |
| KDA | 256×8k | 506.4 | 482.8 | −4.7% | 0.015 |
| KDA | 128×8k | 403.0 | 378.4 | −6.1% | 0.013 |

KDA-b512 rounds 1–3 (45-min agents, warm-started): all null. ~13 ideas tried (TRT-LLM MoE tuner ceiling at
2× rows, bf16-activation MXFP4 path — no SM100 kernel at B≥256, route+quant JIT specialisations that never
engaged for the 112/top-2 path, KDA TMA stage counts, in-kernel TRT-LLM routing — slower and re-routed 2 of
512 tokens so the strict decode CHECK rejected it). Round 1 measured +1.2% at B=512 against a 1.5%
requirement (3σ inflated by concurrent VibeSim JIT fills on the shared GPU); rounds 2–3 were 0.0/+0.1%.
VibeSim ranks the routed MXFP4 MoE first at every batch (R0 961 µs vs R5 53 µs at B=512, 57% of the step):
the KDA layer is at the closed-cubin wall at B=512 as it was at B=128. MLA-b512: round 1 (20:13) was infra noise (B=512 baseline reps 2104/995/3363/3530/3419 µs while Codex B12 kernel-profiled on the same GPU; stopped and restarted after B12 with a clean baseline: pristine 991.7 µs, σ 0.4); round 2 FAIL null (0.00%: MoE tactic buckets, PDL toggle, low-priority overlap −1 µs, bf16 front GEMM rejected on correctness); round 3 (01:00–) tried bf16-activation MoE ×2 (no kernel), route+quant cap 64→512 (exact, −1.5 µs, kept), variable-schedule attention (null), tuning ceiling 512→1024 (null), TGV for the m=512 front GEMM (+8% slower) and shared-down (+2% slower) → FAIL null (judge: 978.3→976.6, +0.17% improvement); round 4 (01:47–02:37) six ideas (PDL policy regressed, variable-sequence scheduler neutral) → FAIL null 0.00%. MLA-b512 closed: inherited −1.4% at 512×8k is the result.

**Harness fidelity gap found 23:05 (via Codex B12):** the driver never called sglang's
`initialize_bf16_gemm_config()` (the production scheduler does; `auto` → `cutedsl` on SM100), so every bf16
GEMM ran cuBLAS in all runs so far, while production dispatches eligible shapes to the CuTe-DSL TGV/split-K
kernels. Quantified on GPU 7 with a new driver flag `--bf16-gemm-init` (CUDA-graph µs, legacy → production
dispatch): KDA pristine 406.9/255.6/134.7 → 407.0/255.5/132.6 (B=128/32/1), KDA best tree 388.6/245.2/128.5 →
388.6/245.3/124.4; MLA pristine 304.7/493.0/293.3 → 302.6/493.0/292.3 (1×1M / 128×8k / 16×64k mix), MLA best
tree 253.4/475.7/271.8 → 253.4/475.7/271.9. The decode shapes are essentially not TGV-eligible under
production's heuristic, so the reported gains stand (≤0.7% shift) and rounds 21/23 go beyond the production
heuristic rather than duplicating it. The original decode cases keep the legacy dispatch (history stays
consistent); the b512 and prefill cases run with `--bf16-gemm-init` from here on.

**Claude Opus 5.5 (Bedrock) as agent, b512 round 1 (02:56–03:50, GPUs 1/2 in parallel):** KDA PASS 675.2→600.4 @512 (+11.1% judge improvement), 482.7→472.4 @256, 380.2→378.2 @128 — one-shot MoE autotune (harness gap, ~9 pts) + vectorized bf16 state ld/st with launch_bounds in `kda_fused_decode.cuh` (kernel 92.6→83.0 µs) + bfa overlap limit 128→512; MLA PASS 976.3→904.6 @512 (+7.3%), 638.3→617.8 @256, mixed 271.8→270.7 — tuned MoE tactic (gap, ~6.4 pts) + fp8 `set_mla_kv_concat_q` satfinite/NOSAT kernel (17.8→8.0 µs). All exact. Both rejected splitting the merged front across streams. Round 2 started 03:49.

**Claude b512 round 2 (03:49–04:44):** KDA FAIL-by-noise (603.5→595.3 @512, +1.4%, exact; σ 7.3 µs → 3.65% required; fused-kernel prologue hoist + quad-row warp reduction, kernel 80.4→76.6 µs); MLA PASS 904.6→888.0 @512 (+1.8%): tail split of the persistent MLA decode kernel's last wave (68 trailing requests split into 136 half-KV pseudo-requests, LSE merge; new `kernels/ops/attention/mla_decode_tail_split.py`). Round 3 started 04:42. Driver autotune now caches tactics on disk to remove the per-rep tactic re-pick noise.

**Claude MLA-b512 round 3 (04:42–05:23): PASS** 887.3→877.8 @512 (+1.1%), 619.8→609.2 @256, mixed 269.5→261.5 (+3.0%): output-gate GEMM overlapped into the MLA decode kernel tail + pending residual add fused into the attn-res TMA kernel; exact. Claude MLA-b512 campaign done: 976.3→877.8 µs (3/3 PASS).

**Claude KDA-b512 round 3 (04:42–05:36): FAIL correctness** — 601.4→582.9 @512 (+3.1%) but rel 0.029/0.022/0.021 (8/1/1 rows over tol): attn-res TMA fused residual add (fp32 add + RNE) changed rounding; other parts (r2 kernel port via the KB, routed-before-shared capture order) exact. Operator-stacked tree without the fusion: same rel 0.029/0.022/0.021 (+2.4%) → the drift is the fused-decode kernel's reduction reorder (r2 port), FAIL; nothing salvaged. Claude KDA-b512 campaign: 1/3 PASS (r1 −11.1%).

**Claude MLA-prefill round 1 (04:52–05:45, GPU 6): PASS** 12128.5→11573.4 (+4.6%) / 8573.0→8226.0 (+4.1%) / 8057.8→7848.0 (+2.6%), bit-exact: Triton `mla_kv_pack_quantize_fp8` (one pass K/V pack + fp8 quant for chunk and prefix, Q cast once) + residual add fused into the attn-res TMA kernel (418→361 µs). First accepted prefill optimization. Round 2 started 05:45.

**Claude MLA-prefill round 2 (05:45–06:18): PASS** 11538.6→11229.3 @pf49152 (+2.7%), +0.7% / +1.3% others; prefix attention → CuTe-DSL JIT FMHA with softmax correction (rel 0.0156). Round 3 started 06:17. **Claude KDA-prefill round 1: judge OOM** on GPU 2 (169 GB tenant appeared; free 1.7 GB) → no verdict; the agent's tree (copy removal in the KDA prefill path + host-sync drop, 9007→8584 µs self-measured, bit-exact) is re-judged on a free GPU and rounds 2–3 continue there.

**Claude KDA-prefill round 1 (re-judged 06:21–06:32 on GPU 7): PASS** 9071.9→8564.0 (+5.6%) / 9169.2→8590.5 (+6.3%) / 8931.5→8453.2 (+5.4%), max rel 0.0, 0 rows over tol; copy removal (strided l2norm q/k, o_norm gate, v in recompute_w_u) + dropped int(query_start_loc[-1]) host sync. Rounds 2–3 continue on GPU 7.

**Claude MLA-prefill round 3 (06:17–07:08):** first verdict infra noise (27247/18194/18132 µs baselines: foreign job on GPU 6; kept as trial_3_verdict_v1_noise.json); re-judged clean on GPU 6: FAIL null (11277.4→11331.8 @pf49152, −0.5%; 4x4k +2.0%). MLA-prefill Claude campaign closed at the round-2 tree: 12128.5→11229.3 µs (−7.4%).

**Second fidelity gap (03:20, found by both Claude Opus 5.5 b512 agents):** no FlashInfer autotune warmup in the driver → fallback MXFP4 MoE tactic; new `--flashinfer-autotune`; pristine legacy → production (GPU 6): KDA 755.1→693.6 @512, 516.6→501.2 @256, 407.0→406.9 @128; MLA 991.6→930.2 @512, 662.0→650.5 @256, 295.3→294.3 mixed; B ≤ 128 unchanged. Claude round-1 PASSes on b512 are this warmup (harness-gap reproduction).

**Chunked prefill.** Driver point tag `B,L,pf[<prefix>]` (ForwardMode.EXTEND; KDA runs `chunk_kda` with the
carried-in conv/recurrent state, MLA runs the production `trtllm_mla` MHA_CHUNKED_KV path: fp8 ragged
attention + chunked prefix-KV merge; eager timing since sglang does not graph-capture prefill). Cases
`issue_k3_{kda,mla}_prefill.json`: 1×16384 first chunk (B200 default `chunked_prefill_size`), 1×16384 as
chunk 4 of a 64k prompt (prefix 49,152), 4×4096 mixed batch. Pristine baselines: KDA 22.1 / 22.5 / 22.1 ms,
MLA 20.6 / 29.5 / 17.8 ms (GEMM-dominated; ~6 ms of the MLA prefix point is the fp8 prefix attention).
Prefill correctness is judged per token (≤0.5% rows over tolerance, p99 row error ≤ tol, mean drift ≤1%,
state exact) because the MoE/GEMM autotuners pick among near-equal tactics at m = thousands across
processes, re-routing ~0.1% of tokens (replays within one process are identical). Prefill rounds wait for
VibeSim prefill support (Codex B12: kinds `kda_chunk_prefill`, `causal_conv1d_prefill`,
`mla_prefill_attention`, `mla_prefix_gather`, `mla_merge_state`; worklet prefill branches; presets), then
`run_k3_prefill_when_free.sh`.

## Artifacts

`patches/` — per-lever sglang patches (validated to reproduce the best trees), harness and VibeSim patches; see `K3_PIPELINE_AND_RESULTS.md` §6.

`iter_opt_eval_k3_{kda,mla}/trial_<k>_{verdict.json,agent.log,opt_run/,tree_judged.patch}`,
goldens `/raid/yilegu/eval_goldens/golden_k3_<key>/`, controls `run_k3_controls.sh`, direct runner
`run_k3_trials_direct.sh`, workspace branch `kimi-k3-loop`, VibeSim branch `kimi-k3`.
