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

## Artifacts

`iter_opt_eval_k3_{kda,mla}/trial_<k>_{verdict.json,agent.log,opt_run/,tree_judged.patch}`,
goldens `/raid/yilegu/eval_goldens/golden_k3_<key>/`, controls `run_k3_controls.sh`, direct runner
`run_k3_trials_direct.sh`, workspace branch `kimi-k3-loop`, VibeSim branch `kimi-k3`.
