# Kimi-K3 decoder layer on B200 — every technique that worked, across all shapes

*Consolidated 2026-09-27 from campaigns 2–6 (Codex gpt-5.6-luna and Claude Opus 5.5 agents, sglang v0.5.20, one
rank of the cookbook TP8/EP8 deployment: 12 attention heads, 112 local MXFP4 experts, fp8-e4m3 KV, bf16 KDA state).
Companion documents: `K3_PIPELINE_AND_RESULTS.md` (pipeline, campaigns 1–5), `K3_CAMPAIGN6_SHAPES_LCPREFILL.md`
(campaign 6), `k3_opt_history/TECHNIQUES.md` (agent-facing version).*

**Conventions.** Δ = change in step latency, **negative = faster**. Decode numbers are CUDA-graph replay µs, prefill
numbers eager step ms. Every Δ is a judge verdict (5-rep medians, fresh container, correctness vs pristine goldens)
unless marked *agent*. Patch base path:
`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang/` —
each patch below is `<base>/<relative>`; every chain is validated to reproduce its case's best tree exactly
(`export_k3_patches.sh`). Apply inside `python/sglang` of `lmsysorg/sglang:v0.5.20` with `patch -p1`.

## 1. The shapes

| id | shape | regime | judged in |
|---|---|---|---|
| D1 | 1 × 1M decode | single request, long context: MLA reads 0.6 GB fp8 KV; KDA is launch-bound (state, no KV) | MLA decode, shapes |
| D2 | 16 × 64k decode (`mix` for MLA: per-request 64k/48k/32k/16k) | small batch, long context | MLA decode (mix), b512 (mix), shapes |
| D3 | 128 × 8k decode | the production throughput point; MoE weight-bandwidth-bound (~85 active experts) | KDA/MLA decode, b512, shapes |
| D4 | 32 × 8k decode | mid batch | KDA decode, shapes |
| D5 | 1 × 8k decode | latency-bound single request | KDA decode, shapes |
| D6 | 256 × 8k decode | MoE starts to turn compute-bound | b512 |
| D7 | 512 × 8k decode | DP-attention large batch; MXFP4 MoE compute-bound (~1024 local rows) | b512 |
| P1 | 1 × 16k first chunk (prefix 0) | chunked prefill, GEMM-bound MoE at 32k local rows | prefill |
| P2 | 1 × 16k chunk @ prefix 48k | prefill with carried state / prefix attention | prefill |
| P3 | 4 × 4k first chunks | mixed prefill batch | prefill |
| P4 | 1 × 16k chunk @ prefix 131,072 | long-context prefill | lcprefill |
| P5 | 1 × 32k chunk @ prefix 131,072 | long-context prefill, 64k local MoE rows | lcprefill |
| P6 | 1 × 16k chunk @ prefix 245,760 (completes 262,144) | long-context prefill, MLA prefix attention = 66% of step | lcprefill (primary) |
| P7 | 1 × 32k chunk @ prefix 229,376 (completes 262,144) | long-context prefill, heaviest point | lcprefill |
| V1 | 64 × (3 draft + 1 bonus) verify @ 8k (`vk3`) | speculative-decode TARGET_VERIFY, 256 query rows; judged as a CUDA-graph replay (`--verify-graph`) | verify (primary) |
| V2 | 128 × 4 verify @ 8k | verify at the throughput batch | verify |
| V3 | 16 × 4 verify @ 64k | verify with long context | verify |
| X1 | 64 decodes @ 8k + one 16k prefill chunk (`mx16384`) | MIXED (EXTEND) batch, eager; 16,448 rows | mixed (primary) |
| X2 | 128 decodes @ 8k + one 4k chunk (`mx4096`) | decode-heavy mixed batch | mixed |
| X3 | 64 decodes @ 8k + one 16k chunk @ prefix 49,152 | mixed batch with a long-prefix chunk | mixed |

## 2. Cumulative results per shape (best tree of the campaign that owns the shape, vs pristine)

| shape | KDA pristine → best | Δ | MLA pristine → best | Δ | rechecked seeds |
|---|---|---|---|---|---|
| D1 1×1M | 136.6 → 101.8 µs | **−25.5%** | 304.5 → 216.4 µs | **−28.9%** | 0, 1, 2 |
| D2 16×64k | 200.0 → 163.2 | −18.4% | 292.3 → 261.4 (mix) | −10.6% | 0, 1, 2 |
| D3 128×8k | 403.0 → 374.2 | −7.1% | 493.0 → 462.5 | −6.2% | 0, 1, 2 |
| D4 32×8k | 261.6 → 226.5 | −13.4% | 263.6 → 247.2 | −6.2% | 0, 1, 2 |
| D5 1×8k | 136.6 → 105.7 | −22.6% | 146.9 → 128.4 | −12.6% | 0, 1, 2 |
| D6 256×8k | 506.4 → 482.8 (decode tree) | −4.7% | 661.0 → 638.5 (decode tree) | −3.4% | 0 |
| D7 512×8k | 675.2 → 593.5 (b512 Claude, r2 accepted at 15 reps) | −12.1% (≈9 pts autotune harness gap) | 976.3 → 877.8 (b512 Claude) | −10.1% (≈6.4 pts gap) | 0, 1, 2 (r1 tree); r2 at 15 reps seed 0 |
| P1 16k first chunk | 9.07 → 8.56 ms | −5.6% | 12.13 → 11.57 (r1) | −4.0% | 0, 1, 2 |
| P2 16k @ 48k | 8.87 → ~8.3 | −6.3% | 12.13 → 11.23 | −7.4% | 0, 1, 2 |
| P4 16k @ 128k | 9.20 → 8.14 | −11.5% | 18.37 → 14.99 | −18.4% | 0, 1, 2 |
| P5 32k @ 128k | 18.07 → 16.14 | −10.7% | 35.09 → 30.33 | −13.6% | 0, 1, 2 |
| P6 16k @ 245k | 9.03 → 8.13 | **−10.0%** | 27.69 → 20.45 | **−26.1%** | 0, 1, 2 |
| P7 32k @ 229k | 17.94 → 16.15 | −10.0% | 48.36 → 39.95 | −17.4% | 0, 1, 2 |
| V1 64×4 verify @8k (graph) | 509.3 → 480.6 µs (seed 1) | −5.6% / −5.7% | 497.0 → 466.4 µs (seed 1) | −6.2% / −6.0% | 0, 1, 2 |
| V2 128×4 verify @8k | 653.8 → 599.4 | −8.3% / −8.1% | 650.6 → 620.2 | −4.7% / −4.7% | 0, 1, 2 |
| V3 16×4 verify @64k | 322.9 → 302.4 | −6.3% / −6.0% | 417.1 → 388.4 | −6.9% / −6.7% | 0, 1, 2 |
| X1 64 dec + 16k chunk @8k | seed −13.4%, + r2 −3.4% (judge) | (seed recheck 2467 pending) | 15065 → 8290 µs (judge chain) | **≈ −45% + −2.3%** (seed recheck pending) | 0; 1, 2 pending |
| X2 128 dec + 4k chunk | seed −18.2%, r2 flat | pending | 11512 → 2481 (judge chain) | ≈ −78% | 0; 1, 2 pending |
| X3 64 dec + 16k chunk @48k | seed −12.7%, + r2 −4.5% | pending | 37898 → 13081 (judge chain) | ≈ −65% | 0; 1, 2 pending |

(P4–P7 updated 2026-09-27 after the follow-up rounds 4–6 with the B12c-corrected long-context oracles; V1–V3 and X1–X3
added 2026-09-28 from follow-up #4; see `K3_CAMPAIGN6_SHAPES_LCPREFILL.md` §9.6. The verify trees: KDA = shapes tree +
r3, MLA = pristine + r1 + r3 (every accepted MLA decode tree regressed the verify step). The mixed trees: lcprefill
tree + accepted mixed rounds.)

The D1–D5 trees are the shape-matched campaign's (`iter_opt_eval_k3_{kda,mla}_shapes_claude/best_tree`), which
contain the decode levers + the b512 Claude levers + the shape rounds; D7 is the b512 Claude tree; P1–P3 the prefill
Claude tree; P4–P7 the lcprefill Claude tree (= prefill Claude tree + two rounds each).

## 3. Technique × shape matrix

Legend: **gain** = judged faster on that shape; ok = engaged, neutral (≤0.5%); — = does not engage (guard/regime);
✗ = measured slower there; blank = not measured on that shape.

| # | technique | layer | D1 1×1M | D2 16×64k | D3 128×8k | D4 32×8k | D5 1×8k | D6/D7 256–512 | P1–P3 prefill ≤48k | P4–P7 long prefill |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | trtllm-gen MLA decode kernel (was CuteDSL) | MLA | **gain −12.1%** | ok −0.7% | ok −1.6% | ok | ok | ok | — | — |
| T2 | persistent MLA decode schedule (`is_var_seq=False`) + 16-warp KV-concat CTA | MLA | gain −0.8% | **gain −7.4% (mix)** | ok −0.4% | ok | ok | ok | — | — |
| T3 | persistent-decode tail split (LSE merge) | MLA | ok | ok | ok | ok | ok | **gain −1.8% @512** | — | — |
| T4 | B=1 KV split k=8 + LSE merge for L≥64k, q replicas from the fused concat | MLA | **gain −10.2%** | ok | — | — | — | — | — | — |
| T5 | output-gate GEMM issued into the decode kernel's tail wave | MLA | ok | gain −3.0% (mix) | ok | ok | ok | **gain −1.1% @512** | — | — |
| T6 | fp32-output front GEMMs (m≤16) cuBLAS → CuTe TGV | MLA | **gain −1.5%** | ok | — | — | gain (part of −12.6%) | ✗ +8% at m=512 (rejected) | — | — |
| T7 | latent_up / shared_down (m≤16) → CuTe TGV | MLA | **gain −1.5%** | ok | — | — | gain | ✗ +2% at m=512 (rejected) | — | — |
| T8 | shared `down` GEMM (m≤128) → CuTe bf16 GEMM | KDA | gain | gain | **gain (in −4.8%)** | gain | gain | — | — | — |
| T9 | split merged MoE front at m≤32: routed rows first, gate-up + shared on alt stream | KDA | **gain −11.5%** | gain −6.8% | ok −0.5% | gain −5.4% | **gain −13.0%** | — | — | — |
| T10 | shared experts ⟂ routed MoE on the alt stream | MLA, KDA | **gain −2.4%** | gain −3.4% | gain −2.2% | gain | gain | **gain (carries the −1.4/−6.5% to 512)** | | |
| T11 | bfa side-stream overlap limit 128 → 512 | KDA | | | ok | | | **gain @512** | — | — |
| T12 | L2 prefetch of o_proj + routed-front weights on the alt stream | KDA | **gain −5.6%** | gain −3.5% | ok | gain −2.0% | ok (+0.9%, noise) | | — | — |
| T13 | packed KDA decode fast path kept for lower-bounded gate | KDA | gain | gain | **gain (in −4.8%)** | gain −5.4% | gain −6.0% | gain | — | — |
| T14 | bf16-state fused KDA decode kernel (conv + delta rule + gated norm, one launch) | KDA | gain −3.1% | gain | **gain −1.3%** | gain −2.5% | gain −3.1% | gain | — | — |
| T15 | vectorized bf16 state ld/st + `__launch_bounds__` in the fused decode kernel | KDA | gain | gain | ok | gain | gain | **gain @512 (92.6→83.0 µs kernel)** | — | — |
| T16 | Triton recurrent kernel `num_warps` retune | KDA | ok | ok | gain (small) | ok | ok | ok | — | — |
| T17 | attn-res direct kernel for T≤32 rows | KDA | **gain −2.0%** | gain −2.4% | ok | ok | **gain −7.2%** | | | |
| T18 | attn-res `cluster_small` kernel + fused MoE finalize+RMSNorm | MLA | gain −0.9% | ok | ok | ok | gain −1.6% | | | |
| T19 | fp8 KV-concat kernel with satfinite convert + exact fixup (17.8→8.0 µs) | MLA | gain | gain | gain | gain | gain | **gain @512 (in −7.3%)** | — | — |
| T20 | residual add fused into the attn-res TMA kernel | MLA | gain | gain (mix −3.0%) | ok | ok | ok | gain @512 | **gain (418→361 µs @16k)** | gain |
| T21 | fp8 K/V pack + quantize in one Triton pass (replaces ~800 µs glue) | MLA | — | — | — | — | — | — | **gain −4.6%** | gain (inherited) |
| T22 | prefix attention TRT-LLM ragged → CuTe-DSL FMHA (softmax correction, fp8 P prescale) | MLA | — | — | — | — | — | — | **gain −2.7% @48k** | (silently off until T23) |
| T23 | 16-byte-align the prefix-chunk `cum_seqlen_k` row so T22 engages at long prefixes; empirical KV-split policy | MLA | — | — | — | — | — | — | ok (already aligned) | **gain −12.7% / −7.5% / −7.4% / −4.4%** |
| T24 | prefix-FMHA softmax: 40 of 128 exp2 per tile on the FMA pipe (degree-2 poly) | MLA | — | — | — | — | — | — | (not measured) | **gain −4.7% / −5.7% / −2.1% / −3.1%** |
| T25 | strided KDA chunk kernels (l2norm, gated norm, recompute_w_u), no copies, no host sync | KDA | — | — | — | — | — | — | **gain −5.6% / −6.3%** | gain (inherited −4.2…−6.4%) |
| T26 | CUDA `mma.sync` h-scan replacing Triton `chunk_delta_h` (516→281 µs) | KDA | — | — | — | — | — | — | ok −1.1% (below 3σ) | **gain −1.6% / −3.1%** |
| T27 | conv1d BLOCK_M=16 / 2 warps | KDA | — | — | — | — | — | — | | gain (with T26) |
| T28 | fused SiLU-and-mul for the MoE activation | KDA | | | | | | | | **gain −1.4% / −4.0%** |
| T29 | route+quant JIT specialised to 112 experts / top-2 (harness-specific) | MLA | gain −0.75% | ok | ok | — | — | ok | | |
| T30 | one-shot FlashInfer autotune of the MXFP4 MoE tactic (= production warmup; harness gap) | both | ok (0) | ok | ok (0) | | | **gain 6–9 pts @512, 2–3 pts @256** | (flag `--flashinfer-autotune` instead) | (flag) |
| T31 | fused-decode kernel prologue load hoist + quad-row warp reduction (accepted on the 15-rep re-judge) | KDA | | | ok −0.5% | | | **gain −1.0% @512, −1.1% @256** | — | — |
| T32 | fp8 packing of the prefix-chunk `kv_b_proj` output fused into the GEMM epilogue | MLA | — | — | — | — | — | — | (not measured) | **gain −1.1% @245k, −0.7% @128k** |
| T33 | causal in-chunk attention pass on the alt stream, overlapping the prefix passes; join before `merge_state` | MLA | — | — | — | — | — | — | (not measured) | **gain −1.6% @245k, −0.8% @128k** |
| T34 | warp-specialized 4-stage CUDA h-scan + pinned config for the inter-chunk solve | KDA | — | — | — | — | — | — | (not measured) | **gain −1.6% @245k, −3.5% @128k** |

### 3b. Verify and mixed shapes (follow-up #4, 2026-09-28)

| # | technique | layer | V1 64×4 @8k | V2 128×4 @8k | V3 16×4 @64k | X1 64 dec + 16k | X2 128 dec + 4k | X3 64 dec + 16k @48k |
|---|---|---|---|---|---|---|---|---|
| T35 | CUDA JIT verify recurrence kernel (`kda_verify_recurrent.cuh`) replacing the Triton `fused_recurrent` verify path | KDA | **gain −3.55%** | **gain −4.85%** | gain −2.1% | — | — | — |
| T36 | satfinite fp8 cvt in `set_mla_kv_concat_q` + shared/routed alt-stream overlap + output-gate `g_proj` enqueued after the MLA node (fork at `forward_absorb_core`, ≤512 tokens) | MLA | **gain −4.15%** | **gain −5.0%** | **gain −5.8%** | | | |
| T37 | residual add fused into the attn-res TMA aggregate + SM carveout on the shared-expert down GEMM | MLA | **gain −2.1%** | ok −0.3% | ok −0.8% | | | |
| T38 | lift the one-wave dispatch guard so the 4-stage CUDA `kda_chunk_h` scan serves mixed batches (N = 65/129 sequences) | KDA | — | — | — | **gain −3.4%** | ok −0.1% | **gain −4.5%** |
| T39 | decode rows of a MIXED batch → absorbed MLA decode kernel; chunked-prefix MHA only for the prefill request | MLA | — | — | — | **gain −45%** | **gain −77%** | **gain −65%** |
| T40 | causal in-chunk attention pass restricted to the prefill rows of the mixed batch | MLA | — | — | — | **gain −2.3%** | gain −1.5% | gain −2.0% |
| (inherited) shapes-tree levers T1–T20 on the verify step | KDA | seed −12% (eager) → 0 under the graph metric: the eager gain was host time | | +3.6% at V3 eager | | | |
| (inherited) accepted MLA decode trees on the verify step | MLA | ✗ +5.8…+31% (shapes tree worst) → MLA verify was run from pristine | ✗ | ✗ | | | |
| (inherited) lcprefill trees on the mixed step | both | | | | KDA −13.4% / MLA −29.0% | −18.2% / −54.4% | −12.7% / −17.0% |

## 4. The techniques, by mechanism

### 4.1 Attention kernel selection and scheduling (MLA)

- **T1 — CuteDSL → trtllm-gen MLA decode.** `<base>/mla/01_r08_cutedsl_to_trtllm_mla_decode.patch` (5 lines,
  `srt/layers/attention/cutedsl_mla_backend.py`). At D1 the step is a read of 0.6 GB fp8 latent KV; the TRT-LLM
  generation kernel streams pages with fewer split partials and no merge pass, sitting closer to the 172 µs HBM floor.
  Bit-exact. Decode-only; the DCP path keeps CuteDSL.
- **T2 — persistent schedule + KV-concat CTA.** `<base>/mla/06_r25_is_var_seq_persistent_kvconcat_warps.patch`.
  `is_var_seq=False` → FlashInfer `is_persistent=True`: resident CTAs pull KV pages instead of one CTA per
  (request, split); per-request lengths are still honoured (verified on the mixed batch: −7.4%). The 13-item B=1
  KV-concat grid runs in one 16-warp CTA.
- **T3 — tail split.** `<base>/claude/mla_b512_02_r2_persistent_decode_tail_split.patch` (`mla_decode_tail_split.py`).
  B = 148·3 + 68 leaves a last wave of 68 SMs; the trailing requests' KV is halved into pseudo-requests and merged by
  LSE, filling the wave. Exact. Large-batch lever.
- **T4 — B=1 KV split k=8.** `<base>/claude/mla_shapes_01_r1_b1_kv_split8_lse_merge_kvconcat_q_replicas.patch`. At
  B=1 the decode kernel runs one CTA per head over 1M tokens (12 CTAs); eight KV chunks with an LSE merge put 96
  CTAs on the read, and the fused fp8 concat kernel writes the eight q replicas in the same pass (no extra launch).
  Gated to L≥64k, 256 MB workspace. rel 0.006.
- **T5 — output gate in the decode tail.** `<base>/claude/mla_b512_03_r3_outgate_tail_overlap_attnres_fused_add.patch`.
  The output-gate GEMM is launched so it overlaps the decode kernel's last, under-occupied wave. Exact.
- **T22/T23/T24 — prefix attention on CuTe-DSL FMHA.** `<base>/claude/mla_prefill_02_r2_prefix_attention_cutedsl_fmha.patch`
  moves the non-causal prefix pass off the TRT-LLM ragged FMHA (fp8 P prescale 2^8, full softmax correction; rel
  0.0156 at 48k). `<base>/claude/mla_lcprefill_01_r1.patch` fixes the 16-byte misalignment of the second row of
  `prefix_chunk_cu_seq_lens` that made the backend fall back to the ragged kernel at prefixes not divisible by the
  chunk size — at 245k that was 66% of the step — and adds the S=4 (≤6 waves) / S=2 KV-split policy; bit-exact.
  `<base>/claude/mla_lcprefill_02_r2.patch` rebalances the MUFU-bound softmax onto the FMA pipe (degree-2 exp2
  polynomial for 40/128 elements per tile) without changing the fp8-quantized P; bit-exact.

### 4.2 Small-m GEMM dispatch (weight-streaming regime, m ≤ 16…128)

- **T6/T7 — fp32-output front GEMMs and latent_up/shared_down → CuTe TGV.** `<base>/mla/03_r21_front_fp32_gemm_cute_tgv.patch`,
  `<base>/mla/04_r23_latent_up_shared_down_bf16_tgv.patch` (`_k3_bf16_gemm` in `srt/models/kimi_k3.py`). cuBLAS
  tiles for m ≤ 16 leave most SMs idle on a 230 MB weight stream; the TGV kernel is built for that GEMV-like regime
  and has an fp32-output variant so router logits stay exact. **Guarded to m ≤ 16 on purpose:** at m = 512 the same
  kernel is 8% (front) / 2% (shared_down) slower — measured and rejected in the b512 rounds.
- **T8 — shared `down` → CuTe bf16 GEMM at m ≤ 128.** Part of `<base>/kda/01_stacked_fastpath_overlap_cutedsl_gemm_warps.patch`.
- **T9 — split the merged MoE front at m ≤ 32.** `<base>/claude/kda_shapes_01_r1_split_moe_front_small_m_alt_stream_tgv_fp32.patch`.
  The merged front GEMM (router + shared gate-up) ran serially ahead of a latency-bound routed-MoE chain (41 µs on the
  critical path at B=1); issuing the routed rows first and the gate-up/shared halves on the alt stream (fp32 TGV
  halves keep routing exact) hides them under the MoE. Bit-exact at B=1/16; the largest single decode lever for KDA.

### 4.3 Critical-path overlap and cache warming

- **T10 — shared experts on the alt stream.** `<base>/mla/05_r24_shared_routed_alt_stream_overlap.patch` (MLA,
  `KimiK3MoE._forward_fused`) and part of `<base>/kda/01_…` (KDA, single-rank event join). Disjoint output slices,
  no arithmetic reordered, exact; engages at every batch size — the reason the decode trees still transfer to 512.
  Once the shared path is on the side stream, making it faster barely moves the critical path (levers stop adding).
- **T11 — bfa overlap limit 128 → 512.** Part of `<base>/claude/kda_b512_01_r1_moe_autotune_fused_decode_vector_ldst_bfa_overlap.patch`.
- **T12 — L2 prefetch.** `<base>/claude/kda_shapes_02_r2_l2_prefetch_oproj_routed_front.patch` (new JIT kernel
  `l2_prefetch.cuh`). At B=1 the o_proj and routed-front weight reads follow the KDA recurrence, during which HBM is
  idle; a 32-CTA prefetch on the alt stream pulls those weights into L2 so the GEMMs hit cache (VibeSim: both nodes
  bandwidth-bound, R0/R5 1.3–1.7). Bit-exact.

### 4.4 KDA recurrent decode kernels

- **T13 — keep the packed fast path.** Part of `<base>/kda/01_…`: the dispatcher excluded `lower_bound` layers from
  `fused_recurrent_kda_packed_decode` although the kernel implements the lower-bounded sigmoid gate; K3 uses
  `lower_bound = −5`, so every layer took the multi-launch path. Bit-exact, batch-independent.
- **T14 — bf16-state fused decode kernel.** `<base>/kda/02_r22_bf16_state_fused_kda_decode_kernel.patch`
  (`kda_fused_decode.cuh/.py`). `covered()` required an fp32 state; the port loads/stores `__nv_bfloat162` pairs and
  inserts the unfused chain's intermediate roundings (`bf16_round`, `conv_silu_bf16`), so one launch replaces three
  within rel 0.013.
- **T15 — vectorized ld/st + launch bounds.** Part of `<base>/claude/kda_b512_01_…`: 8-byte state accesses and
  `__launch_bounds__` (≥5 blocks/SM), 92.6 → 83.0 µs kernel time at B=512; carries to every batch.
- **T16 — `num_warps` retune** of the Triton recurrent kernel for the B=128 grid (part of `<base>/kda/01_…`).

### 4.5 KDA chunked-prefill kernels

- **T25 — strided chunk kernels.** `<base>/claude/kda_prefill_01_r1_strided_chunk_kernels_no_copies_no_host_sync.patch`
  (8 files). l2norm, gated norm and `recompute_w_u` accept strided q/k/v, removing three materialised copies of the
  16k × (12 × 128) activations; the `int(query_start_loc[-1])` host sync is dropped. Bit-exact.
- **T26/T27 — CUDA h-scan + conv1d tuning.** `<base>/claude/kda_lcprefill_01_r1.patch` (the earlier unaccepted
  `UNACCEPTED_kda_prefill_r2_cuda_kda_chunk_h_hscan.patch`, ported via the warm-start history). The Triton
  `chunk_delta_h` state recurrence is serial over chunks; the `mma.sync` version halves it. Below the 3σ floor at
  16k/48k, above it on the long-context judge. Bit-exact.
- **T28 — fused SiLU-and-mul.** `<base>/claude/kda_lcprefill_02_r2.patch` (`situ_and_mul.cuh`): one pass over
  32–64k rows instead of two. rel 0.007.

### 4.6 Elementwise and fusion kernels

- **T19 — fp8 KV-concat satfinite.** Part of `<base>/claude/mla_b512_01_r1_moe_autotune_kvconcat_satfinite.patch`:
  hardware `cvt.satfinite` for the fp8 write with an exact no-saturate fixup, 17.8 → 8.0 µs, every batch.
- **T20 — attn-res fused residual add.** In `<base>/claude/mla_b512_03_…` and `<base>/claude/mla_prefill_01_…`: the
  pending residual row rides in the TMA ring and is added in registers (418 → 361 µs at 16k tokens). Exact for MLA;
  the analogous KDA attempt changed rounding (rel 0.02–0.03) and was rejected.
- **T21 — fp8 K/V pack + quantize.** `<base>/claude/mla_prefill_01_r1_kv_pack_quantize_fp8_attnres_fused_add.patch`
  (Triton `mla_kv_pack_quantize_fp8`): one pass packs and quantizes K/V for chunk and prefix and casts Q once;
  replaces ~800 µs of elementwise glue per 16k chunk. Bit-exact.
- **T17 — attn-res direct kernel (T ≤ 32).** `<base>/claude/kda_shapes_03_r3_fused_decode_attnres_tuning.patch`:
  skips the TMA-ring setup that dominated the kernel at B=1/16; the T ≤ 128 variant raised rel error to 0.019 and was dropped.
- **T18 — `cluster_small` attn-res + fused finalize+RMSNorm.** `<base>/claude/mla_shapes_02_r2_attnres_cluster_small_moe_finalize_rmsnorm.patch`:
  two launch-bound B=1 tails fused; −0.9%, above 3σ.

### 4.7 Routing and MoE tactic (know what is harness-specific)

- **T29 — route+quant JIT for 112/top-2.** `<base>/mla/02_r20_route_quant_fused_112_top2.patch`. Engages the fused
  router+quant launch for the EP8-rank emulation; production routes 896/top-16, where the fused path already exists.
  Kept because it is exact and removes a harness artefact. The same specialisation broke KDA numerics at B=32/1 in one
  round (rel 0.42) — always check every point.
- **T30 — one-shot MoE autotune.** In both `*_b512_01_r1_*` patches: reproduces production's FlashInfer autotune
  warmup; without it the MXFP4 cubin runs a fallback tactic (identical at B ≤ 128, ~9% slower at 512). This is a
  **harness gap, not a kernel improvement** — the driver now carries `--flashinfer-autotune`; when porting, keep the
  kernel parts of those patches and drop the autotune.

### 4.8 Speculative verify and mixed prefill+decode batches (follow-up #4)

- **The verify metric must be a CUDA-graph replay.** The eager TARGET_VERIFY step is ~65% host launch time (KDA
  64×4 @8k: 1420 µs eager vs 502 µs graph; MLA 1523 vs 493). Under CPU contention its σ was 92–418 µs, so the 3σ gate
  demanded 22–99% and rejected KDA r1 (−14.7% eager) and r2 (−21.7% eager) — and r1's eager gain was **entirely host
  time** (−0.05% under the graph metric). Production graph-captures verify; the driver now does too (`--verify-graph`,
  graph replay bit-exact vs eager, σ ≈ 0.05 µs).
- **T35 (KDA verify).** `<base>/claude/kda_verify_01_r3.patch`: a CUDA JIT kernel for the verify recurrence (the Triton
  path was 2× off its roofline). Store-policy / occupancy / TMA variants of the new kernel were neutral.
- **T36–T37 (MLA verify, from pristine).** `mla_verify_01_r1.patch`, `mla_verify_02_r3.patch`. The accepted MLA
  decode trees all *regressed* the verify step (+5.8% b512 tree, +7.5% decode best, +31% shapes tree): their small-m
  GEMM dispatch and B=1 attention scheduling assume one query row per request, verify has four. Start verify work
  from pristine or re-gate those levers on `q_len`.
- **T38 (KDA mixed).** `kda_mixed_01_r2.patch` (4 files, 70 lines): the CUDA h-scan's one-wave guard was measured on
  the old single-stage kernel; with the 4-stage ring the CUDA scan beats Triton (645 µs in the mixed batch) even at
  6240 CTAs.
- **T39 (MLA mixed, the largest single win of the whole effort).** `mla_mixed_01_r2.patch` (3 files, 224 lines). sglang's
  `handle_attention_trtllm_mla` routes every `is_extend_without_speculative()` batch through `MHA_CHUNKED_KV`, so the
  1-token decode rows of a mixed chunk each run a chunked-prefix MHA pass (prefix_chunk_len = capacity // batch_size:
  64–128 tiny passes per layer). Routing those rows to the absorbed decode kernel (trtllm-gen) cuts the step 45–77%.
  Applies to any deployment with `--enable-mixed-chunk`; a no-op for pure decode / pure prefill. **Do not** merge the
  prefix into one chunk to go faster: log2-LSE vs ln `merge_state` makes the output chunk-dependent (CHECK FAIL).
- **T40 (MLA mixed).** `mla_mixed_02_r4.patch`: with T39 in place the causal in-chunk pass is needed only for the
  prefill rows; restricting it is bit-exact and worth another 2%.

## 5. Dead ends, across shapes

| idea | shapes tried | outcome |
|---|---|---|
| eager-metric "wins" on verify (fewer launches / less Python) | V1–V3 | 0% under the graph metric — host time only |
| inheriting the MLA decode best trees into verify | V1–V3 | +5.8…+31% regression; start from pristine |
| trtllm-gen MLA kernel at q_len = 4 (verify) | V1 | 2× slower than CuteDSL (microbench) |
| one big prefix chunk instead of chunked prefix passes (mixed) | X3 | faster (11095 µs) but CHECK FAIL: `merge_state` LSE base mismatch makes output chunk-dependent |
| MoE tail add3 → in-place `addmm_` into the attn-res prefix sum (mixed) | X1–X3 | exact, −1.2%, but +4.3% at X2 → rejected |
| TGV / CuTe small-m GEMMs at m ≥ 256 | D7 | +8% front, +2% shared_down → rejected; small-m only |
| bf16-activation MXFP4 MoE dispatch | D3, D6, D7 | no SM100 tactic / no gain |
| MoE tactic buckets, PDL off, tuning ceiling 512 → 1024 | D3, D7 | neutral or regression |
| in-kernel TRT-LLM routing | D7 | slower and re-routed 2/512 tokens (CHECK FAIL) |
| fused-decode reduction reordering (KDA) | D7 | rel 0.02–0.03 → correctness FAIL |
| attn-res fused add on KDA | D7 | changes rounding → FAIL (fine on MLA) |
| custom small-m MXFP4 MoE kernel | D1 | correct, +0.4% |
| side-stream overlap of in-proj GEMMs in prefill | P6 | ≤0.3%: persistent GEMMs serialize, step is gap-free |
| KV-split prefix FMHA without wave-aware policy | P7 | regressed the 32k point until the S=4/S=2 policy |
| forcing the fused KDA decode kernel onto the bf16 state without porting | D3 | crash |
| making the shared GEMM faster once it is on the side stream | D3 | no critical-path change |

## 6. Transfer rules learned

1. **Bandwidth-bound decode (B ≤ 16):** attention kernel choice and split (T1, T4), small-m GEMM dispatch (T6–T9),
   critical-path overlap and L2 prefetch (T10, T12), launch fusion (T17, T18) — all carry between D1 and D5 for KDA
   because its step is context-independent; for MLA the attention levers are D1-specific.
2. **Mid batch (32–128):** overlap (T10), KDA fast path and fused kernel (T13–T15) carry; small-m GEMM levers switch off by guard.
3. **Large batch (256–512):** only overlap, the KDA kernel work, the tail split and the KV-concat kernel survive; the
   MXFP4 cubin dominates and the autotune warmup must be on (T30) or the comparison is invalid.
4. **Prefill:** none of the decode levers apply (m = 16k–64k); the wins are elementwise removal (T21, T25, T28),
   the prefix-attention kernel and its enablement (T22–T24), and the KDA scan kernel (T26). Check that an accepted
   "fast path" actually engages at the new shape — T22 was silently disabled by an alignment corner case until T23.
5. **Correctness envelope:** most accepted levers are bit-exact; the ones that are not (T14 rel 0.013, T4 0.006,
   T28 0.007, T22 0.0156) all passed seeds 0/1/2 with fresh goldens; the closest call is the KDA shape tree at D4
   under seed 2 (rel 0.0198 vs the 0.02 tolerance).
6. **Speculative verify (q_len = 4 per request):** judge it as a graph replay or you measure the host. The decode levers
   that assume one query row per request (small-m GEMM dispatch, B=1 KV split, persistent decode schedule) regress
   verify; the wins are the verify-specific kernels (T35) and stream-level overlap that is row-count agnostic (T36, T37).
7. **Mixed prefill+decode batches:** check which attention path each *row class* takes before tuning kernels — the
   whole-batch dispatch (T39) was worth 45–77%, more than every kernel lever in this document combined; the KDA analogue
   was a stale dispatch guard (T38). Prefill-tree levers transfer to the prefill rows of a mixed batch (−13…−54% seed
   transfer) but the decode rows need decode-path treatment.
