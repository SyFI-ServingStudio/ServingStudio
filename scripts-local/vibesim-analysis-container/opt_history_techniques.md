# Kimi-K3 single-layer optimization: techniques that worked, and dead ends

*Curated 2026-09-24 from 30+ judged trials on sglang v0.5.20 / B200 (one rank of the cookbook
TP8/EP8 deployment: 12 attention heads, 112 local MXFP4 experts, fp8-e4m3 KV, bf16 KDA state).
All effect sizes below are CUDA-graph decode-step numbers on the DECODE workload (B ≤ 128 @ 8k,
1×1M, 16×64k). A new workload (B = 256/512, chunked prefill) shifts the bottleneck: MoE moves from
weight-bandwidth-bound to GEMM-bound above ~B=256 local rows ≈ 512-1024, prefill attention and
KDA chunk kernels appear. Use this as a map of levers and pitfalls, not as a result table.*

## A. Accepted levers (all in the current best trees; bit-exact or rel_err ≤ 0.014)

| # | lever | where | effect on decode | transfer notes |
|---|---|---|---|---|
| A1 | Route non-DCP MLA decode to the TRT-LLM MLA generation kernel instead of the CuteDSL split-KV kernel | `srt/layers/attention/cutedsl_mla_backend.py` (15 lines) | −12.1% @1×1M | decode-only lever; prefill uses a different kernel |
| A2 | fp32-output front GEMMs (15984×7168, 6016×7168; m ≤ 16) cuBLAS → CuTe TGV | `srt/models/kimi_k3.py` + `kernels/ops/gemm/cutedsl_bf16_gemm.py` | +1.5% @1×1M, +3.2% @128×8k | TGV is a small-m GEMV-style kernel: **not** for m ≥ 256 or prefill; re-check the m guard |
| A3 | `latent_up` (7168×3584) and `shared_down` (7168×6144) → BF16 TGV / CuteDSL bf16 GEMM | `srt/models/kimi_k3.py` | +1.5% (MLA), −3% (KDA trial 8) | same small-m caveat as A2 |
| A4 | Overlap the dense shared experts with the routed MXFP4 MoE on the layer's alt stream; join before the collective | `KimiK3MoE._forward_fused` in `srt/models/kimi_k3.py` | +2.4% (MLA), ≈+2% (KDA) | valid at any m as long as the two branches write disjoint slices; at large m both branches are compute-bound so the gain shrinks |
| A5 | bf16-state port of the fused KDA decode JIT kernel (`covered()` accepted only fp32 states) | `kernels/jit/csrc/.../kda_fused_decode.cuh` + `kernels/ops/.../kda_fused_decode.py` | +1.3% @128, +3.1% @1 | decode-only (recurrent step); prefill runs `chunk_kda` |
| A6 | Keep the packed KDA decode fast path for K3's lower-bounded sigmoid gate (dispatcher wrongly excluded `lower_bound` layers) | `srt/layers/attention/linear/kda_backend.py` | ≈−2% | decode-only |
| A7 | Triton `fused_recurrent_kda_packed_decode` launch tuning (num_warps for the B=128 grid) | `kernels/ops/attention/fla/...` | small, bit-exact | any B: re-tune per grid |
| A8 | `is_var_seq=False` → FlashInfer's persistent TRT-LLM MLA decode schedule for the fp8 layout; 16-warp CTA for the B=1 KV-concat grid | `srt/layers/attention/trtllm_mla_backend.py`, `kernels/ops/attention/set_mla_kv_concat_q.py` | +0.8% @1×1M, −7.4% @16×64k mixed | verified with per-request lengths 64k/48k/32k/16k; scheduling choice, not a length assumption |
| A9 | `route_quant_fused` JIT specialised for the 112-expert / top-2 routing shape | `kernels/jit/csrc/moe/route_quant_fused.cuh`, `kernels/ops/moe/moe_route_quant_fused.py`, `srt/layers/moe/topk.py` | +0.75% | harness-specific (production routes over 896 experts / top-16 globally); the same specialisation broke numerics on KDA at B=32/1 (rel 0.42, round 21) — check every point |

Levers A2–A4 do not add linearly: once the shared GEMM runs on the side stream, making it faster
barely moves the critical path (stacked check −4.8%, not −7%).

## B. Dead ends on the decode workload (and why)

| idea | outcome | why / when it might still apply |
|---|---|---|
| Force the fused KDA decode kernel onto the bf16 state without porting it | crash at B=128 (KDA 5) | the port (A5) is the right form |
| bf16-activation MoE dispatch instead of MXFP4 act-quant | no B200 tactic / no gain | the MXFP4 cubin is weight-BW-bound at B ≤ 128; at m ≥ 1024 the GEMM becomes compute-bound and activation format matters again — re-evaluate for B=512 / prefill |
| TRT-LLM MoE tactic buckets / tactic re-selection | no effect | tactics are already autotuned per m; new m values (512, 1024, 16k+) are untuned — worth one look |
| PDL (programmatic dependent launch) off | regression | keep on |
| Fused MXFP4 router / route+quant on KDA | +6 µs or numerics FAIL | routing is ~2% of the step |
| Fused finalize+shared JIT kernel (MLA 27) | regressed, reverted | |
| `cutedsl_bf16_gemm.py` tile tweaks (MLA 22, 26) | 0.0% | |
| KV-concat CTA size changes beyond A8 | ±0 | |
| Making the shared `down` GEMM faster once it is already on the side stream | no critical-path change | measure the critical path (graph replay), not the kernel |

## C. What the remaining time is (decode, B=128 @ 8k, after all levers)

KDA 379 µs: ~52% MXFP4 routed MoE (closed TRT-LLM cubin, weight-BW-bound over ~85 active experts),
~20% KDA recurrent chain, rest projections/norms/router. MLA 253 µs @1×1M: MLA decode attention
~45%, MoE ~40%. At B = 256–512 the MoE reads the same weights but does 2–4× the FLOPs: expect the
cubin to turn compute-bound (tactic choice, activation quant, expert-parallel dispatch/finalize
ordering, and the shared-expert overlap become the visible levers). In chunked prefill the MoE is a
large-m GEMM (m = chunk × local top-k share), KDA runs `chunk_kda` (chunked scan) and MLA prefill
runs the ragged/paged prefill kernels plus the fp8 KV write.

## D. Harness facts that have bitten agents

- Measure before/after with the IDENTICAL flags; `--split`/`--profile-kernels` perturb graph time ~10%.
- Re-run every point before finishing: a kernel that works at B=1 may reject B=128 / long context
  (dtype/layout guards) → hard FAIL.
- The judge carries `.py .cu .cuh .h .hpp .cc .cpp .inc .jinja` edits only; keep JIT sources inside
  the sglang tree.
- Correctness is always vs the ORIGINAL model's output; rel_err ≤ 0.02 on output and post-step
  state, every point.
- Verify (`vk`) points are judged as CUDA-graph replays (`--verify-graph`). Anything that only
  removes host launch work (fewer Python ops, fewer launches) is worth exactly 0 there: the KDA
  verify r1 tree showed −14.7% in eager mode and −0.05% under the graph metric. Measure with the
  same flags the judge uses; if `latency_mode` says `graph`, optimize GPU time only.

## E. Speculative-verify (TARGET_VERIFY, 4 tokens/request) and mixed prefill+decode batches

| # | lever | where | effect | notes |
|---|---|---|---|---|
| E1 | CUDA JIT verify recurrence kernel replacing the Triton `fused_recurrent` verify path (KDA) | `kernels/jit/csrc/attention/kda_verify_recurrent.cuh` + `kernels/ops/attention/kda_verify_recurrent.py`, `kda_backend.py` | −3.55% @64×8k vk3, −4.85% @128×8k | the Triton verify scan was 2× off its roofline; store-policy / occupancy / TMA variants of the new kernel were neutral |
| E2 | satfinite fp8 cvt in `set_mla_kv_concat_q` + shared/routed alt-stream overlap + MLA output-gate `g_proj` enqueued after the MLA node (fork event at `forward_absorb_core`, overlap up to 512 tokens) (MLA) | `kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh`, `srt/models/kimi_k3.py` | −4.15% @64×8k vk3, −5.8% @16×64k, bit-exact | from PRISTINE: every accepted MLA decode tree regressed verify (+5.8…+31%), so verify cases must not inherit them blindly |
| E3 | residual add fused into the attn_res TMA aggregate + SM carveout on the shared-expert down GEMM so the MoE routing kernel is not blocked (MLA) | `kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh`, `srt/layers/attn_residual.py`, `srt/models/kimi_k3.py` | −2.09% @64×8k vk3 | forking the shared experts after topk was worse (482.5 vs 474.5) |
| E4 | Lift the one-wave dispatch guard (`cdiv(V,16)*N*H <= #SMs`) so the 4-stage cp.async CUDA `kda_chunk_h` scan also serves mixed batches with many short sequences (N = 65/129) (KDA mixed) | `kernels/ops/attention/kda_chunk_h.py` / `fla/chunk_delta_h.py` | −3.4% @64 dec + 16k chunk, −4.5% @48k prefix | the guard was measured on the old single-stage kernel; the Triton h-scan cost 645 µs in the mixed batch |
| E5 | In a MIXED (EXTEND) batch, route the 1-token decode rows to the absorbed MLA decode kernel (trtllm-gen) and run the chunked-prefix MHA only for the prefill request (MLA mixed) | `trtllm_mla_backend.py`, `forward_batch_deepseek_mha_mixin.py`, `attention_forward_methods/forward_mha.py` | **−45% @64 dec + 16k chunk, −77% @128 dec + 4k, −65% @48k prefix** | production's `handle_attention_trtllm_mla` sends every extend batch through `MHA_CHUNKED_KV` (64–128 tiny prefix passes per layer for the decode rows); applies to deployments with `--enable-mixed-chunk`. Do NOT merge the prefix into one chunk: log2-LSE vs ln `merge_state` makes the output chunk-dependent (CHECK FAIL) |
| E6 | Restrict the causal in-chunk attention pass to the prefill rows of the mixed batch (MLA mixed, on top of E5) | same three files | −2.3% @64 dec + 16k chunk | bit-exact; the decode rows no longer need the causal pass at all |

Dead ends here: trtllm-gen MLA at q_len=4 (2× slower than CuteDSL, microbench); MLA split_kv override,
TGV / deep_gemm GEMM swaps at these m; variants of the KV-concat kernel beyond E2 (±0).
