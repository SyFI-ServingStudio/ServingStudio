# KDA_MIXED_CLAUDE trial 2: agent iterations

## agent log.md

# log
- iter_00: baseline plain 8529/2714/8560 us (mx16384 / 128x mx4096 / mx16384p49152); VibeSim p_7935...: kda_chunk_prefill R0/R5=43x, merged_front 1.81x; Triton h-scan 645 us in the mixed batch (CUDA kda_chunk_h gated off by a one-wave guard).

## iter_00
### analysis.md

# iter_00 -- baseline analysis
VibeSim prediction: `p_7935251272b445188fbd5338ed932b36` (run `k3_kda_mixed`, the fixed before-state
prediction; `simulate?prediction=.:before` resolves to it). Sum over the 3 mixed cases: R0 22.85 ms,
R5 10.91 ms, optimality 0.48.

optimality (scope=iter), top headroom (R0-R5):
| node | R0 ms | R5 ms | R0/R5 |
|---|---|---|---|
| moe.merged_front_prefill (fp32-out front GEMM) | 6.81 | 3.75 | 1.81 |
| attention.kda_chunk_prefill (chunk_kda chain) | 2.49 | 0.057 | 43.7 |
| moe.mxfp4_fused_moe_prefill | 4.24 | 2.16 | 1.96 |
| attention.qkvbfg_a_proj_prefill | 2.62 | 1.44 | 1.81 |
| moe.shared_down_prefill | 2.52 | 1.44 | 1.75 |
necessary_share is null for all nodes (R6 not modelled for this run).

Measured per-kernel table (profile_B64_L8192mx16384.json, eager 8.60 ms, 31 launches):
nvjet fp32-out front GEMM 2280 us; 2x nvjet badd (shared_down/latent_up folded) 1368; nvjet in-proj 928;
MXFP4 MoE gemm1 875 + gemm2 321; **Triton chunk_gated_delta_rule_fwd_kernel_h_blockdim64 645 us**;
attn_res 399; o_proj 222; conv_l2norm 187; situ 163; chunk intra/solve/o/recompute ~425.

Decision: kda_chunk_prefill has by far the worst R0/R5 (43x). Inside it, the h-scan runs the TRITON
kernel even though the tree carries the CUDA kda_chunk_h (lcprefill_claude_1/4): its dispatch guard
`cdiv(V,16)*N*H <= #SMs` (from kda_prefill_claude_2, measured on the older single-stage kernel)
rejects the mixed batch (N = 65 sequences -> 6240 CTAs). Target it first: bit-exact, no numerics
risk. The GEMMs (merged front at ~1.65 PFLOP/s ~ 73% of dense peak) are the next candidates.

## iter_01
### hypothesis.md

# iter_01 -- hypothesis
Node: attention.kda_chunk_prefill -> h-scan (chunk_gated_delta_rule_fwd_h, fla/chunk_delta_h.py).

Builds on: lcprefill_claude_1/4 (the CUDA warp-specialized kda_chunk_h, bit-exact vs Triton, already in
the tree). Rules out / revisits: kda_prefill_claude_2's one-wave dispatch guard
`cdiv(V,16)*N*H <= #SMs`, measured on the older single-stage kernel. With the 4-stage cp.async ring
the CUDA scan is ~3x faster than Triton on long sequences even at 6240 CTAs, but the guard sends
every mixed batch (N = 65/129 sequences) to Triton (645 us at the primary point).

Hypothesis: the CUDA scan loses only when its per-CTA fixed cost (1-chunk decode rows) dominates.
Two changes:
1. Guard on shape instead of wave count: use CUDA when extra_chunks (chunks beyond each
   sequence's first) >= N. Primary: N=65, extra=255 -> CUDA. mx4096: N=129, extra=63 -> Triton
   (same path as baseline). p49152: same as primary.
2. Kernel: the per-CTA chunk-offset prefix was a serial dependent-load walk over cu_seqlens
   (up to 64-128 loads for the last decode rows); make it a lane-strided sum + __reduce_add_sync.
   Integer math, so still bit-exact.

Microbench (bench_h3.py, fair: Triton metadata precomputed), CUDA vs Triton us:
16384+64d 211/672, 4096+128d 228/223, 2048+128d 213/149, 512+256d 407/221, 16384+256d 515/694,
4x4096 131/189, 32x512 150/156. Ring stages 2/3 (bench_h4.py) don't help short rows.
A first guard `4*extra >= N` also sent mx4096 to CUDA: in-layer 235 us vs 188 us Triton,
+2% at that point -> tightened to `extra >= N`.
Expected: primary -300..-360 us (h-scan 645 -> ~285 us), others unchanged.

(diff.patch: 44 lines, files: b/python/sglang/kernels/jit/csrc/attention/kda_chunk_h.cuh, b/python/sglang/kernels/ops/attention/fla/chunk_delta_h.py)
