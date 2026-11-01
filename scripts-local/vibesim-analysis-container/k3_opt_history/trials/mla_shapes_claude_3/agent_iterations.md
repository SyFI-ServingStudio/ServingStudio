# MLA_SHAPES_CLAUDE trial 3: agent iterations

## iter_00
### analysis.md

# iter_00 — baseline (starting tree, no edit)
Plain (same fixed flags): 1x1M 218.3 | 16x64k mix 259.4 | 128x8k 462.1 | 32x8k 249.1 | 1x8k 128.3 us. Golden -> /workspace/opt_run/golden.pt.

## VibeSim
simulate(.:before) -> p_c7687e585abe4b5fbacdc89caf0c9761 (fixed before-state; id does not resolve in analyze, queried via `.:k3_mla`).
run_summary: optimality 0.127; batching 33%, hardware_gap 37%, imbalance 18%.
optimality (scope=iter, us summed over its cases):
- unified.mla.moe.mxfp4_fused_moe: r0 379.2 / r5 8.6 / r6 227.8, r0/r5 44.2, necessary_share 0.60 -> kernel-efficiency bottleneck (top headroom node). kernels: has_cached_alternative=false.
- mla_decode_attention: 358.8 / 227.6 / 454.2 (r6>r0) -> bandwidth-bound; only scheduling left.
- merged_front 158.8/121.1 (1.31), shared_down 1.47, latent_up 1.65 -> near limit.

## Measured (nsys, graph replay, 1x1M, span 213 us)
pre-attn chain 18 | fmha 101 (604 MB, 6.0 TB/s, grid 144, KV split 8) | post-attn 19 | merged_front TGV 36 (229 MB, 6.35 TB/s) |
routed MoE chain 30.5 (route_quant 9.6, routingIndicesBlock 5.6, GEMM1 9.5, GEMM2 8.4, finalize+norm 3.3; shared_down 15.3 in parallel) | latent_up 10.5 + add3 3.
1x8k: same MoE window 30.3 us (GEMM1 14.9 there, overlapping shared_down).

## Calibration microbench (scratch/bw_bench.py)
TGV GEMV reaches 6.5-6.6 TB/s independent of CTA count (125 vs 148 CTAs: 6.49 vs 6.60 TB/s; 56 CTAs reach 98 GB/s/CTA)
-> HBM practical ceiling ~6.5 TB/s; dense GEMMs are at it. fmha at 6.0-6.2 TB/s: <=8 us left, closed cubin, split factor already swept (prior round: 2/4/8/16).

## Decision
MoE window: bytes = shared_down 88 MB + 2 experts x 16.5 MB = 121 MB -> 18.6 us at 6.5 TB/s, measured 30.5 us.
At B=1 the TRT-LLM routed MoE is latency-bound (GEMM2 16.5 MB in 8.4 us = 2 TB/s, 56 CTAs; routingIndicesBlockKernel 5.6 us
of pure permutation bookkeeping). Target = mxfp4_fused_moe node (VibeSim top, kernel-efficiency rung) for small m.
