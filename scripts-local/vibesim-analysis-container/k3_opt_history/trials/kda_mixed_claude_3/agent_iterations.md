# KDA_MIXED_CLAUDE trial 3: agent iterations

## agent log.md

iter_00 baseline: mx16384 8320.1 | mx4096 2726.0 | p49152 8301.7 (all CHECK pass)
iter_01 MoE autotune exact num_tokens: mx16384 8248.4 | mx4096 2502.9 | p49152 8326.4 (all pass)

## iter_00
### hypothesis.md

# iter_00: baseline only (no edit). Prior trials: kda_mixed_claude_2 (CUDA kda_chunk_h, conv_l2norm fusion).
Candidates: E1 MoE tune at exact num_tokens; E2 split trailing 1-token decode rows out of chunk_kda;
E3 bf16 front + separate router GEMM; E4 deep_gemm in-proj; E5 merge bfa into in-proj.

### analysis.md

# iter_00 baseline analysis (primary 64,8192,mx16384 = 8320 us)
Per-kernel (--profile-kernels): front fp32 GEMM N=15984 2304; shared_down+latent_up addmm 1391; in-proj 950;
MoE gemm1 895 / gemm2 326; attn_res 2x198; kda_chunk_h 242; o_proj 225; situ 163; intra 126; solve 108;
chunk_o 108; conv_l2norm 100; w_u 100; finalize 70; cumsum 54; bfa 47; quant 47; rmsnorm 37; lng 31.
nsys: no idle gaps; shared-expert alt-stream overlap is effectively serial (persistent nvjet GEMM holds SMs).
VibeSim (run=k3_kda_mixed, scope=iter) top R0-R5 headroom: merged_front 1.81x, kda_chunk_prefill 43x,
mxfp4 MoE 1.96x, qkvbfg 1.81x, shared_down 1.75x.
GEMM microbench (bench/gemm1.py): cuBLAS already best for front/shared_down; deep_gemm ~10% better for in-proj.

### result.json

```
{"iter":0,"change":"baseline","latency_us":{"64,8192,mx16384":8320.1,"128,8192,mx4096":2726.0,"64,8192,mx16384p49152":8301.7},"check":"all pass"}
```

## iter_01
### hypothesis.md

# iter_01 hypothesis (node: unified.kda.moe mxfp4 routed MoE)
Tune at exact n (tune_max_num_tokens=n) so the tactic is profiled on the real shape. Expected gain is mostly
on mx4096 (~150 us), with a small gain on the primary. Numerics are unchanged (same kernels, different tactic).
Prior trials: iter_00 baseline. A/B'd with the env toggle K3X_MOE_EXACT (ab/base0 vs ab/moeexact).

### analysis.md

# iter_01 analysis
MoE trtllm-gen mxfp4 autotune: sglang passes tune_max_num_tokens=next_power_of_2(n); for n=4224 (mx4096 point)
this tunes the 8192 bucket and picks a tactic tuned for twice the tokens. MoE gemm1+gemm2 is ~1.2 ms of the
primary point and a larger share of mx4096.

### result.json

```
{"iter":1,"change":"MoE autotune at exact num_tokens","before_us":{"64,8192,mx16384":8320.1,"128,8192,mx4096":2726.0,"64,8192,mx16384p49152":8301.7},"after_us":{"64,8192,mx16384":8248.4,"128,8192,mx4096":2502.9,"64,8192,mx16384p49152":8326.4},"check":"all pass"}
```

(diff.patch: 13 lines, files: /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)
