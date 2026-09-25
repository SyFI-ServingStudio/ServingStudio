# MLA trial 20: agent iterations

## agent log.md

iter_00: baseline profiled and simulated; MoE MXFP4 was the top actionable leaf (379.160 us R0, 0.600686 necessary share); plain graph 265.696/485.952/300.608 us.

## iter_00
### hypothesis.md

No source edit was made in iteration 00. The next hypothesis is to extend the
existing correctness-preserving K3 route+quant fusion to the measured local
EP8 shape (112 experts, top-2). It will produce the same selected ids, weights,
packed routing values, and MXFP8 group scales as the current separate kernels,
but in one CUDA launch. The fused kernel only replaces preparation; expert
weights, attention, and recurrent state writes are unchanged.

### analysis.md

Iteration 00 measured the starting worktree (including the pre-existing
cutedsl_mla_backend.py change). The diagnostic profile used the required three
points and produced 29 graph launches per point:

- B=1, L=1048576: graph 273.824 us; kernel sum 316.765 us.
- B=128, L=8192: graph 481.696 us; kernel sum 535.011 us.
- B=16, L=65536: graph 303.616 us; kernel sum 347.866 us.

VibeSim was called after the profile. simulate with prediction=.:before built
prediction_id p_c7687e585abe4b5fbacdc89caf0c9761. The API accepts that value as
prediction_id for the analysis verbs. analyze/operator reported the largest
aggregate leaves as mxfp4_fused_moe (0.379160 ms, 32.72%) and
mla_decode_attention (0.358836 ms, 30.96%). analyze/run_summary reported
R0=1.158899 ms, R5=0.147162 ms, R6=0.147162 ms, optimality ratio 0.126985;
batching was 32.79%, hardware gap 36.88%, and necessary work 12.70% of R0.

optimality?scope=iter ranked the routed MoE leaf first:

- unified.mla.moe.mxfp4_fused_moe: R0=379.160 us, R5=8.573 us,
  R6=R7=227.756 us, necessary_share=0.600686, r0/r5=44.226.
- unified.mla.attention.mla_decode_attention: R0=358.836 us,
  R5=227.559 us, R6=R7=454.164 us, necessary_share=1.265660,
  r0/r5=1.577; kernels reported a cached TRT-LLM alternative.

kernels?kernel_set=mxfp4_fused_moe mapped the MoE node to backend
sglang_trtllm_mxfp4, shape hidden=3584, intermediate=3072, local experts=112,
top_k=2, activation=situ. The measured per-kernel table showed the two routed
TRT-LLM MXFP4 GEMMs plus separate routing and per-token-group quantization
launches. That mapping and the high MoE headroom make it the first target;
attention is left unchanged because its measured TRT-LLM FMHA path is already
the efficient cached alternative for the scored long-context shape.

### result.json

```
{
  "mode": "baseline",
  "plain_fixed_flags_graph_latency_us": {
    "B1_L1048576": 265.6959891,
    "B128_L8192": 485.9519899,
    "B16_L65536": 300.6080091
  },
  "golden": [
    "/tmp/golden_B1_L1048576.pt",
    "/tmp/golden_B128_L8192.pt",
    "/tmp/golden_B16_L65536.pt"
  ],
  "checks": "captured reference; replay checks are recorded in the edited iteration"
}
```

## iter_01
### hypothesis.md

The current path misses route_quant_fused coverage because its CUDA JIT is
specialized for 896 experts/top-16, while this measured one-rank Kimi-K3 layer
uses 112 local experts/top-2. Add a second compile-time router trait for
112/top-2 and dispatch to it from the same JIT entry point. The route algorithm
and quantizer are shared with the existing path; only valid expert count/top-k
constants and the scalar tail-safe load differ. The fused output remains the
same top-k ids/weights, packed ids, and UE8M0 scales, so the MoE result,
attention output, and all post-step state should be unchanged within the
driver's tolerance.
