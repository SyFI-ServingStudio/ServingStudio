# KDA trial 4: agent iterations

## agent log.md

iter_00: overlapped KimiK3MoE shared tail with routed MXFP4 tail; CHECK passed at all points, B128 630.240->550.176 us, B32 365.056->308.768 us, B1 180.736->183.744 us; rejected pending B1 guard.
iter_01: gated the overlap for num_tokens>1; exact /tmp/golden.pt replay CHECK passed at all points, B128 630.240->550.400 us, B32 365.056->309.792 us, B1 180.736->181.696 us; accepted.

## iter_00
### hypothesis.md

# Iteration 00 hypothesis

In the plain-TP fused-front path of `KimiK3MoE._forward_fused`, the merged
front GEMM has already completed before either tail branch reads its views.
`shared_output` and `latent` are disjoint slices of the same buffer, and the
shared-expert GEMM only writes `shared_output` while the routed MXFP4 call only
writes `latent`. Launching `_forward_shared` on the already supplied MoE side
stream after an explicit dependency on the front GEMM lets it overlap the
routed MXFP4 work. A wait on the current stream before the existing collective,
normalization, up projection, and final add preserves the same operation order
and values; only independent GPU work is concurrent. No weights, routing,
activation, recurrent state, or output layout changes.

This should reduce the critical path by the shared tail time, especially at
B=128, while leaving the B=32 and B=1 numerical results and post-step KDA
state unchanged.

### analysis.md

# Iteration 00 baseline

The required initial prediction was built with:

    GET /api/v1/simulate?prediction=.:before

VibeSim returned prediction id `p_38ccb83a6c4e44298f45b2251bd4cf54`, fixed
workspace `k3_kda`, architecture `kimi_k3_sglang`, and NVIDIA B200.
The analysis endpoints use the latest simulated prediction when no prediction
query is supplied, so the recorded calls were `analyze?level=operator`,
`analyze?level=run_summary`, `analyze?level=iteration`,
`optimality?scope=iter`, and `kernels?kernel_set=<node>`.

The run-summary prediction is 1.006 ms of modeled kernel work. The largest
operator is `unified.kda.moe.mxfp4_fused_moe` at 0.543 ms (53.96 percent),
inside `unified.kda.moe` at 0.764 ms (the largest node). Its roofline ladder is
R0=0.542878 ms and R5=0.009836 ms, giving R0/R5=55.19 and headroom
R0-R5=0.533042 ms. `necessary_share` is unavailable because R6 is null in
this fixed prediction. The parent MoE node has R0/R5=4.89 and headroom
0.608094 ms. The next leaf candidates are the KDA fused input projection
(R0/R5=3.45, R0=0.145 ms) and recurrent decode (R0/R5=5.21,
R0=0.0425 ms), both much smaller in absolute time.

The leaf drill-down identifies backend `sglang_trtllm_mxfp4`, shape
`hidden_size=3584`, `intermediate_size=3072`, `num_local_experts=112`,
`top_k=16`, MXFP4 E2M1/UE8M0 group-32 weights, and
`has_cached_alternative=false`. The source call is
`python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py:1108`,
`trtllm_fp4_block_scale_routed_moe`.

The profiler's B=128 graph replay table confirms that this source path is
really launched: the two largest kernels are the MXFP4 routed MoE BMMs at
293.02 us and 147.52 us. The full graph has 26 launches and 630.24 us
latency in the baseline profile. B=32 has 27 launches and 365.06 us; B=1
has 25 launches and 180.74 us.

The tested overlap produced exact output and state checks at every point, with
replay results of 550.176 us (B=128), 308.768 us (B=32), and 183.744 us
(B=1). Because the B=1 result regressed versus the 180.736 us baseline, this
iteration is rejected as the final change. The post-edit profile was also
re-measured: 622.0 us, 363.0 us, and 182.8 us respectively; run-to-run
variation is visible, but the replay result consistently shows the small-batch
side-stream launch/wait overhead.

### result.json

```
==============================================================================
kimi_single_layer_decode v3 (K3/KDA)  experts=112 EP=8 layer_idx=5 iters=40 warmup=10 split=False moe_backend=flashinfer_mxfp4 attention_backend=n/a page_size=1 attn_heads=12 cuda_graph=True kv_cache_dtype=bf16 mamba_ssm_dtype=bfloat16
torch=2.13.0+cu130  dev=NVIDIA B200
sglang=0.5.20
==============================================================================
[cfg] wrote /tmp/k3_cfg/config.json  experts=112 (EP=8) top_k=16 shared=2 layers=8  moe_backend=flashinfer_mxfp4 hidden_act=situ  attn_type=kda target_layer=5 mla_heads=64 kda_heads=12 kda_layers=[1, 2, 3, 4, 5, 6, 7, 8] full_attn=[]
[moe] flashinfer_mxfp4_moe_precision=default
[moe] runner_backend=MoeRunnerBackend.FLASHINFER_MXFP4 is_flashinfer_mxfp4=True is_deep_gemm=False
[boot] tp=1 attn_tp=1 world=1  is_kda_layer(5)=True  attention_arch=1 dtype=torch.bfloat16
[layer] mxfp4 quant_config=Mxfp4Config quant_format=None
[layer] UnquantizedLinearMethod from sglang.srt.layers.quantization.unquant
[layer] patched Mxfp4Config.get_quant_method: LinearBase/RadixAttention -> bf16 unquantized fallback
[layer] KimiK3DecoderLayer(layer_idx=5): self_attn=KimiK3DeltaAttention (KDA=True MLA=False)  mlp=KimiK3MoE (MoE=True)
[layer] float params=0.242B (fp32 kept: 0.02M)
[layer] random-init 0.242B float params -> cuda/bf16/eval; conv_weights refreshed dev=cuda:0
[mxfp4] experts=FusedMoE tensors:
[mxfp4]   w13_weight                       (112, 6144, 1792)      torch.uint8 <- e2m1 randint
[mxfp4]   w13_weight_scale                 (112, 6144, 112)       torch.uint8 <- ue8m0 fill 127
[mxfp4]   w13_weight_bias                  (112, 6144)            torch.bfloat16 <- bias normal
[mxfp4]   w2_weight                        (112, 3584, 1536)      torch.uint8 <- e2m1 randint
[mxfp4]   w2_weight_scale                  (112, 3584, 96)        torch.uint8 <- ue8m0 fill 127
[mxfp4]   w2_weight_bias                   (112, 3584)            torch.bfloat16 <- bias normal
[mxfp4] quant_method=Mxfp4MoEMethod use_flashinfer=True use_marlin=False _fi_kernel=trtllm_sm100 precision=default
[mxfp4] quant_method.process_weights_after_loading(experts) OK
[layer] post_load_merge: front_w=(15984, 7168) fused_front_eligible=True bias_dtype=torch.float32 bfa_w=True kda_fused_decode_ready=True
[moe] resolved expert method: {'class': 'Mxfp4MoEMethod', 'use_flashinfer': True, 'use_marlin': False, '_fi_kernel': 'trtllm_sm100', 'flashinfer_mxfp4_moe_precision': 'defau
[... truncated]
```

(diff.patch: 24 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_01
### hypothesis.md

# Iteration 01 hypothesis

Guard the new side-stream overlap with `num_tokens > 1`. B=1 then executes
the exact pre-edit launch sequence, so it cannot pay the extra stream launch
and wait. B=32 and B=128 retain the independent shared/routed execution and
the synchronization before the existing tail, so their values and post-step
state remain unchanged.

### analysis.md

# Iteration 01 analysis

The current-code profile carried forward from iteration 00 was measured after
the shared/routed tail overlap. VibeSim was simulated again with
`prediction=.:before` and returned the same fixed prediction id
`p_38ccb83a6c4e44298f45b2251bd4cf54`. The operator, run-summary, and
iteration analyses still identify `unified.kda.moe.mxfp4_fused_moe` as the
largest leaf (R0=0.542878 ms, R5=0.009836 ms, R0/R5=55.19,
headroom=0.533042 ms; `necessary_share` unavailable because R6 is null).
The parent `unified.kda.moe` remains the largest node (R0/R5=4.89,
headroom=0.608094 ms). The kernel drill-down still reports backend
`sglang_trtllm_mxfp4`, no cached alternative, and the same two large BMM
kernels confirmed by the profiler.

The remaining issue is not numerical or dataflow correctness: the overlap
adds a second-stream dependency even when the shared tail is too small to
hide. The B=1 replay was 183.744 us versus 180.736 us before the edit, while
B=128 and B=32 improved substantially. The next edit keeps the original
serialized path for one-token batches and retains overlap for B>=2.

The post-edit profile launched 26/27/25 kernels and measured 622.1/362.9/180.7
us in that run. The exact `/tmp/golden.pt` replay acceptance measurement is
550.400/309.792/181.696 us graph latency for B=128/32/1, recorded in
`result.json`. The
post-edit VibeSim re-analysis returned the fixed prediction above and the same
leaf/kernel mapping; no new safe kernel alternative was exposed.

### result.json

```
==============================================================================
kimi_single_layer_decode v3 (K3/KDA)  experts=112 EP=8 layer_idx=5 iters=40 warmup=10 split=False moe_backend=flashinfer_mxfp4 attention_backend=n/a page_size=1 attn_heads=12 cuda_graph=True kv_cache_dtype=bf16 mamba_ssm_dtype=bfloat16
torch=2.13.0+cu130  dev=NVIDIA B200
sglang=0.5.20
==============================================================================
[cfg] wrote /tmp/k3_cfg/config.json  experts=112 (EP=8) top_k=16 shared=2 layers=8  moe_backend=flashinfer_mxfp4 hidden_act=situ  attn_type=kda target_layer=5 mla_heads=64 kda_heads=12 kda_layers=[1, 2, 3, 4, 5, 6, 7, 8] full_attn=[]
[moe] flashinfer_mxfp4_moe_precision=default
[moe] runner_backend=MoeRunnerBackend.FLASHINFER_MXFP4 is_flashinfer_mxfp4=True is_deep_gemm=False
[boot] tp=1 attn_tp=1 world=1  is_kda_layer(5)=True  attention_arch=1 dtype=torch.bfloat16
[layer] mxfp4 quant_config=Mxfp4Config quant_format=None
[layer] UnquantizedLinearMethod from sglang.srt.layers.quantization.unquant
[layer] patched Mxfp4Config.get_quant_method: LinearBase/RadixAttention -> bf16 unquantized fallback
[layer] KimiK3DecoderLayer(layer_idx=5): self_attn=KimiK3DeltaAttention (KDA=True MLA=False)  mlp=KimiK3MoE (MoE=True)
[layer] float params=0.242B (fp32 kept: 0.02M)
[layer] random-init 0.242B float params -> cuda/bf16/eval; conv_weights refreshed dev=cuda:0
[mxfp4] experts=FusedMoE tensors:
[mxfp4]   w13_weight                       (112, 6144, 1792)      torch.uint8 <- e2m1 randint
[mxfp4]   w13_weight_scale                 (112, 6144, 112)       torch.uint8 <- ue8m0 fill 127
[mxfp4]   w13_weight_bias                  (112, 6144)            torch.bfloat16 <- bias normal
[mxfp4]   w2_weight                        (112, 3584, 1536)      torch.uint8 <- e2m1 randint
[mxfp4]   w2_weight_scale                  (112, 3584, 96)        torch.uint8 <- ue8m0 fill 127
[mxfp4]   w2_weight_bias                   (112, 3584)            torch.bfloat16 <- bias normal
[mxfp4] quant_method=Mxfp4MoEMethod use_flashinfer=True use_marlin=False _fi_kernel=trtllm_sm100 precision=default
[mxfp4] quant_method.process_weights_after_loading(experts) OK
[layer] post_load_merge: front_w=(15984, 7168) fused_front_eligible=True bias_dtype=torch.float32 bfa_w=True kda_fused_decode_ready=True
[moe] resolved expert method: {'class': 'Mxfp4MoEMethod', 'use_flashinfer': True, 'use_marlin': False, '_fi_kernel': 'trtllm_sm100', 'flashinfer_mxfp4_moe_precision': 'defau
[... truncated]
```

(diff.patch: 11 lines, files: python/sglang/srt/models/kimi_k3.py)
