# MLA_B512_CLAUDE trial 3: agent iterations

## agent log.md

# Kimi-K3 MLA layer, B200 -- latency_us (plain flags) B512@8k / B256@8k / B16@64k mix
- iter_00: baseline starting tree 888.2 / 619.3 / 269.7; golden captured; MoE tactic space fully swept (1736, no gain); GEMM backends all equal.
- iter_01: output-gate GEMM forked before / launched after the MLA decode kernel (fills its tail): 879.0 / 613.8 / 265.5; CHECK pass x3, rel 0.0.
- iter_02: front GEMM split (routed first, shared gate_up on alt stream), post/pre: 895.4/609.5/271.6 & 895.0/620.9/271.6 -> RULED OUT (routing cluster kernel blocked, shared_down lands on GEMM2); reverted to iter_01 (879.97/613.57/265.50, CHECK pass).
- iter_03: pending residual add fused into attn-res TMA kernel (addend row in the TMA ring, add in registers; kFuseAdd/run_add): 879.97/613.57/265.50 -> 876.96/609.63/261.44; CHECK pass x3, rel 0.0. (prologue-add first version 888.0 rejected)
- final: tree = iter_01 + iter_03; replay all points 876.99 / 608.58 / 261.47 (baseline 888.2 / 619.3 / 269.7); CHECK pass x3, max_rel_err 0.0, state_ok. Dead ends noted: set_mla_kv_concat_q (3.5 us standalone), qkv_a GEMM (cuBLAS 12.6 us best, DG/transposed slower).

## iter_00
### analysis.md

# iter_00 -- baseline (starting tree, no edit)
Plain: B512@8k 888.2 us | B256@8k 619.3 | B16@64k mix 269.7. Golden -> /tmp/golden.pt.

## VibeSim
simulate(prediction=.:before) -> p_6b53ccf740d742fd8779d087584820d7 (fixed before-state, `.:k3_mla_b512`; the
id itself does not resolve in analyze -- queried via the `.:k3_mla_b512` handle). vs_*.json in this folder.
run_summary: optimality 0.24; batching 48%, imbalance 19%, hardware gap 9%.
optimality (3-case sums, us): mxfp4_fused_moe r0 776.9 / r5 46.0 / r6 245.7 (r0/r5 16.9, nec 0.32 -> kernel-efficiency);
mla_decode_attention 686.4 / 502.7 / 1000 (r6>r0 -> bandwidth-bound, only scheduling left);
merged_front 188.1/116.4/83.8 (1.62); shared_down 1.72; latent_up 1.86; fused_qkv_a 2.67; output_gate 2.87 (nec 0.24).

## Measured (nsys graph nodes, B512, span 887.7 us)
front 44 | g_proj (nvjet 96x64, 13.1 us) SERIAL right before fmha | fmha 366 + LSE/merge 3 | post-attn 33 |
merged_front 73.1 | routing chain 18 | MoE GEMM1 213.9 + GEMM2 97.9 + finalize 8.3 | norm+latent_up 22+add3 8.
## What each top node allows
- MoE: swept ALL 1736 valid (tile_N, config) tactics on the real B512 inputs (iter_01/sweep512.log): best 319.5 us,
  current (16,377) within 0.03 us -> tactic space exhausted (confirms/extends mla_b512_claude_1 iter_01).
- fmha: 2.42 GB in 366 us = 6.6 TB/s; B256 second wave (108 CTAs) takes as long as the first (148) -> per-CTA latency bound
  at 1 CTA/SM; tail split already in tree. Closed cubin, no knob left.
- Dense GEMMs (scratch/gemm_bench.py): cuBLAS = DeepGEMM = FlashInfer cudnn at m=512 (0.9-1.3 PF), all bit-exact;
  chip peak ~1.6 PF (8k^3); merged_front already at peak in-layer. No faster kernel.
=> remaining lever: dataflow/overlap. g_proj (output_gate node, nec 0.24) is independent of attention and sits serially
on the critical path -> target.

### result.json

```
==============================================================================
kimi_single_layer_decode v3 (K3/MLA)  experts=112 EP=8 local_topk=2 hidden_scale=1.0 layer_idx=4 iters=40 warmup=10 split=False moe_backend=flashinfer_mxfp4 attention_backend=cutedsl_mla page_size=64 attn_heads=12 cuda_graph=True kv_cache_dtype=fp8_e4m3 mamba_ssm_dtype=float32 bf16_gemm_init=True flashinfer_autotune=False
torch=2.13.0+cu130  dev=NVIDIA B200
sglang=0.5.20
==============================================================================
[cfg] wrote /tmp/k3_cfg/config.json  experts=112 (EP=8) top_k=16 shared=2 layers=8  moe_backend=flashinfer_mxfp4 hidden_act=situ  attn_type=mla target_layer=4 mla_heads=12 kda_heads=96 kda_layers=[1, 2, 3, 4, 6, 7, 8] full_attn=[5]
[boot] bf16 GEMM backend initialized as in the production scheduler: Bf16GemmBackend.CUTEDSL
[moe] flashinfer_mxfp4_moe_precision=default
[moe] runner_backend=MoeRunnerBackend.FLASHINFER_MXFP4 is_flashinfer_mxfp4=True is_deep_gemm=False
[boot] tp=1 attn_tp=1 world=1  is_kda_layer(4)=False  attention_arch=1 dtype=torch.bfloat16
[layer] mxfp4 quant_config=Mxfp4Config quant_format=None
[layer] UnquantizedLinearMethod from sglang.srt.layers.quantization.unquant
[layer] patched Mxfp4Config.get_quant_method: LinearBase/RadixAttention -> bf16 unquantized fallback
[layer] KimiK3DecoderLayer(layer_idx=4): self_attn=KimiK3MLAAttention (KDA=False MLA=True)  mlp=KimiK3MoE (MoE=True)
[layer] float params=0.228B (fp32 kept: 0.00M) norm gains ~1: 7
[layer] random-init 0.228B float params -> cuda/bf16/eval (MLA: absorbed kv_b_proj -> w_kc(12, 128, 512) w_vc(12, 512, 128))
[mxfp4] experts=FusedMoE tensors:
[mxfp4]   w13_weight                       (112, 6144, 1792)      torch.uint8 <- e2m1 randint
[mxfp4]   w13_weight_scale                 (112, 6144, 112)       torch.uint8 <- ue8m0 fill 127
[mxfp4]   w13_weight_bias                  (112, 6144)            torch.bfloat16 <- bias normal
[mxfp4]   w2_weight                        (112, 3584, 1536)      torch.uint8 <- e2m1 randint
[mxfp4]   w2_weight_scale                  (112, 3584, 96)        torch.uint8 <- ue8m0 fill 127
[mxfp4]   w2_weight_bias                   (112, 3584)            torch.bfloat16 <- bias normal
[mxfp4] quant_method=Mxfp4MoEMethod use_flashinfer=True use_marlin=False _fi_kernel=trtllm_sm100 precision=default
[mxfp4] quant_method.process_weights_after_loading(experts) OK
[layer] post_load_merge: front_w=(15984, 7168) fused_front_eligible=True bias_dtype
[... truncated]
```

## iter_01
### hypothesis.md

# iter_01 hypothesis
g_proj(hidden_states) only depends on the layer input, yet the seeded tree issues it on the main stream right before the
MLA decode kernel (13.1 us serial). Forking the alt stream before the decode kernel and enqueuing g_proj AFTER the kernel
launch lets the block scheduler place its CTAs on SMs freed by the decode kernel's last (half) wave; the o_proj wrap
already joins the alt stream before the gate multiply. Same kernel, same inputs -> bit-exact output and state.
Prior trials: mla_24 / TECHNIQUES A4 (alt-stream overlap, shared experts) -- same mechanism on another node;
mla_b512_2 "low-priority overlap (-1us)" (Codex) -- ruled in a different placement (issue after the attention launch, not
at layer start: the 'front' variant here reproduces that ~-2 us result). Seeded tree's `_proto_hook`/EXP code replaced
by a named pre/post decode-kernel hook.

### analysis.md

# iter_01 -- output-gate GEMM moved into the fmha tail
VibeSim node: unified.mla.attention.output_gate (r0 31.9 / r5 11.1 / r6 7.7 us, necessary_share 0.24, 3-case sum;
prediction .:k3_mla_b512 = p_6b53ccf740d742fd8779d087584820d7). Low necessary share + independent of attention
=> scheduling, not a kernel swap.
Variants (plain flags, replay): serial (seed) 888.2/618.9/269.7 | front (alt stream at layer start) 886.2/614.8/264.5 |
post (fork before fmha, launch after) 880.0/613.7/265.5. All CHECK pass, rel 0.0.
nsys (nsys_post.sqlite): g_proj now 398.3-417.5 us, i.e. inside the fmha's last CTAs + LSE merge; the gate-mul
kernel starts 0.2 us after it. Span 887.7 -> 876.2 us.
Remaining critical path: fmha 364, MoE 310, merged_front 73, small kernels.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 512, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 879.9359798431396, "latency_mode": "graph", "us_step": 1940.9600496292114, "us_step_graph": 879.9359798431396, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [512, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 256, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 613.6320233345032, "latency_mode": "graph", "us_step": 1870.2399730682373, "us_step_graph": 613.6320233345032, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [256, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "mixed": true, "tag": "mix", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 265.56798815727234, "latency_mode": "graph", "us_step": 1668.8319444656372, "us_step_graph": 265.56798815727234, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "atten
[... truncated]
```

(diff.patch: 102 lines, files: srt/layers/attention/trtllm_mla_backend.py, srt/models/kimi_k3.py)

## iter_02
### hypothesis.md

# iter_02 hypothesis
Builds on iter_01 (side-stream work placed where SMs idle) and on iter_00's finding that GEMM1 tolerates
concurrent work while GEMM2 does not. Re-tests mla_b512_claude_1 iter_02 (front split regressed there) with a
different ordering: split the merged front into routed [router|routed_input] (25.9 us) + shared gate_up; run the
routed part first on the main stream so routing/quant/GEMM1 start ~47 us earlier, and run gate_up+situ+shared_down
on the alt stream overlapping GEMM1. Variants: "post" (gate_up enqueued after the MoE) and "pre" (gate_up forked
before the routed GEMM). Expected: -20..-40 us at B512 if shared work hides inside GEMM1.
Outcome: ruled out (+16 us); see analysis.md. Confirms mla_b512_claude_1 iter_02's failure mode.

### analysis.md

# iter_02 -- merged front GEMM split (routed part first, shared gate_up on alt stream)  [RULED OUT]
VibeSim (prediction .:k3_mla_b512 = p_6b53ccf740d742fd8779d087584820d7, vs_optimality.json here), 3-case sums (us):
merged_front r0 188.1 / r5 116.4 / r6 83.8 (r0/r5 1.62, nec 0.45); shared_down 73.8/43.0/30.7 (1.72);
shared_gate_up_activation 8.6/3.6/0 ; mxfp4_fused_moe 776.9/46.0/245.7. merged_front is near its hardware
limit in-layer (73.1 us, ~1.6 PF), so the only lever is to shorten the critical path by starting routing
before the shared gate_up columns finish (routed GEMM alone = 25.9 us).

Result (plain flags, replay, CHECK pass x3, rel 0.0 -- split is bit-exact at m>=128):
  post  895.4 / 609.5 / 271.6   pre 895.0 / 620.9 / 271.6   vs iter_01 879.0 / 613.8 / 265.5  -> B512 +16 us.
nsys (nsys_split.sqlite, post):
  - gate_up nvjet 192x128 (256 CTAs, 71.9 us, s144) occupies every SM from 467.9 us; the 8-CTA x1024-thread
    routingIndicesClusterKernel cannot be co-resident and starts only at 535.5 (baseline 528.7).
  - GEMM1 (48x169 CTAs) then fills all SMs, so situ+shared_down cannot run until GEMM1's tail: shared_down
    (39.8 us) lands on GEMM2, which slows 97.6 -> 123.3 us (GEMM2 is overlap-intolerant, see iter_00).
  - GEMM1 itself is 213 -> 193.5 us without shared_down at its head, i.e. in the baseline shared_down costs
    GEMM1 ~20 us, about its standalone cost -- there is no idle hardware left to hide it in.
Conclusion: total work is bandwidth/SM-bound around the MoE; moving the shared branch only shifts where it is
paid, and the routing cluster kernel's co-residency makes it worse. Reverted to iter_01 (replay_reverted.log:
879.97 / 613.57 / 265.50, CHECK pass x3, rel 0.0).

### result.json

```
{"iter": "02", "status": "ruled_out_reverted",
 "variant_post": {"latency_us": [895.36, 609.50, 271.58], "check": "pass x3, rel 0.0"},
 "variant_pre":  {"latency_us": [895.01, 620.93, 271.65], "check": "pass x3, rel 0.0"},
 "tree_after_revert": {"latency_us": [879.97, 613.57, 265.50], "check": "pass x3, rel 0.0"}}
```

(diff.patch: 135 lines, files: sglang/srt/models/kimi_k3.py)

## iter_03
### hypothesis.md

# iter_03 hypothesis
Builds on iter_01 (tree state) and iter_00's conclusion that the big nodes (fmha, MoE tactic, dense GEMM backends)
are exhausted, leaving small-kernel fusion/scheduling. iter_02 (front split) ruled out moving MoE-side work.
The two torch bf16 adds ahead of the attn-res kernels (VibeSim input/post_attention_layernorm, necessary share 0)
are a pure extra memory pass + launch; folding the add into the aggregation kernel that already streams the prefix
row should save ~3-4 us per layer without changing a bit (the fp32 add + single RNE rounding equals torch's).
The existing run_pull_rs prologue pattern was the first attempt (rejected, see analysis.md); the in-ring variant
keeps the producer/consumer pipeline intact.

### analysis.md

# iter_03 -- pending residual add fused into the attn-res TMA kernel
VibeSim (prediction .:k3_mla_b512 = p_6b53ccf740d742fd8779d087584820d7; vs_optimality.json here), 3-case sums (us):
unified.mla.attention.input_layernorm 12.9 / r5 5.6 / r6 0.0 and post_attention_layernorm 12.9 / 5.6 / 0.0
(r0/r5 2.29, necessary_share 0.00) -> the attn-res aggregation points (score/softmax/mix/RMSNorm) carry memory
traffic that is not necessary in the fused-scope view: the materialized `prefix = prefix_a + prefix_b`.
nsys (iter_01 tree, B512): vectorized_elementwise_kernel grid 3584 (bf16 add over [512,7168], 3.4-4.2 us) runs
right before each attn_res_fused_tma_kernel (2x per layer) -- _aggregate_fused_add's torch add.

Change: new AttnResFusedTmaKernel::run_add (kFuseAdd trait): the producer bulk-copies the addend row into one extra
smem row of the prefix chunk's slot (same mbarrier); consumers add it onto the prefix row in registers (fp32 add,
one RNE rounding == torch bf16 add) before the score pass and store the sum to prefix_out. Bank prefetch before
the PDL wait is untouched. Unit test (scratch/attn_res_add_test.py): bit-exact vs torch add + old kernel for
T in {1,16,100,256,512,1000} x nvb {1,2,3,4,5,8} x write_prefix.
First version (per-CTA add prologue before Trait::forward, like run_pull_rs) was bit-exact but slower:
888.0/615.7/265.6 -- the kernel grew to 17.4 us (latency-bound prologue + lost PDL bank prefetch). Rejected.
Result (plain, replay): 879.97/613.57/265.50 -> 876.96/609.63/261.44 (rerun 876.90/608.58/261.44); CHECK pass x3, rel 0.0.

### result.json

```
{"iter": "03", "status": "accepted",
 "before_iter01_tree": {"latency_us": [879.97, 613.57, 265.50]},
 "after": {"latency_us": [876.96, 609.63, 261.44], "rerun": [876.90, 608.58, 261.44],
           "check": "pass x3, max_rel_err 0.0, state_ok"},
 "rejected_first_version_prologue_add": {"latency_us": [888.00, 615.74, 265.60], "check": "pass x3, rel 0.0"}}
```

(diff.patch: 428 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py, /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py, /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)
