# MLA_VERIFY_CLAUDE trial 3: agent iterations

## agent log.md

# Optimization log (latency_us: 64,8192 / 128,8192 / 16,65536)
- iter_00: baseline 483.7 / 627.0 / 400.8 (VibeSim p_c7687e58...)
- iter_01: fuse residual add into attn_res TMA aggregate -> A/B 484.3->480.6 / 627.1->626.0 / 400.9->396.6; CHECK pass x3 (rel_err 0); KEPT
- iter_02: SM carveout on shared-expert down GEMM (unblocks MoE routing kernel) -> A/B 480.6->474.5 / 626.1->626.0 / 395.9->396.6 (A bimodal, noise); CHECK pass x3 (rel_err 0); KEPT. Ruled out: fork shared after topk (482.5)

## iter_00
### hypothesis.md

# iter_00 -- baseline (no change)
Starting tree already contains accepted changes (satfinite fp8 in set_mla_kv_concat_q, shared-expert
alt-stream overlap, forked output gate, _gate_bs_limit=512). Ruled out from history/measurement:
merging g_proj into qkv_a (+7.4us GEMM at M=256), dense GEMM backend swaps, trtllm-gen for verify,
uniform split_kv>1 (74.5 vs 63us MLA).

### analysis.md

# iter_00 -- baseline analysis
VibeSim prediction_id: p_c7687e585abe4b5fbacdc89caf0c9761 (handle `.:k3_mla`, kimi_k3_sglang, B200, 4 cases).
Plain baseline latency_us: 64,8192=483.7  128,8192=627.0  16,65536=400.8.

Critical path B=64 (~490us per-kernel profile): MoE fc1 179us + fc2 92us (HBM roofline ~7TB/s),
MLA decode 62.6us (302MB fp8 KV, ~4.9TB/s; 64 two-CTA clusters on 74 cluster slots -> 20 SMs idle),
merged_front 47us; g_proj on alt stream overlaps MLA.

Optimality ladder (vs_opt.json), selected nodes:
- unified.mla.attention.input_layernorm (residual_rms_norm): r0=12.7 r5=1.1 necessary_share=0.00
- unified.mla.attention.post_attention_layernorm (residual_rms_norm): r0=12.7 r5=1.1 necessary_share=0.00
MoE GEMMs are at r5 (necessary work) -> kernel swaps won't move them.

### result.json

```
{"iter":0,"change":"none","latency_us":{"64,8192,vk3":483.7,"128,8192,vk3":627.0,"16,65536,vk3":400.8},"check":"golden captured from original"}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 -- fuse residual add into attn_res TMA aggregate
Hypothesis: have the TMA producer bulk-load the residual row alongside the prefix row and add in
the consumer (fp32 add, one bf16 rounding -- same as torch bf16 add), writing the summed prefix out.
Removes one launch + one [T,7168] bf16 round-trip per norm site (2 per layer).
Built on: mla_b512_claude_3 (accepted the same add fusion), mla_prefill_claude_1.
Ruled-out risk: kda_b512_claude_3 failed correctness with a fused add; this version is verified
bit-exact vs unfused in micro/attn_res_add_test.py (T in {1,7,64,256,512} x nvb in {1,2,3,4,8}).

### analysis.md

# iter_01 -- analysis
VibeSim prediction_id: p_c7687e585abe4b5fbacdc89caf0c9761 (handle `.:k3_mla`).
Node: unified.mla.attention.input_layernorm / post_attention_layernorm (residual_rms_norm).
Ladder: r0=12.7us r5=1.1us necessary_share=0.00 -> the separate residual add + aggregate is
unnecessary work: prefix_sum + residual is materialised by one kernel then re-read by the attn_res
TMA aggregate kernel.
Source: python/sglang/srt/layers/attn_residual.py::_aggregate_fused_add -> attn_res_fused_tma
(kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh).

### result.json

```
{"iter":1,"change":"fused residual add into attn_res TMA aggregate",
 "ab_interleaved_3reps":{"64,8192,vk3":{"A":484.3,"A_sd":0.58,"B":480.6,"B_sd":0.08},
  "128,8192,vk3":{"A":627.1,"A_sd":0.04,"B":626.0,"B_sd":0.04},
  "16,65536,vk3":{"A":400.9,"A_sd":0.21,"B":396.6,"B_sd":0.02}},
 "check":{"64,8192,vk3":{"pass":true,"max_rel_err":0.0},"128,8192,vk3":{"pass":true,"max_rel_err":0.0},"16,65536,vk3":{"pass":true,"max_rel_err":0.0}},
 "smoke":"LAYER_SMOKE_OK","kept":true}
```

(diff.patch: 333 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py, /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py)

## iter_02
### hypothesis.md

# iter_02 -- SM carveout on the shared-expert down GEMM
Hypothesis: cap the shared down GEMM's SM count (cuBLASLt SM_COUNT_TARGET via
torch._C._set_sm_carveout_experimental, captured into the graph) so the routing cluster kernel and
fc1 co-schedule with it; the shared GEMM's HBM traffic then hides under fc1's memory-bound waves.
Built on: iter_01 tree; starting-tree shared-expert alt-stream overlap (kept).
Ruled out in this iteration: variant A -- forking the shared experts after topk instead of after the
front (482.5 vs 480.6us): the down GEMM still wins the SM race against routing (tl_a timeline).
Ordering between routing and fc1 cannot be hooked (both inside trtllm_fp4_block_scale_moe).
Sweep (single plain runs, B64/B128/B16): cv0 480.6/626.0/396.4, cv16 480.6/626.0/396.5,
cv24 476.4/626.0/396.6, cv32 475.5/625.9/395.5, cv40 475.2/628.1/395.6, cv48 474.4/628.0/395.7,
cv56 474.5/631.1/396.4, cv64 474.4/630.1/396.6, cv100 475.5/628.1/396.7.
B128 regresses at >=40 -> carveout 48 for <=256 tokens, 32 above.

### analysis.md

# iter_02 -- analysis
VibeSim prediction_id: p_c7687e585abe4b5fbacdc89caf0c9761 (handle `.:k3_mla`).
Node: unified.mla.moe.shared_down: r0=65.4us r5=44.5us r6=11.0us necessary_share=0.17
(headroom r0-r5=20.8us; excess over necessary r5-r6=33.5us).
Graph-node nsys timeline at B=64 (iter_02/tl_cuda_gpu_trace.csv, diagnostic): the shared down GEMM
(nvjet 112x128 2cta, 84 clusters > 74 slots, 25.5us) becomes ready right after the merged front and
fills every SM; the routed MoE's cluster-launched routingIndicesClusterKernel (input ready at +674.5)
cannot schedule until it drains (+690.3) -> ~16us stall directly on the critical path
(routing -> fc1 173us -> fc2 89us -> finalize). The shared GEMM itself is off-critical.
Source: srt/models/kimi_k3.py::KimiK3MoE._forward_shared (alt_stream branch of _forward_fused).

### result.json

```
{"iter":2,"change":"cuBLAS SM carveout (48 if tokens<=256 else 32) on shared-expert down GEMM",
 "ab_interleaved_3reps_vs_iter01":{"64,8192,vk3":{"A":480.6,"A_sd":0.05,"B":474.5,"B_sd":0.04},
  "128,8192,vk3":{"A":626.1,"A_sd":0.05,"B":626.0,"B_sd":0.02},
  "16,65536,vk3":{"A":395.9,"A_sd":0.49,"B":396.6,"B_sd":0.08}},
 "check":{"64,8192,vk3":{"pass":true,"max_rel_err":0.0},"128,8192,vk3":{"pass":true,"max_rel_err":0.0},"16,65536,vk3":{"pass":true,"max_rel_err":0.0}},
 "smoke":"LAYER_SMOKE_OK","kept":true}
```

(diff.patch: 371 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py, /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py, /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)
