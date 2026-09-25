# KDA trial 8: agent iterations

## agent log.md

iter_00 baseline: VibeSim selected mxfp4_fused_moe (R0 412.401 us, R5 9.454 us, 43.62x); plain graph 402.976/263.648/138.752 us for B=128/32/1; golden captured.
iter_01 fused MXFP4 router: CHECK passed with state_ok on all points, but replay 402.976->409.120 / 263.648->265.728 / 138.752->138.816 us; rejected after VibeSim/profile confirmed extra conversion and slower routing-index kernel.
iter_02 KDA TMA threshold 512->1024: exact CHECK/state pass; replay 402.976->403.072 / 263.648->263.648 / 138.752->138.784 us; neutral, rejected.

## iter_00
### hypothesis.md

# Baseline hypothesis for next iteration

Target the MXFP4 MoE router boundary. The current K3 fused-front path already computes exact fp32 router logits, then materializes a separate standard top-k result before calling the routed MXFP4 kernel. FlashInfer's SM100 `trtllm_fp4_block_scale_moe` accepts the same logits and correction bias and implements sigmoid, bias addition, top-k, and renormalization inside the expert launch. Passing a bypassed top-k object should remove the standalone router launch while preserving the routing contract; the replay CHECK across all points will decide whether its internal rounding changes are within the required tolerance.

### analysis.md

# Baseline analysis

VibeSim simulation: `p_38ccb83a6c4e44298f45b2251bd4cf54` (`vibesim:k3_kda`, fixed before-state Kimi-K3 KDA prediction on NVIDIA B200).

The run summary predicts 0.782953 ms total kernel time, with the gap split into 61.14% hardware gap, 29.01% batching, and 7.68% hardware-necessary work. Operator analysis ranks `unified.kda.moe.mxfp4_fused_moe` first at 0.412401 ms / 52.67%, followed by the merged front at 0.120183 ms / 15.35% and KDA `qkvbfg_a_proj` at 0.051857 ms / 6.62%.

`optimality?scope=iter` ranks `unified.kda.moe.mxfp4_fused_moe` first: R0=412.401 us, R5=9.454 us, R0/R5=43.62x, with R6/R0 unavailable (`null`) and no cached alternative in the kernel drill-down. The next actionable leaves are `qkvbfg_a_proj` (R0=51.857 us, R5=33.566 us) and `kda_recurrent_decode` (R0=42.495 us, R5=8.161 us). The ladder indicates the MXFP4 node has the largest implementation headroom, while the missing R6 value means VibeSim cannot quantify necessary-share savings for that external fused kernel.

The B=128 profiler confirms the expected MXFP4 path: the routed expert FC1 and FC2 BMMs take 125.199 us and 66.413 us, and the step also launches `_router_triton_kernel` (3.737 us), the FlashInfer routing-index kernel (3.622 us), and the finalize kernel (7.245 us). There are 26 launches and 423.424 us of kernel time in this diagnostic profile. B=32 and B=1 use the same routed BMM family with shape-specialized variants.

Plain fixed-command graph references are 402.976 us (B=128), 263.648 us (B=32), and 138.752 us (B=1). The profile command is diagnostic only; its graph values were 370.176/246.240/139.552 us.

### result.json

```
{
  "mode": "baseline",
  "timing_reference": [
    {"B": 128, "seq_len": 8192, "latency_us": 402.97600626945496},
    {"B": 32, "seq_len": 8192, "latency_us": 263.64800333976746},
    {"B": 1, "seq_len": 8192, "latency_us": 138.75199854373932}
  ],
  "golden": [
    "/tmp/golden_B128_L8192.pt",
    "/tmp/golden_B32_L8192.pt",
    "/tmp/golden_B1_L8192.pt"
  ],
  "checks": "golden captured; no post-edit CHECK yet"
}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# Hypothesis

The fixed K3 shape satisfies the guarded fused-front MXFP4 conditions: local experts use the SM100 TRT-LLM runner, precision is `default`, activation is SiTU, and routing is one sigmoid group with top-k=2. Returning `BypassedTopKOutput` lets `trtllm_fp4_block_scale_moe` perform the same sigmoid, correction-bias ranking, and renormalization in its fused routing path, removing the standalone `_router_triton_kernel`. The explicit path remains selected for every other backend or shape, and the post-step state is owned by KDA, so the replay CHECK is the acceptance criterion for numerical equivalence.

### analysis.md

# Iteration 01 analysis

VibeSim prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`. The service reports a fixed before-state prediction, so the operator and optimality values are unchanged after this source edit. The selected node remains `unified.kda.moe.mxfp4_fused_moe`: R0=412.401 us, R5=9.454 us, R0/R5=43.62x, R6/R0=null, and `has_cached_alternative=false` from the `mxfp4_fused_moe` kernel drill-down. The secondary `unified.kda.attention.kda_recurrent_decode` node is R0=42.495 us, R5=8.161 us, R0/R5=5.21x.

The post-edit profiler confirms the fused route removed `_router_triton_kernel`, but B=128 replaced it with a BF16 copy (3.789 us) and used a slower internal `SigmoidBiasPreprocess/ScaledSumNormalizePostprocess` routing-index kernel (5.296 us). The two expert BMMs were 126.947 us and 66.717 us versus 125.199 us and 66.413 us before. Diagnostic graph values were 376.320/248.416/139.808 us with 26/27/25 launches.

The fused-router CHECK passed all points and state checks, but the uncontaminated replay values were 409.120/265.728/138.816 us for B=128/32/1. This is not an accepted speedup. The next target is the KDA fused decode launch configuration, which VibeSim identifies as the largest non-MoE implementation-headroom node.

### result.json

```
{
  "mode": "replay",
  "timing_reference": [
    {"B": 128, "seq_len": 8192, "latency_us": 409.11999344825745},
    {"B": 32, "seq_len": 8192, "latency_us": 265.7279968261719},
    {"B": 1, "seq_len": 8192, "latency_us": 138.8159990310669}
  ],
  "checks": [
    {"B": 128, "max_rel_err": 0.008403360779605998, "state_ok": true, "pass": true},
    {"B": 32, "max_rel_err": 0.009569377257846718, "state_ok": true, "pass": true},
    {"B": 1, "max_rel_err": 0.005813952947539261, "state_ok": true, "pass": true}
  ],
  "accepted": false
}
```

(diff.patch: 300 lines, files: /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)

## iter_02
### hypothesis.md

# Hypothesis

VibeSim identifies `unified.kda.attention.kda_recurrent_decode` as the next implementation-headroom node after the rejected MXFP4 routing change (R0=42.495 us, R5=8.161 us, R0/R5=5.21x). The fused KDA source selects 3 TMA state-staging stages when `B * H >= 512`, so B=128/H=12 uses 3 stages while B=32/H=12 and B=1/H=12 use 4. The 4-stage variant is already compiled and was documented as the small-grid choice; using it at B=128 should reduce state-load synchronization without changing arithmetic, memory addresses, state writes, or output.

### analysis.md

# Iteration 02 analysis

VibeSim prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`; the service documents this as a fixed before-state prediction, so the modeled ladder remains unchanged. `unified.kda.attention.kda_recurrent_decode` remains the selected secondary node with R0=42.495 us, R5=8.161 us, R0/R5=5.21x, R6/R0=null, and no cached alternative. The kernel drill-down confirms the only profiled backend is `sglang_triton` for 12 heads, 128x128 bf16 state, and lower_bound=-5.

The source experiment changed the fused decode stage threshold from `B*H >= 512` to `B*H >= 1024`, selecting the precompiled 4-stage TMA variant for B=128/H=12. The diagnostic profile still shows the same 26/27/25 launch counts and graph values 370.176/246.240/139.680 us. The required plain replay values were 403.072/263.648/138.784 us, effectively unchanged from 402.976/263.648/138.752 us. This is not an accepted speedup; the exact CHECKs and post-step states pass at all points.

### result.json

```
{
  "mode": "replay",
  "timing_reference": [
    {"B": 128, "seq_len": 8192, "latency_us": 403.0719995498657},
    {"B": 32, "seq_len": 8192, "latency_us": 263.64800333976746},
    {"B": 1, "seq_len": 8192, "latency_us": 138.7840062379837}
  ],
  "checks": [
    {"B": 128, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 32, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 1, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "accepted": false
}
```

(diff.patch: 18 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh)

## iter_03
### hypothesis.md

Iteration 03 hypothesis

The initial VibeSim prediction and profiler identify the serialized K3 MoE path as the main opportunity. The `unified.kda.moe.shared_down` node measures 50.179 us in the initial profile and has 16.613 us of roofline headroom. In the fixed one-GPU run, TP=1 makes `k3_ar_fusion.enabled()` false, so `_forward_fused` computes the shared down projection on the current stream before launching the routed MXFP4 MoE.

The change launches that same `_forward_shared` call on the already-created alternate stream after it waits for the fused front, while the current stream runs `_forward_routed`. The current stream waits for the shared stream before the existing collective/tail. The operations, tensors, weights, and arithmetic are unchanged; only independent launch ordering changes. The join is before any consumer of `shared_output`, so output and post-step state should be unchanged.

### analysis.md

Iteration 03 pre-edit analysis

VibeSim workspace-info was queried before simulation. The fixed-state prediction is `p_38ccb83a6c4e44298f45b2251bd4cf54` from `simulate?prediction=.:before`; the service reports the Kimi-K3 KDA B200 recipe and uses the prediction id as a fixed before-state.

The operator analysis measured 0.782953 ms of leaf kernel time. The largest node was `unified.kda.moe.mxfp4_fused_moe` at 0.412401 ms (52.67%), followed by `unified.kda.moe.merged_front` at 0.120183 ms (15.35%). The iteration optimality ladder reported:

- `mxfp4_fused_moe`: R0=412.401 us, R5=9.454 us, R0/R5=43.62x; no R6/necessary-share estimate.
- `merged_front`: R0=120.183 us, R5=91.108 us, R0/R5=1.32x.
- `shared_down`: R0=50.179 us, R5=33.566 us, R0/R5=1.49x.

The profiler confirms the routed MXFP4 GEMM pair (125.199 + 66.413 us) and the shared/front CUDA GEMMs are actually launched. Because the shared down projection is independent of routing after the fused front and is currently serialized on TP1, stream overlap is the selected code change. The fused-MoE kernel itself has no cached alternative in VibeSim, so replacing it without a proven equivalent backend is not justified.

Post-edit VibeSim was fetched with `simulate?prediction=.:after` and returned the same fixed prediction id. Operator/run-summary/iteration analysis was then rerun. The model still attributes 0.412401 ms and 52.67% to `mxfp4_fused_moe`; the shared-down node remains 50.179 us with R5=33.566 us and `has_cached_alternative=true`. The kernel drill-down confirms the same `sglang_trtllm_mxfp4` routed backend and the same no-alternative MXFP4 shape, so the optimization is overlap/scheduling rather than a backend substitution.

The diagnostic profile after the edit reports 26/27/25 launches and graph latencies 360.000/238.080/131.584 us for B=128/32/1. The required plain replay measured 392.736/253.440/130.592 us against the baseline 402.976/263.648/138.752 us. All three `CHECK` records have `max_rel_err=0`, `state_ok=true`, and `pass=true`. The B=128 reduction is 2.54%, below the task's 5% acceptance threshold, so this iteration is retained as a valid partial optimization and another iteration is required.

### result.json

```
{
  "iteration": 3,
  "before_plain_graph_us": {
    "128,8192": 402.97600626945496,
    "32,8192": 263.64800333976746,
    "1,8192": 138.75199854373932
  },
  "after_plain_graph_us": {
    "128,8192": 392.7359879016876,
    "32,8192": 253.4399926662445,
    "1,8192": 130.5920034646988
  },
  "checks": {
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "32,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "1,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "accepted_for_final": false,
  "reason": "Valid overlap and universal speedup, but B=128 is only 2.54% faster and below the 5% judge threshold."
}
```

(diff.patch: 23 lines, files: /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)

## iter_04
### hypothesis.md

Iteration 04 hypothesis

The post-iteration-03 VibeSim ladder leaves `unified.kda.attention.kda_recurrent_decode` as a 42.496 us node with an 8.161 us R5 hardware limit and a 5.21x R0/R5 ratio. The KDA fused decode source chooses three TMA state-staging stages when `B * heads >= 512`; for the scored B=128, H=12 this selects the 3-stage kernel. The source comment says the 3-stage variant is intended for larger grids, but this B200 workload has only 128*12 CTAs and the decode state is large enough that the 4-stage variant may hide the state loads better.

This iteration changes only the dispatch threshold so B=128 uses the existing 4-stage KDA kernel; B=32 and B=1 already use four stages and are unchanged. The kernel implementation, arithmetic, state writes, and output path are unchanged, so numerical output and post-step state must remain identical.

### analysis.md

Iteration 04 pre-edit analysis

VibeSim was queried after the iteration-03 profile with workspace-info already established. `simulate?prediction=.:after` returned the fixed prediction `p_38ccb83a6c4e44298f45b2251bd4cf54`; operator, run-summary, iteration, optimality, and kernel analyses were run before selecting this experiment.

The top VibeSim node remains `unified.kda.moe.mxfp4_fused_moe` (R0=412.401 us, R5=9.454 us, R0/R5=43.62x, no cached alternative). The next attention candidate is `unified.kda.attention.kda_recurrent_decode` (R0=42.496 us, R5=8.161 us, R0/R5=5.21x). Its profiler leaf is the already-launched fused KDA decode path, so the only narrow source-level variable available without changing numerics is the compiled TMA stage count. This iteration tests the existing 4-stage variant at B=128.

Post-edit profiling still reports 26/27/25 launches and graph latencies 359.904/237.888/131.552 us in the diagnostic run. The plain replay was 392.736/253.440/130.592 us, identical to the iteration-03 candidate within the timer resolution, and all three checks passed with zero error and `state_ok=true`. VibeSim again reports the same KDA and MXFP4 ladders and the kernel drill-down confirms the same fused KDA leaf. The threshold change is neutral and is being reverted.

### result.json

```
{
  "iteration": 4,
  "before_plain_graph_us": {
    "128,8192": 392.7359879016876,
    "32,8192": 253.4399926662445,
    "1,8192": 130.5920034646988
  },
  "after_plain_graph_us": {
    "128,8192": 392.7359879016876,
    "32,8192": 253.4399926662445,
    "1,8192": 130.5920034646988
  },
  "checks": {
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "32,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "1,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "accepted_for_final": false,
  "reason": "4-stage KDA dispatch was neutral; source restored to the 512 threshold."
}
```

(diff.patch: 11 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh)

## iter_05
### hypothesis.md

Iteration 05 hypothesis

VibeSim identifies `unified.kda.attention.kda_recurrent_decode` as the largest attention leaf after the MoE nodes (R0=42.496 us, R5=8.161 us, R0/R5=5.21x). The per-kernel profiler maps this node to `fused_sigmoid_gating_delta_rule_update_kernel`, which is the actual decode path for the fixed driver's Triton KDA backend and BF16-state workload. Its launcher uses one warp per program even though each program reduces a 128-wide K dimension and produces a 32-wide V tile.

This iteration changes only Triton's launch geometry from one warp to two. The kernel body, floating-point operations, safe-gate formula, state load/store addresses, and output dtype are unchanged; only cooperative execution geometry changes. The golden replay must verify both output and recurrent state.

### analysis.md

Iteration 05 pre-edit analysis

VibeSim workspace-info was called first in the overall loop. After the iteration-04 profile, `simulate?prediction=.:after` returned fixed prediction `p_38ccb83a6c4e44298f45b2251bd4cf54`; operator, run-summary, iteration, optimality, and kernel analyses were run before this selection.

The fixed prediction still reports `mxfp4_fused_moe` at R0=412.401 us/R5=9.454 us and no cached alternative. For the next actionable attention leaf, `kda_recurrent_decode` is R0=42.496 us/R5=8.161 us, 5.21x over its roofline limit. The profiler confirms the launched Triton kernel is `fused_sigmoid_gating_delta_rule_update_kernel` (29.4 us in the baseline B=128 table), so this iteration targets its launch configuration rather than the unused JIT-CUDA fused-decode source.

Post-edit profiling reports the same launch counts and graph latency within timer noise: 360.032/237.728/131.520 us for B=128/32/1. The plain replay was 392.704/251.424/130.528 us; B=128 did not improve over the iteration-03 candidate. The two-warp kernel passed all checks (`max_rel_err=0.008403` at B=128, `state_ok=true`, all `pass=true`), but the arithmetic reduction order changed the output within tolerance without yielding speed. The source is being restored to one warp.

### result.json

```
{
  "iteration": 5,
  "before_plain_graph_us": {
    "128,8192": 392.7359879016876,
    "32,8192": 253.4399926662445,
    "1,8192": 130.5920034646988
  },
  "after_plain_graph_us": {
    "128,8192": 392.7040100097656,
    "32,8192": 251.42401456832886,
    "1,8192": 130.52800297737122
  },
  "checks": {
    "128,8192": {"max_rel_err": 0.008403360779605998, "state_ok": true, "pass": true},
    "32,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "1,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "accepted_for_final": false,
  "reason": "Two Triton warps changed reduction order but did not improve the scored latency."
}
```

(diff.patch: 0 lines, files: )

## iter_06
### hypothesis.md

Iteration 06 hypothesis

VibeSim's `unified.kda.moe.shared_down` leaf is 50.179 us with R5=33.566 us and reports cached alternatives. The profiler maps the current shared projection to an `nvjet` BF16 GEMM. The repo already contains a PDL-enabled CuTe TGV BF16 GEMM with a caller-owned-output API, so the shared projection can use that implementation without changing the tensor contract or adding a copy.

The change is guarded to the observed K3 shared-down shape `[M, 6144] x [6144, 7168]`, `M<=128`, contiguous BF16 input, and BF16 output. The FP32 merged front does not match the guard. GEMM arithmetic remains BF16 input/BF16 output with FP32 accumulation; only tile scheduling/backend changes. The existing alternate-stream launch and join remain in place, so all consumers see a completed `shared_output` before the tail.

### analysis.md

Iteration 06 pre-edit analysis

VibeSim workspace-info was called first in the loop. The post-iteration-05 profile was fetched with `simulate?prediction=.:after`, yielding fixed prediction `p_38ccb83a6c4e44298f45b2251bd4cf54`; operator, run-summary, iteration, optimality, and kernel analyses were run before this edit.

The dominant node remains `unified.kda.moe.mxfp4_fused_moe` (R0=412.401 us, R5=9.454 us, no cached alternative). The selected child is `unified.kda.moe.shared_down` (R0=50.179 us, R5=33.566 us, R0/R5=1.49x, `has_cached_alternative=true`). The B=128 profiler confirms the actual shared projection is the `nvjet_sm100_tss_240x64_64x9_1x2_2cta_h_bz_TNN` leaf, so the source-level backend substitution is guarded to that exact K3 tensor shape.

Post-edit VibeSim was fetched and all analysis verbs were rerun. The post-edit profiler confirms the replacement kernel `TgvGemmCuteExtKernel_cta64x128x128_2cta1_pdl1` at 22.592 us; the original shared `nvjet` kernel was 51.392 us. The diagnostic graph latencies are 357.888/236.032/131.488 us. Plain replay is 390.688/251.392/130.560 us, with zero output error, `state_ok=true`, and `pass=true` at all points. This is a valid improvement over iteration 05 but still below the 5% B=128 threshold, so the shared backend remains a candidate for combination with another optimization.

### result.json

```
{
  "iteration": 6,
  "before_plain_graph_us": {
    "128,8192": 392.7040100097656,
    "32,8192": 251.42401456832886,
    "1,8192": 130.52800297737122
  },
  "after_plain_graph_us": {
    "128,8192": 390.6880021095276,
    "32,8192": 251.39200687408447,
    "1,8192": 130.5599957704544
  },
  "checks": {
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "32,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "1,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "accepted_for_final": false,
  "reason": "CuTe TGV shared-down backend is correct and improves B=128 by about 0.5% over the overlap-only candidate, but the combined result is still below 5%."
}
```

(diff.patch: 44 lines, files: /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)

## iter_07
### hypothesis.md

Iteration 07 hypothesis

Iteration 06 mapped the VibeSim `shared_down` alternative to the existing CuTe TGV kernel and reduced the standalone shared kernel to 22.592 us. Its heuristic chooses tactic 24 (`cta64x128`, two-CTA cluster) for the scored shape. Tactic 27 (`cta128x64`, two-CTA cluster) is already compiled and has the same math/output contract but a different tile and stage schedule; it may use the SMs more compatibly with the concurrent routed MXFP4 GEMMs.

This iteration selects tactic 27 only for the exact scored shared-down dimensions `(M,N,K)=(128,7168,6144)`. Smaller batches retain the existing heuristic. The kernel arithmetic and output buffer are unchanged, and the existing stream join remains the correctness fence.

### analysis.md

Iteration 07 pre-edit analysis

VibeSim workspace-info was called first. After iteration 06, `simulate?prediction=.:after` returned fixed prediction `p_38ccb83a6c4e44298f45b2251bd4cf54`; operator, run-summary, iteration, optimality, and kernel analyses were run before this tactic experiment.

The remaining selected node is `unified.kda.moe.shared_down`, whose VibeSim leaf reports R0=50.179 us, R5=33.566 us, and cached alternatives. The real profile now launches `TgvGemmCuteExtKernel_cta64x128x128_2cta1_pdl1` at 22.592 us. Tactic 27 is the existing `cta128x64` two-CTA implementation in the same source, so this is a narrow scheduling experiment with unchanged arithmetic and state dependencies.
