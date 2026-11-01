# KDA_VERIFY_CLAUDE trial 1: agent iterations

## agent log.md

- iter_00: baseline 64:1530.3 128:1510.8 16x64k:1746.8 us; host-bound (GPU ~520us/step at B=64)
- iter_01: MoE cached-tactic direct FFI (flashinfer_trtllm.py) 64:1530.3->1133.0 128:1510.8->1157.5 16x64k:1746.8->1246.6 us (cross-node); CHECK pass, rel 0.0
- iter_01 same-node A/B (iter_02/ab): base 1582.5/1544.5/1684.4 -> v01 1419.5/1383.7/1484.8 us
- iter_02: rms_norm_gated no-grad bypass + skip identity compiled topk postprocess: v01 -> 1304.5/1259.0/1340.9 us (same node); CHECK pass, rel 0.0

## iter_00
### hypothesis.md

# iter_00 hypothesis
Baseline only; no change. Prior trials (opt_history/TECHNIQUES.md) were all CUDA-graph decode levers.
This workload runs eagerly, so the B-section GPU-kernel dead ends (tactic re-selection, fused router)
are ruled out as the primary lever here, and host overhead reduction is new territory.

### analysis.md

# iter_00 - baseline analysis

VibeSim prediction id: p_38ccb83a6c4e44298f45b2251bd4cf54 (handle `.:k3_kda`, GPU-only decode model)

## Measured baseline (run.sh plain, eager TARGET_VERIFY)
| point | latency_us |
|---|---|
| 64,8192,vk3 | 1530.3 |
| 128,8192,vk3 | 1510.8 |
| 16,65536,vk3 | 1746.8 |

## Findings
- VibeSim optimality top node: `unified.kda.moe.mxfp4_fused_moe` (r0/r5 = 43.6, largest headroom);
  next are qkvbfg projections and kda_recurrent_decode. VibeSim does not model host/launch idle.
- The per-kernel profile (profile.json) sums to ~520 us of GPU time at B=64, while the step takes ~1360-1530 us.
  nsys shows the GPU idle between launches, so the step is **host (CPU launch) bound** at all 3 points.
- cProfile per step: the MoE `_forward_fused` takes ~1008 us of the host time. Of that, the flashinfer
  `trtllm_fp4_block_scale_moe_op` takes 227 us: MoERunner + `_make_tuning_config` 39 us, `choose_one` 53 us,
  kwargs/self 26 us, and only 88 us in the FFI launch. The topk path takes 174 us and the Triton launches
  ~50 us each (x4).
  The attention host path is ~630 us.
- Conclusion: kernel swaps do not move latency. Removing host Python work on the critical path does, 1:1.

### result.json

```
{"iter":0,"change":"baseline","latency_us":{"64,8192,vk3":1530.3,"128,8192,vk3":1510.8,"16,65536,vk3":1746.8},"check":"pass (golden captured)"}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis
The routed MXFP4 tactic is a pure function of the input shapes/dtypes and the autotuner cache. Cache
(raw moe_op, tactic) per shape key (invalidated when the autotuner caches grow), and call the raw
FFI `moe_op.trtllm_fp4_block_scale_moe` with the identical positional ABI as flashinfer's `run_selected_tactic`.
Expected: -100..-140 us/step at all points, bit-exact (same kernel, same tactic).
Prior trials: TECHNIQUES.md B "tactic re-selection: no effect" - we do not change the tactic, only
the host path to reach it, so that dead end does not apply. Guarded off for autotune mode, tuning overrides,
DA autotune, the deferred-finalize path and the self-tune context.

### analysis.md

# iter_01 analysis
VibeSim prediction id: p_38ccb83a6c4e44298f45b2251bd4cf54 (`.:k3_kda`). Node picked: `unified.kda.moe.mxfp4_fused_moe`
(optimality top, r0/r5 = 43.6). The VibeSim gap is GPU-only, but the measured bottleneck of this node is host time.
profile.json is the iter_00 kernel profile (this change launches the same kernels; only host time moves).
Findings (cProfile, iter_00/steady_1.prof): flashinfer `trtllm_fp4_block_scale_moe_op` takes 227 us/step of host time,
and only 88 us of that is the FFI launch. The rest is MoERunner construction, `_make_tuning_config`, the
runner_kwargs dict and `AutoTuner.choose_one` cache lookup, all recomputed every call for an unchanged shape.

### result.json

```
{"iter":1,"node":"unified.kda.moe.mxfp4_fused_moe","files":["srt/layers/moe/moe_runner/flashinfer_trtllm.py"],
 "latency_us_before":{"64,8192,vk3":1530.3,"128,8192,vk3":1510.8,"16,65536,vk3":1746.8},
 "latency_us_after":{"64,8192,vk3":1133.0,"128,8192,vk3":1157.5,"16,65536,vk3":1246.6},
 "note":"before/after from different slurm nodes; same-node A/B in iter_02/ab",
 "check":{"64,8192,vk3":{"pass":true,"max_rel_err":0.0},"128,8192,vk3":{"pass":true,"max_rel_err":0.0},"16,65536,vk3":{"pass":true,"max_rel_err":0.0}}}
```

(diff.patch: 225 lines, files: /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_02
### hypothesis.md

# iter_02 hypothesis
Builds on iter_01 (host-bound; every host us is a latency us). Skip autograd.Function.apply when grad is
disabled (same forward code, ctx dropped), and skip the compiled identity postprocess when both of its optional
args are None. Expected about -60 us, bit-exact. No prior trial touched host overhead, so nothing is ruled out.

### analysis.md

# iter_02 analysis
VibeSim prediction id: p_38ccb83a6c4e44298f45b2251bd4cf54 (`.:k3_kda`). Nodes: `unified.kda.attention` o_norm (rms_norm_gated)
and the MoE router (select_experts). VibeSim models neither node's host cost. profile.json is unchanged from
iter_00, since these changes launch the same kernels.
Findings (cProfile, iter_00/steady_1.prof):
- `rms_norm_gated` goes through `torch.autograd.Function.apply` even under inference_mode: 124 us cum vs 93 us of forward.
- `_post_process_topk_ids` calls the torch.compile'd `_biased_grouped_topk_postprocess`. With no EPLB dispatch info and
  no padded-token count it is an identity (body 2 us), yet dynamo guard evaluation costs 48 us.

### result.json

```
{"iter":2,"node":["unified.kda.attention.o_norm","unified.kda.moe.router"],"files":["kernels/ops/attention/fla/fused_norm_gate.py","srt/layers/moe/topk.py"],
 "same_node_ab":{"base":{"64,8192,vk3":1582.5,"128,8192,vk3":1544.5,"16,65536,vk3":1684.4},
  "v01":{"64,8192,vk3":1419.5,"128,8192,vk3":1383.7,"16,65536,vk3":1484.8},
  "v02":{"64,8192,vk3":1304.5,"128,8192,vk3":1259.0,"16,65536,vk3":1340.9}},
 "check":{"64,8192,vk3":{"pass":true,"max_rel_err":0.0},"128,8192,vk3":{"pass":true,"max_rel_err":0.0},"16,65536,vk3":{"pass":true,"max_rel_err":0.0}}}
```

(diff.patch: 84 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/fused_norm_gate.py, /sgl-workspace/sglang/python/sglang/srt/layers/moe/topk.py)
