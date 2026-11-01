# KDA_VERIFY_CLAUDE trial 2: agent iterations

## agent log.md

# opt log (TARGET_VERIFY, eager; latency_us 64x8k / 128x8k / 16x64k)
- iter_00: baseline 1599.8 / 1597.8 / 1733.8; host-bound (CPU issue ~= step), VibeSim r0 783us GPU-only
- iter_01: MXFP4 MoE cached-tactic direct launch + strided o_norm gate -> replay 1066.4 / 1055.1 / 1181.7 (same-node A/B r1: 1250.8->1063.3 / 1244.6->1057.2 / 1378.0->1189.6); CHECK pass x3, max_rel_err 0
- iter_02: Triton fast_launch x4 + skip identity topk torch.compile + cheap stream ctx in _forward_fused -> replay 974.2 / 954.9 / 1104.2 (same-node A/B r1: 1241.5->1012.1 / 1235.4->998.9 / 1365.5->1116.0); CHECK pass x3, max_rel_err 0

## iter_00
### analysis.md

# iter_00 — baseline analysis (TARGET_VERIFY, vk3)

## Measurements (plain fixed flags, eager)
64x8k vk3: 1599.8 us; 128x8k: 1597.8 us; 16x64k: 1733.8 us (golden-capture run: 1449/1470/1565)
Latency is ~flat in B (16 -> 128 requests) => not GPU-bound.

## Per-kernel profile (profile.json)
B=64: 24 launches/step, kernel sum 516 us vs 1487 us eager latency.
B=128: kernel sum 696 us vs 1634 us. B=16x64k: kernel sum 333 us vs 1765 us.
=> 65-80% of the step is GPU idle.

## VibeSim
`simulate?prediction=.:before` -> p_38ccb83a6c4e44298f45b2251bd4cf54 (run k3_kda, the fixed
decode-graph prediction served by this container). `analyze?run=k3_kda&level=run_summary`:
r0 783 us GPU, r5 60 us, buckets: hardware_gap 61%, batching 29%. `optimality?scope=iter` ranks
unified.kda.moe.mxfp4_fused_moe first (r0/r5 = 43.6, headroom 403 us), then qkvbfg,
kda_recurrent_decode, merged_front. necessary_share is null for all nodes (R6 not modelled).
The ladder is GPU-time-only: it cannot see the host-side gap that dominates the eager verify
step (the GPU is idle ~1000 us of a ~1500 us step), so the top GPU node is NOT the critical
path here. The "batching / launch / scheduling" reading of the gap is the relevant one.

## Host (CPU) profile (tools/cpuprof.py, torch.profiler with stacks; ~2.2x inflation)
Without profiler: CPU issue time 1219 us/step, total incl. drain 1225 us, cuda-event time 1256 us
=> the step is purely host-issue-bound (no device syncs).
Profiled tree of one layer step (2421 us inflated):
- KDA attention 873: qkvbfg 140 (linear + 2 tiny gemm), target_verify 408 (conv1d Triton
  launch 136, recurrence Triton launch 167, python), o_norm gated 159 (autograd.Function +
  reshape->clone copy kernel + Triton launch), o_proj 73
- MoE 1407: front gemm 44, stream plumbing ~100, shared 75, TopK 271 (moe_fused_gate Triton 125,
  torch.compile'd _post_process_topk_ids 63, env reads), FusedMoE 561 (act quant 73,
  flashinfer autotuner lookup+tuning config ~105, launch 126), latent rmsnorm (flashinfer
  wrappers) 83, up_proj 73, add3 35
- attn-residual aggregate x2: 81

## Decision
Target: host overhead along the whole layer (the critical path), keeping every launched kernel
and its arguments identical so the output is bit-exact.

## iter_01
### hypothesis.md

# iter_01 hypothesis

Built on iter_00/analysis.md: the TARGET_VERIFY step is host-bound (CPU issue ~= step time,
the GPU sits idle 65-80% of the step). VibeSim (run k3_kda) only models kernel time
(r0 783 us), so it cannot rank host costs; ranking uses tools/cputime.py (real inclusive
perf_counter per call) instead.

(a) MXFP4 routed MoE: FusedMoE.forward cost 388 us of host time/step. Most of it was FlashInfer
    Python overhead: flashinfer_api wrapper, MoERunner and MoeRunnerInputs construction,
    _make_tuning_config, AutoTuner.choose_one (46 us), DA gating. Plan: record the tactic
    choose_one returns once per shape (spying on the tuner instance), then call the same
    C++ op (moe_op.trtllm_fp4_block_scale_moe) directly with the identical argument list.
    The fast path is bypassed and the cache cleared under tuning mode, the sglang self-tune
    context, defer_finalize, stream capture, or when DA is enabled. The cache key includes
    the sizes of the tuner's profiling/file caches. Expected: about -150 us/step,
    bit-exact (same kernels, same tactic).
    Ruled out earlier (opt_history summary row: "TRT-LLM MoE tactic buckets / tactic
    re-selection: no effect -- tactics are already autotuned per m"), so this change keeps
    the tactic and only removes host work.
(b) Gated o_norm: port the strided-gate lever from trials/kda_prefill_claude_1
    (incremental.diff l.709-784). The Triton kernel reads the [T/H, H, D] head-split gate
    view in place, which removes the g.reshape copy kernel and its launch. rms_norm_gated
    also skips autograd.Function.apply when grad is disabled. Bit-exact: the same values
    are loaded.

### analysis.md

# iter_01 analysis

Evidence: replay.log, ab.log, cputime.log, cpuprof.log, profile.json, vs_*.json.

- Correctness: CHECK passes at all 3 points with max_rel_err 0.0 and state_ok true. The same kernels and the same MoE tactic run, so the output is bit-exact.
- Latency (replay run): 1599.8/1597.8/1733.8 -> 1066.4/1055.1/1181.7 us (64x8k/128x8k/16x64k).
  Same-node A/B round 1: orig 1250.8/1244.6/1378.0 -> new 1063.3/1057.2/1189.6.
  Round 2 was noisy because a concurrent job was running (1381.7/1434.9/1508.7 vs 1332.5/1267.1/1432.8), but new still won on every point.
- Host (cputime, 64x8k): layer 1405.9 -> 1078.3 us. FusedMoE.forward 343 -> 190 us.
  AutoTuner.choose_one 39.5 -> 0 (the cached-tactic path engages). FusedRMSNormGated 72.5 -> 62.9 us.
- GPU: 24 -> 23 kernels per step (the o_norm gate reshape copy is gone). Kernel sums are otherwise unchanged
  (513.5/710.8/329.1 us). VibeSim r0 is 783 us GPU-only, and it predicts no change because it does not model the host.
- Remaining step is still host-bound (CPU issue ~= step). Next largest host items:
  Triton JITFunction.run bookkeeping (~10 us per launch, 4 launches), the identity
  torch.compile post-process in select_experts (~30 us), and torch.cuda stream-context
  device-index resolution in KimiK3MoE._forward_fused (~30-40 us).

### result.json

```
{
  "iter": 1,
  "edits": ["srt/layers/moe/moe_runner/flashinfer_trtllm.py: cached-tactic direct MXFP4 routed-MoE launch",
            "kernels/ops/attention/fla/fused_norm_gate.py: strided o_norm gate + no-autograd path"],
  "before_latency_us": {"64,8192,vk3": 1599.8, "128,8192,vk3": 1597.8, "16,65536,vk3": 1733.8},
  "after_latency_us_replay_run": {"64,8192,vk3": 1066.4, "128,8192,vk3": 1055.1, "16,65536,vk3": 1181.7},
  "same_node_ab_us": {"orig": [[1250.8,1244.6,1378.0],[1381.7,1434.9,1508.7]],
                       "new":  [[1063.3,1057.2,1189.6],[1332.5,1267.1,1432.8]],
                       "note": "one GPU job, interleaved orig/new; a concurrent replay job added noise"},
  "cputime_64x8k_us": {"layer_before": 1405.9, "layer_after": 1078.3, "FusedMoE.forward": [343.1, 189.6],
                        "AutoTuner.choose_one": [39.5, 0], "FusedRMSNormGated.forward": [72.5, 62.9]},
  "kernels_per_step_B64": [24, 23],
  "check": {"64,8192,vk3": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
            "128,8192,vk3": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
            "16,65536,vk3": {"max_rel_err": 0.0, "state_ok": true, "pass": true}},
  "vibesim": "GPU-time prediction unchanged (kernels are bit-identical except one removed copy); host gap is not modelled"
}
```

(diff.patch: 374 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/fused_norm_gate.py, /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_02
### hypothesis.md

# iter_02 hypothesis

Based on iter_01/analysis.md: the step remains host-bound after iter_01 (layer CPU 1078 us vs GPU 514 us at 64x8k).
VibeSim cannot rank host costs because it models only GPU time (r0 783 us).
The ranking below comes from iter_01/cputime.log and cpuprof.log.

(a) Triton launch bookkeeping. A tools/tlaunch_bench.py microbenchmark on the B200 node measured the full K[grid]() at 14.0 us.
    The binder plus kernel.run alone cost ~6.8 us. The difference comes from JITFunction.run's per-call work:
    stream lookup through torch, hook dispatch, launch_metadata, and the grid tuple.
    Add kernels/ops/triton_launch.fast_launch. It reuses Triton's own binder and cache key, so it resolves to the same
    compiled kernel, and on a cache hit it calls compiled.run directly. It falls back whenever hooks
    or stage inspection are active. Apply it to the four eager-path launches (o_norm, fused_sigmoid
    recurrent, causal_conv1d_update, router gate). Expected: ~-10 us per launch, bit-exact.
(b) _post_process_topk_ids enters the torch.compile'd _biased_grouped_topk_postprocess even
    when expert_location_dispatch_info and num_token_non_padded are both None, which makes it an identity.
    Skip that call in this case. Expected: ~-30 us.
(c) KimiK3MoE._forward_fused (overlap_shared path): torch.cuda.current_stream() and
    torch.cuda.stream(alt) resolve the device index on every call. Replace them with thin helpers that call
    torch._C._cuda_getCurrentStream/_cuda_setStream with the alt stream's device index. Same semantics
    for a same-device side stream. Expected: ~-20-30 us.
No prior trial in opt_history covers these; the iter_01 MoE fast path is separate (tactic kept, host only).

### analysis.md

# iter_02 analysis

Evidence: smoke.log (LAYER_SMOKE_OK), replay.log, ab.log, cputime.log, profile.json, vs_*.json.

- Correctness: CHECK passes at all 3 points with max_rel_err 0.0 and state_ok true.
- Latency (replay run): 1066.4/1055.1/1181.7 (iter_01) -> 974.2/954.9/1104.2 us.
  Same-node A/B, original vs iter_02 tree. Round 1: orig 1241.5/1235.4/1365.5 -> new 1012.1/998.9/1116.0.
  Round 2 was noisy (orig 1201.7/1220.0/1728.2, new 1197.6/1137.2/1353.8), but new was never slower.
- Host (cputime, 64x8k), iter_01 -> iter_02: layer 1078.3 -> 1000.4 us. select_experts 105 -> 75 us
  (_post_process_topk_ids 2.7 us). _forward_fused ~600 -> 533 us.
  Triton wrappers each dropped ~10 us: conv1d 71->59, fused_sigmoid 80->67, moe_fused_gate 61->51, RMSNormGated 63->53.
- GPU: 23/23/24 kernels per step, unchanged. Kernel sums are 514.9/712.0/330.4 us, matching iter_01 within noise.
  VibeSim: no change predicted (GPU-only model, same kernels).
- Remaining host time (64x8k): FusedMoE.forward 190 us, mostly FlashInfer C++ launcher allocations outside
  the tree. KDA target_verify 175 us. qkvbfg 76 us. _k3_bf16_gemm x2 48 us. mxfp8 activation prep 46 us.
  _latent_norm 43 us (FlashInfer rmsnorm calls get_device_capability on every call, in external code).

### result.json

```
{
  "iter": 2,
  "edits": ["kernels/ops/triton_launch.py (new): fast_launch for cached Triton kernels",
            "fused_norm_gate.py, fused_sigmoid_gating_recurrent.py, causal_conv1d_triton.py, moe_fused_gate.py: use fast_launch",
            "srt/layers/moe/topk.py: skip identity torch.compile post-process",
            "srt/models/kimi_k3.py: cheap current-stream / stream-context helpers on the overlap_shared path"],
  "before_latency_us_replay_run": {"64,8192,vk3": 1066.4, "128,8192,vk3": 1055.1, "16,65536,vk3": 1181.7},
  "after_latency_us_replay_run": {"64,8192,vk3": 974.2, "128,8192,vk3": 954.9, "16,65536,vk3": 1104.2},
  "same_node_ab_us": {"orig": [[1241.5,1235.4,1365.5],[1201.7,1220.0,1728.2]],
                       "new":  [[1012.1,998.9,1116.0],[1197.6,1137.2,1353.8]]},
  "cputime_64x8k_us": {"layer_before": 1078.3, "layer_after": 1000.4, "select_experts": [105, 75.2], "KimiK3MoE._forward_fused": [600, 532.9]},
  "kernel_sum_us": {"64,8192,vk3": 514.9, "128,8192,vk3": 712.0, "16,65536,vk3": 330.4},
  "check": {"64,8192,vk3": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
            "128,8192,vk3": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
            "16,65536,vk3": {"max_rel_err": 0.0, "state_ok": true, "pass": true}}
}
```

(diff.patch: 600 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/fused_norm_gate.py, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py, /sgl-workspace/sglang/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py, /sgl-workspace/sglang/python/sglang/kernels/ops/moe/moe_fused_gate.py, /sgl-workspace/sglang/python/sglang/kernels/ops/triton_launch.py, /sgl-)
