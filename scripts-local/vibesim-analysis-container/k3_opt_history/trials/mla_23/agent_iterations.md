# MLA trial 23: agent iterations

## agent log.md

iter_00: baseline profile; B=1/L=1048576 264.704 us, B=128/L=8192 486.624 us, B=16/L=65536 298.528 us; VibeSim ranked MXFP4 MoE and MLA attention highest.
iter_01: MXFP4 token-tuning experiment; 264.704 -> 265.696 us, 486.624 -> 485.120 us, 298.528 -> 298.464 us; checks pass, rejected.
iter_02: FP16-softmax experiment; SM100 smoke guard rejected the SM107-only option, source restored.
iter_03: fixed-sequence MLA experiment; 264.704 -> 265.664 us, 486.624 -> 484.928 us, 298.528 -> 298.496 us; checks pass, rejected.
iter_04: merged-front cuBLAS experiment; 264.704 -> 267.648 us, 486.624 -> 484.992 us, 298.528 -> 300.512 us; checks pass, rejected.
iter_05: latent_up BF16 TGV `(7168,3584)` in `kimi_k3.py`; 264.704 -> 263.680 us, 486.624 -> 484.896 us, 298.528 -> 298.464 us; all CHECK pass, retained as base.
iter_06: shared_down BF16 TGV out `(7168,6144)` in `kimi_k3.py`; 263.680 -> 261.600 us, 484.896 -> 484.864 us, 298.464 -> 298.304 us; all CHECK pass, accepted.
iter_07: merged_front TGV tactic 15 experiment in `cutedsl_bf16_gemm.py`; 261.600 -> 296.480 us, 484.864 -> 485.888 us, 298.304 -> 298.368 us; all CHECK pass, rejected and reverted.
iter_08: q_b_proj exact-shape TGV experiment in `kimi_k3.py`; 261.600 -> 261.600 us, 484.864 -> 484.864 us, 298.304 -> 298.432 us; all CHECK pass, no gain, reverted.

## iter_01
### hypothesis.md

VibeSim iteration 0 selected `unified.mla.moe.mxfp4_fused_moe` as the node with the largest R0-R5 headroom and a 0.60 necessary-work share. The profiler confirms the 112-expert/top-2 SiTU path launches the TRT-LLM MXFP4 BMM pair.

The current call tunes each launch to exactly `next_power_of_2(num_tokens)`. For the small decode buckets, use a 32-token tuning envelope instead. This changes only FlashInfer's launch/autotuning choice; routing ids, BF16 packed weights, scales, activation, accumulation, output buffer, and all layer state updates are unchanged. The 128-token workload keeps its existing envelope.

### analysis.md

VibeSim prediction: `p_c7687e585abe4b5fbacdc89caf0c9761` (`vibesim:k3_mla`), built with `prediction=.:before` and re-fetched after the variant. The fixed prediction reports `unified.mla.moe.mxfp4_fused_moe` first: R0=379.160 us, R5=8.573 us, R6/R7=227.756 us, necessary share=0.6007, R0/R5=44.23. Its kernel drill-down is the 112-expert/top-2 `sglang_trtllm_mxfp4` leaf with `has_cached_alternative=false`.

The post-edit profile still has the same 28/29 launches and the same TRT-LLM MXFP4 BMM pair. Plain graph replay was 265.696 -> 265.696 us at B=1/L=1048576, 486.624 -> 485.120 us at B=128/L=8192, and 298.528 -> 298.464 us at B=16/L=65536. The primary point did not improve, so this hypothesis is rejected despite all checks passing.

### result.json

```
{
  "hypothesis": "mxfp4 tune_max_num_tokens=max(32, next_power_of_2(tokens))",
  "before_latency_us": [264.703989, 486.624002, 298.527986],
  "after_latency_us": [265.695989, 485.119998, 298.464000],
  "checks": [
    {"point": [1, 1048576], "pass": true, "max_rel_err": 0.0, "state_ok": true},
    {"point": [128, 8192], "pass": true, "max_rel_err": 0.0, "state_ok": true},
    {"point": [16, 65536], "pass": true, "max_rel_err": 0.0, "state_ok": true}
  ],
  "accepted": false
}
```

(diff.patch: 14 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_02
### hypothesis.md

Iteration 0 VibeSim identifies the MLA attention subtree as the second large headroom node, and the per-kernel profile confirms the primary long-context leaf is the TRT-LLM FP8 MLA decode cubin (`fmhaSm100fKernel...`). FlashInfer provides a shape-specific `use_fp16_softmax` variant for the exact 576/512 MLA dimensions.

Enable that variant only for the TRT-LLM FP8 decode call. It changes the softmax accumulator implementation but preserves the same query, paged KV reads, causal bounds, scales, output buffer, and KV post-step write. The replay against the original golden will verify the allowed numerical and state tolerance on all points.

### analysis.md

The iteration-2 attention experiment was rejected during the mandated smoke test. The VibeSim iteration-0 attention node was `unified.mla.attention.mla_decode_attention` (R0=358.836 us, R5=227.559 us), and the profile leaf was the TRT-LLM FP8 MLA decode cubin. FlashInfer then rejected `use_fp16_softmax=True` on the B200 (SM100): that cubin variant is SM107-only. No replay/profile was run; the source was restored and the strict smoke passed.

### result.json

```
{
  "hypothesis": "enable FlashInfer TRT-LLM FP16 softmax for FP8 MLA decode",
  "accepted": false,
  "blocked_at_smoke": true,
  "error": "use_fp16_softmax is only supported on SM107 (Rubin); current device is SM100",
  "source_restored": true
}
```

(diff.patch: 4 lines, files: trtllm_mla_backend.py)

## iter_03
### hypothesis.md

The VibeSim attention node remains the largest primary-work leaf after the rejected MoE tuning attempt, and the profile shows the B=1 long-context kernel uses the `...VarSeq...` TRT-LLM cubin. For one request, sequence-length metadata is inherently uniform, so pass `is_var_seq=False` only when the decode query batch has one row.

This changes only the TRT-LLM scheduling specialization. The query, page table, sequence length, scales, softmax, output, and MLA KV write are unchanged; B>1 keeps the existing variable-sequence path.

### analysis.md

VibeSim prediction `p_c7687e585abe4b5fbacdc89caf0c9761` still ranks `unified.mla.moe.mxfp4_fused_moe` first (R0=379.160 us, R5=8.573 us, R6/R7=227.756 us, necessary share=0.6007); the attention leaf is R0=358.836 us, R5=227.559 us. The profiler confirmed that the B=1 request continued to launch `fmhaSm100fKernel...VarSeq...` despite `is_var_seq=False`, so the backend did not select a new kernel.

The replay checks passed exactly, but graph latency was 264.704 -> 265.664 us at B=1/L=1048576, 486.624 -> 484.928 us at B=128/L=8192, and 298.528 -> 298.496 us at B=16/L=65536. The primary point did not improve; the source change is rejected and restored.

### result.json

```
{
  "hypothesis": "use fixed-sequence TRT-LLM MLA scheduling for B=1",
  "before_latency_us": [264.703989, 486.624002, 298.527986],
  "after_latency_us": [265.664011, 484.928012, 298.496008],
  "checks": [
    {"point": [1, 1048576], "pass": true, "max_rel_err": 0.0, "state_ok": true},
    {"point": [128, 8192], "pass": true, "max_rel_err": 0.0, "state_ok": true},
    {"point": [16, 65536], "pass": true, "max_rel_err": 0.0, "state_ok": true}
  ],
  "accepted": false,
  "kernel_unchanged": true
}
```

(diff.patch: 10 lines, files: python/sglang/srt/layers/attention/trtllm_mla_backend.py)

## iter_04
### hypothesis.md

VibeSim's merged-front `single_gemm` node has implementation headroom (R0=158.805 us, R5=121.067 us, necessary share=0.1892), and the per-kernel table maps the primary front to `TgvGemmCuteExtKernel_cta128x16...` at about 37 us. The current K3-specific override forces that TGV kernel for the 15984x7168 FP32-output merged front at batches <=16.

Remove only the 15984x7168 override so this shape uses the normal SM100 cuBLAS BF16 GEMM path; leave the existing TGV choices for all other K3 shapes. Both implementations compute the same FP32-accumulated linear projection into the same output buffer, and routing/output/state are checked against the original golden.

### analysis.md

VibeSim prediction `p_c7687e585abe4b5fbacdc89caf0c9761` identifies the merged-front `single_gemm` as an implementation-limited node (R0=158.805 us, R5=121.067 us, R6/R7=30.048 us, necessary share=0.1892). The profile A/B mapped the TGV front kernel at 37.19 us and the cuBLAS fallback at 39.18 us.

All replay checks passed, but graph latency changed 264.704 -> 267.648 us at B=1/L=1048576, 486.624 -> 484.992 us at B=128/L=8192, and 298.528 -> 300.512 us at B=16/L=65536. The primary point regressed, so the TGV source specialization is restored.

### result.json

```
{
  "hypothesis": "use cuBLAS for the 15984x7168 FP32 merged-front GEMM",
  "before_latency_us": [264.703989, 486.624002, 298.527986],
  "after_latency_us": [267.648011, 484.991997, 300.511986],
  "checks": [
    {"point": [1, 1048576], "pass": true, "max_rel_err": 0.0, "state_ok": true},
    {"point": [128, 8192], "pass": true, "max_rel_err": 0.0, "state_ok": true},
    {"point": [16, 65536], "pass": true, "max_rel_err": 0.0, "state_ok": true}
  ],
  "accepted": false
}
```

(diff.patch: 11 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_05
### hypothesis.md

VibeSim ranks `unified.mla.moe.latent_up` as an implementation-limited leaf (R0=43.099 us, R5=26.083 us, R6/R7=6.423 us, necessary share=0.1490). In the profiler, the driver reaches this K3 latent-up projection through a generic BF16 `nvjet` GEMM; the repository already has an SM100 TGV implementation for the exact `[3584, 7168]` K3 shape.

For small decode batches, call the existing TGV BF16 GEMM directly for this projection. The weight, input, accumulation type, output shape, and post-projection add are unchanged; only the implementation of the same linear operation changes. Quantized/non-BF16 weights and larger batches keep the existing module path.

### analysis.md

Prediction: `p_c7687e585abe4b5fbacdc89caf0c9761` (`vibesim:k3_mla`).

VibeSim continued to identify `unified.mla.moe.latent_up` as an implementation-limited leaf: R0=43.099 us, R5=26.083 us, R6/R7=6.423 us, and `necessary_share=0.149018`. The R0/R5 ratio is 1.652, so the leaf has meaningful implementation headroom, although its total share is smaller than the fused MoE and MLA attention nodes. The per-kernel profile confirmed that the K3 latent-up projection launched the new `kernel_cutlass_kernel_TgvGemmCuteExtKernel_cta64x16x128_2cta1_pdl1_outbfloat16...` kernel for the small decode shapes.

The retained change is limited to BF16, contiguous K3 latent-up weights with shape `(7168, 3584)` and batches up to 16. It calls the existing CuTe TGV implementation and leaves the module path unchanged for all other shapes and dtypes. The exact replay checks were zero for output and post-step state at all three points.

The first-point replay moved from 264.704 us to 263.680 us in the fixed plain run, about 0.39%. This is a useful base for the next candidate but does not independently clear the 0.5% scoring threshold.

### result.json

```
{
  "iteration": 5,
  "before_latency_us": {
    "1,1048576": 264.70398902893066,
    "128,8192": 486.62400245666504,
    "16,65536": 298.5279858112335
  },
  "after_latency_us": {
    "1,1048576": 263.68001103401184,
    "128,8192": 484.8960041999817,
    "16,65536": 298.46400022506714
  },
  "checks": {
    "1,1048576": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    "16,65536": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "accepted": false,
  "kept_as_base": true
}
```

(diff.patch: 38 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_06
### hypothesis.md

VibeSim ranks `unified.mla.moe.shared_down` as an implementation-limited leaf: R0=65.361 us, R5=44.526 us, R6/R7=11.010 us, and `necessary_share=0.168450`. The current K3 profile contains the shared-expert side-stream GEMM among the remaining BF16 `nvjet` launches, while the fused front and latent-up paths use explicit TGV launches.

The shared down projection is a dense BF16 GEMM with the fixed K3 shape `(7168, 6144)` and writes into the caller-owned BF16 `shared_output` slice. For decode batches up to 16, call the existing CuTe TGV out kernel directly for this exact contiguous shape. The guard preserves the current module path for all other shapes, dtypes, and batch sizes; the mathematical operation, accumulator type, destination tensor, and later add/reduction are unchanged.

### analysis.md

Prediction: `p_c7687e585abe4b5fbacdc89caf0c9761` (`vibesim:k3_mla`).

VibeSim was run after the profile. Its fixed prediction still places `unified.mla.moe.mxfp4_fused_moe` and MLA attention first, followed by `unified.mla.moe.merged_front`; the targeted `unified.mla.moe.shared_down` leaf is R0=65.361 us, R5=44.526 us, R6/R7=11.010 us, with `necessary_share=0.168450` and R0/R5=1.468. This is a necessary-work implementation bottleneck with measurable headroom. The kernel drill-down identifies shape `(n=7168, k=6144, dtype=bf16)` and a cached alternative.

The post-edit profile confirms the exact source path now launches the BF16 TGV out kernel for the shared down projection. The B1 diagnostic profile has two `cta64x16x128_2cta1` BF16 TGV out launches totaling 26.173 us; the first point's plain graph replay is 261.600 us. The larger shapes retain their existing dispatch because the guard is limited to batches up to 16.

### result.json

```
{
  "iteration": 6,
  "before_latency_us": {
    "1,1048576": 263.68001103401184,
    "128,8192": 484.8960041999817,
    "16,65536": 298.46400022506714
  },
  "after_latency_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 484.8639965057373,
    "16,65536": 298.3039915561676
  },
  "checks": {
    "1,1048576": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    "16,65536": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "accepted": true
}
```

(diff.patch: 22 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_07
### hypothesis.md

VibeSim ranks `unified.mla.moe.merged_front` as the next concrete implementation candidate: R0=158.805 us, R5=121.067 us, R6/R7=30.048 us, `necessary_share=0.189216`, and R0/R5=1.312. The B1 profile confirms the merged front is the FP32-output TGV launch `cta128x16x128_2cta0` at about 37 us.

The TGV tactic ladder currently selects tactic 12 for the observed `(M=1, N=15984, K=7168)` front shape because its 128x16 one-CTA grid is the first ladder entry under one wave. Tactic 15 uses the same one-CTA 128xN tile family with three pipeline stages and the same one-wave grid. Test that exact shape-specific choice; it changes only scheduling/tile selection for the same FP32-accumulating GEMM and leaves all other dispatches unchanged.

### analysis.md

Prediction: `p_c7687e585abe4b5fbacdc89caf0c9761` (`vibesim:k3_mla`).

VibeSim identified `unified.mla.moe.merged_front` as the tested leaf with R0=158.805 us, R5=121.067 us, R6/R7=30.048 us, `necessary_share=0.189216`, and R0/R5=1.312. The profile confirmed the candidate kernel, but the alternative tactic 15 changed the launch to `cta128x128x128_2cta0` and raised its measured kernel time to 72.576 us, versus about 37.3 us for the original `cta128x16x128_2cta0`. The experiment was therefore rejected and the source override was reverted. Replay checks remained exact at all three points.

### result.json

```
{
  "iteration": 7,
  "before_latency_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 484.8639965057373,
    "16,65536": 298.3039915561676
  },
  "after_latency_us": {
    "1,1048576": 296.4800000190735,
    "128,8192": 485.8880043029785,
    "16,65536": 298.3680069446564
  },
  "checks": {
    "1,1048576": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    "16,65536": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "accepted": false,
  "reverted": true
}
```

(diff.patch: 11 lines, files: /workspace/opt_run/iter_07/variant/cutedsl_bf16_gemm.py)

## iter_08
### hypothesis.md

VibeSim ranks `unified.mla.attention.q_b_proj` as a small but implementation-limited leaf: R0=16.335 us, R5=3.679 us, R6/R7=0.885 us, `necessary_share=0.054162`, and R0/R5=4.440. Its drill-down shape is `(n=2304, k=1536, dtype=bf16)` and the B1 profile contains the corresponding small attention GEMM among the remaining non-TGV launches.

Kimi-K3 uses the inherited `q_b_proj_forward` implementation, whose low-latency fused-A verification table does not include this exact K3 shape. For the MLA workload, dispatch the same BF16 linear through the existing CuTe TGV kernel when the raw weight is contiguous `(2304, 1536)` and the decode batch is at most 16. The fallback remains the inherited method for every other model/layout/shape, and the result is reshaped identically for the attention core.

### analysis.md

Prediction: `p_c7687e585abe4b5fbacdc89caf0c9761` (`vibesim:k3_mla`).

VibeSim identified `unified.mla.attention.q_b_proj` with R0=16.335 us, R5=3.679 us, R6/R7=0.885 us, `necessary_share=0.054162`, and R0/R5=4.440. The post-edit profile shows the exact-shape dispatch did launch an additional `cta64x16x128_2cta1` TGV kernel, but the three TGV out launches totaled 30.614 us versus 26.173 us in the retained base profile. CUDA-graph latency stayed at 261.600 us on the first point and did not improve. The override was reverted after the required smoke passed.

### result.json

```
{
  "iteration": 8,
  "before_latency_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 484.8639965057373,
    "16,65536": 298.3039915561676
  },
  "after_latency_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 484.8639965057373,
    "16,65536": 298.43199253082275
  },
  "checks": {
    "1,1048576": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    "16,65536": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "accepted": false,
  "reverted": true
}
```

(diff.patch: 28 lines, files: /workspace/opt_run/iter_08/variant/kimi_k3.py)
