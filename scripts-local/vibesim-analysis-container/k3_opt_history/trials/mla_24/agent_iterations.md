# MLA trial 24: agent iterations

## iter_01
### hypothesis.md

Baseline profile and VibeSim both identify the fused MoE front as a material
candidate: the profile's B=1 and B=16 front launch is
TgvGemmCuteExtKernel_cta128x16, about 38.1 us, and VibeSim reports
merged_front R0=158.8 us with necessary_share=18.9% in its aggregate decode
prediction. The exact B=1 shape is M=1, N=15984, K=7168.

Change only the TGV tactic selection for that shape to the already compiled
cta128x8/stage-6 variant (tactic 11). The kernel performs the same bf16 x bf16
fp32-accumulate GEMM and writes the same fp32 output; only the output tile
width changes, so routing logits, selected experts, KV/cache state, and layer
output should remain unchanged. The guard is exact and leaves B=16, B=128,
other models, and all other shapes on their previous tactic selection.

### analysis.md

VibeSim prediction: `p_c7687e585abe4b5fbacdc89caf0c9761`, fetched with
`simulate?prediction=.:iter_01` (the service returns the fixed before-state
prediction). The operator analysis reports 1.1589 ms total predicted kernel
time: `mxfp4_fused_moe` 379.2 us (32.7%), `mla_decode_attention` 358.8 us
(31.0%), and `merged_front` 158.8 us (13.7%).

The iteration roofline ranks `mxfp4_fused_moe` first (R0=379.2 us,
R5=8.6 us, R6/R7=227.8 us, necessary_share=60.1%), then the attention plus
output-gate subtree (R0=436.2 us, R5=128.9 us, R6/R7=462.0 us,
necessary_share=105.9%). The leaf `mla_decode_attention` is R0=358.8 us,
R5=227.6 us, R6/R7=454.2 us, necessary_share=126.6%. `merged_front` is
R0=158.8 us, R5=121.1 us, R6/R7=30.0 us, necessary_share=18.9%.

The post-edit profiler confirmed the intended B=1 launch changed from
`TgvGemmCuteExtKernel_cta128x16` to `TgvGemmCuteExtKernel_cta128x8`, with
37.0 us measured kernel time. Graph latency did not move from 261.6 us, so
the front GEMM is overlapped/off the B=1 critical path. The VibeSim `kernels`
endpoint in this service accepted `kernel_set` but returned only a placeholder
entry with `has_cached_alternative=false`; the real per-leaf confirmation is
in the profiler JSON files listed above.

Decision: reject this candidate because the primary graph latency did not
improve. The exact attention kernel remains the next candidate to test.

### result.json

```
{
  "status": "rejected_no_primary_gain",
  "before_plain_graph_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 486.8159890174866,
    "16,65536": 298.46400022506714
  },
  "replay_graph_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 485.8880043029785,
    "16,65536": 296.4160144329071
  },
  "checks": [
    {"point": "1,1048576", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "16,65536", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ]
}
```

(diff.patch: 14 lines, files: python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py)

## iter_02
### hypothesis.md

Iteration 01 showed the B=1 front tile is off the CUDA-graph critical path.
This iteration removes that no-op experiment and tests the attention kernel
implementation selected by the same `cutedsl_mla` workload flag. The current
class is named CuteDslMLABackend but its non-DCP constructor was changed to
`backend="trtllm-gen"`; the base class can also launch `backend="cute-dsl"`.

The only behavioral change under test is the FlashInfer MLA kernel selection.
Both paths consume the same fp8 paged KV cache, query layout, sequence metadata,
and return the same attention tensor. The pre-edit golden checks output and all
captured post-step state, so any numerical or cache-state difference will fail
the replay.

### analysis.md

VibeSim prediction: prediction_id p_c7687e585abe4b5fbacdc89caf0c9761,
fetched with simulate?prediction=.:iter_02. The fixed prediction remains
1.1589 ms predicted kernel time; operator shares are mxfp4_fused_moe 379.2 us
(32.7%), mla_decode_attention 358.8 us (31.0%), and merged_front 158.8 us
(13.7%). The optimality ladder remains: mxfp4 fused MoE R0=379.2 us,
R5=8.6 us, R6/R7=227.8 us, necessary_share=60.1%; MLA decode attention
R0=358.8 us, R5=227.6 us, R6/R7=454.2 us, necessary_share=126.6%.

The profiler mapped the alternate source path to the
flashinfercute_dslattentionmonolithicmla_decode kernel. It measured 167.1 us
for B=1/L=1M versus 129.9 us for the current TRT-LLM kernel, and added a
launch. The VibeSim kernels call returned its service placeholder with
has_cached_alternative=false; the real launch table is in the raw profile
files.

Decision: reject CuteDSL for this non-DCP one-token graph. Restore
backend="trtllm-gen" and continue with the faster implementation.

### result.json

```
{
  "status": "rejected_slower",
  "before_plain_graph_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 486.8159890174866,
    "16,65536": 298.46400022506714
  },
  "replay_graph_us": {
    "1,1048576": 300.5119860172272,
    "128,8192": 493.0559992790222,
    "16,65536": 298.5599935054779
  },
  "checks": [
    {"point": "1,1048576", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "128,8192", "max_rel_err": 0.01376146687989234, "state_ok": true, "pass": true},
    {"point": "16,65536", "max_rel_err": 0.0048515978190613466, "state_ok": true, "pass": true}
  ]
}
```

(diff.patch: 11 lines, files: python/sglang/srt/layers/attention/cutedsl_mla_backend.py)

## iter_03
### hypothesis.md

The current profiler confirms the fused fp8 KV scatter/query preparation
launch, set_mla_kv_concat_q_fp8_kernel<8>, is on the MLA preparation path and
costs about 5.5 us at B=1/L=1M and 16.7 us at B=128/L=8K. Its CUDA wrapper
supports 1, 2, 4, and 8 warps per block. Test the 4-warp variant used by the
pre-existing heuristic.

This changes only CTA occupancy and scheduling. Every row conversion, fp8
quantization operation, KV-cache store, query output, and post-step cache
location is unchanged, so the golden output/state comparison must remain
identical.

### analysis.md

VibeSim prediction: prediction_id p_c7687e585abe4b5fbacdc89caf0c9761,
fetched with simulate?prediction=.:iter_03. Operator and optimality results
remain fixed: mxfp4_fused_moe R0=379.2 us, R5=8.6 us, R6/R7=227.8 us,
necessary_share=60.1%; mla_decode_attention R0=358.8 us, R5=227.6 us,
R6/R7=454.2 us, necessary_share=126.6%; merged_front R0=158.8 us,
R5=121.1 us, R6/R7=30.0 us, necessary_share=18.9%.

The profiler confirmed the source change as
set_mla_kv_concat_q_fp8_kernel<4>. The launch measured 5.3279 us at B=1,
16.6749 us at B=128, and 5.6830 us at B=16, but the primary graph replay was
263.6 us in the diagnostic run and 261.728 us in the plain replay versus the
261.600 us baseline. The VibeSim kernels endpoint returned only its
has_cached_alternative=false placeholder; the raw profiler tables are the
per-leaf evidence.

Decision: reject the 4-warp variant and restore the 8-warp configuration.

### result.json

```
{
  "status": "rejected_no_primary_gain",
  "before_plain_graph_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 486.8159890174866,
    "16,65536": 298.46400022506714
  },
  "replay_graph_us": {
    "1,1048576": 261.7279887199402,
    "128,8192": 485.9519898891449,
    "16,65536": 296.57599329948425
  },
  "checks": [
    {"point": "1,1048576", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "16,65536", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ]
}
```

(diff.patch: 12 lines, files: python/sglang/kernels/ops/attention/set_mla_kv_concat_q.py)

## iter_04
### hypothesis.md

VibeSim's highest-share actionable node is mxfp4_fused_moe
(R0=379.2 us, R6/R7=227.8 us, necessary_share=60.1%). The profiler maps
the production path to the SM100 TRT-LLM MXFP4 routed expert kernels. The
standard SiTU call currently passes tune_max_num_tokens=next_power_of_2(M),
which selects bucket 1 for the primary B=1 decode.

Use the existing bucket-8 tuning configuration for token counts below 8. This
does not change the MXFP8 activation values, expert weights, top-k IDs/weights,
activation type, or finalization; it only asks FlashInfer to select a
different precompiled/tuned launch configuration for the same operation.
B=16 and B=128 retain their existing buckets. Golden output and post-step
state checks will validate the configuration swap.

### analysis.md

VibeSim prediction: prediction_id p_c7687e585abe4b5fbacdc89caf0c9761,
fetched with simulate?prediction=.:iter_04. The fixed operator analysis still
shows mxfp4_fused_moe at 379.2 us (32.7%) and MLA decode attention at
358.8 us (31.0%). The top roofline node remains mxfp4_fused_moe
(R0=379.2 us, R5=8.6 us, R6/R7=227.8 us, necessary_share=60.1%); the MLA
decode leaf remains R0=358.8 us, R5=227.6 us, R6/R7=454.2 us,
necessary_share=126.6%.

The profiler confirmed the production SM100 TRT-LLM MXFP4 kernels launched,
but the bucket hint only exchanged the two B=1 expert GEMM timings and did not
reduce graph latency. The VibeSim kernels endpoint returned its placeholder
entry with has_cached_alternative=false; raw per-kernel tables are retained.

Decision: reject the bucket-8 hint and restore the original
next_power_of_2(token_count) tuning value.

### result.json

```
{
  "status": "rejected_no_primary_gain",
  "before_plain_graph_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 486.8159890174866,
    "16,65536": 298.46400022506714
  },
  "replay_graph_us": {
    "1,1048576": 261.6319954395294,
    "128,8192": 484.8960041999817,
    "16,65536": 298.43199253082275
  },
  "checks": [
    {"point": "1,1048576", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "16,65536", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ]
}
```

(diff.patch: 12 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_05
### hypothesis.md

VibeSim identifies MLA decode attention as a large necessary node
(R0=358.8 us, R6/R7=454.2 us, necessary_share=126.6%). The three required
driver points use a uniform context length within each batch, but the
TRT-LLM call currently inherits FlashInfer's default is_var_seq=True. The
installed FlashInfer implementation documents is_var_seq=False as the
persistent fixed-sequence schedule.

Pass is_var_seq=False on the ordinary non-DCP TRT-LLM decode call. This changes
only the attention scheduler/kernel variant; query, fp8 KV data, page table,
sequence lengths, scale, output layout, and all cache writes remain unchanged.
The DCP-specific CuteDSL call does not use this argument. The golden replay
will verify output and post-step state.

### analysis.md

VibeSim prediction: prediction_id p_c7687e585abe4b5fbacdc89caf0c9761,
fetched with simulate?prediction=.:iter_05. The attention leaf remains
R0=358.8 us, R5=227.6 us, R6/R7=454.2 us, necessary_share=126.6%;
mxfp4_fused_moe remains R0=379.2 us, R5=8.6 us, R6/R7=227.8 us,
necessary_share=60.1%. Operator shares remain 31.0% attention and 32.7%
fused MoE.

The profiler still launched the VarSeq-named TRT-LLM FMHA kernels after
is_var_seq=False, so the source hint did not expose a distinct faster kernel
for the tested graph. The VibeSim kernels endpoint returned its placeholder
with has_cached_alternative=false; raw tables are retained.

Decision: reject this change because the primary graph latency was unchanged.

### result.json

```
{
  "status": "rejected_no_primary_gain",
  "before_plain_graph_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 486.8159890174866,
    "16,65536": 298.46400022506714
  },
  "replay_graph_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 484.8639965057373,
    "16,65536": 297.40801453590393
  },
  "checks": [
    {"point": "1,1048576", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "16,65536", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ]
}
```

(diff.patch: 8 lines, files: python/sglang/srt/layers/attention/trtllm_mla_backend.py)

## iter_06
### hypothesis.md

The attention leaf is the largest measured necessary operator after the MoE
node (VibeSim R0=358.8 us, R5=227.6 us, necessary_share=126.6%), and the
profile maps it to FlashInfer's TRT-LLM FMHA kernel. The installed API supports
use_fp16_softmax for this backend. Enable it to reduce the softmax datapath
cost while leaving Q/K/V, page tables, scales, output shape, and KV state
unchanged. The replay golden check determines whether the reduced-precision
softmax stays within the required 0.02 relative-error bound.

### analysis.md

# Iteration 06 analysis

No new profile was obtained. The required smoke test rejected the candidate on
the target B200 (SM100): FlashInfer reported that `use_fp16_softmax` is only
supported on SM107/Rubin. Therefore there are no new timings to simulate or
analyze. The fixed VibeSim baseline prediction remains
`p_c7687e585abe4b5fbacdc89caf0c9761`; its attention leaf is still the measured
candidate area, but this API option cannot be used on the target architecture.

### result.json

```
{
  "iteration": 6,
  "status": "rejected_smoke_failed",
  "latency_us": {},
  "checks": {},
  "error": "ValueError: use_fp16_softmax is only supported on SM107 (Rubin); current device is sm100",
  "source_restored": true
}
```

(diff.patch: 8 lines, files: python/sglang/srt/layers/attention/trtllm_mla_backend.py)

## iter_07
### hypothesis.md

The measured B=1 long-context attention leaf is the largest device-time
consumer (the VibeSim MLA decode node is R0=358.8 us, R5=227.6 us, with
necessary_share=126.6%). The actual profile maps it to the TRT-LLM FMHA
decode kernel, which is launched with PDL enabled through `_ENABLE_PDL`.

Disable PDL for this decode call as an execution-scheduling experiment. It
does not change any tensor values, page tables, scales, workspace contents,
or KV-cache writes; it only changes kernel launch dependency handling. The
golden replay checks output and post-step state on every required shape.

### analysis.md

# Iteration 07 analysis

The profile was fetched/simulated as `p_c7687e585abe4b5fbacdc89caf0c9761`
using `.:iter_07`, followed by `analyze` at operator, run-summary, and
iteration levels and `optimality?scope=iter`. VibeSim's fixed before-state
prediction still identifies `unified.mla.attention.mla_decode_attention` as a
major leaf (R0=358.836 us, R5=227.559 us, R6/R7=454.164 us,
`necessary_share=1.265660`) and the fused MoE leaf as the largest operator.
The kernel endpoint returned no cached alternatives (`has_cached_alternative`
false), so the profiler table was used to confirm the actual TRT-LLM FMHA
launch.

With PDL disabled, the actual B=1 FMHA kernel measured 131.049 us versus the
baseline profile's approximately 129.7 us, and graph latency was 263.680 us
versus approximately 261.6 us. The candidate is rejected for performance;
the source has been restored to PDL enabled.

### result.json

```
{
  "iteration": 7,
  "status": "rejected_no_primary_gain",
  "before_us": {
    "B1_L1048576": 261.59998774528503,
    "B128_L8192": 486.8159890174866,
    "B16_L65536": 298.46400022506714
  },
  "after_us": {
    "B1_L1048576": 263.68001103401184,
    "B128_L8192": 485.727995634079,
    "B16_L65536": 298.4960079193115
  },
  "checks": {
    "B1_L1048576": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B128_L8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B16_L65536": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "source_restored": true
}
```

(diff.patch: 10 lines, files: python/sglang/srt/layers/attention/trtllm_mla_backend.py)

## iter_08
### hypothesis.md

The profiler confirms two small-batch TGV projection leaves at about 13 us
each. VibeSim's remaining attention-side headroom is limited after the fused
front work, so this iteration tests the Kimi-K3 latent-up shape directly:
`m=1, n=7168, k=3584`.

Select TGV tactic 20 (`cta64x32`, two-CTA, eight stages) for that exact shape.
It uses the same BF16 inputs, FP32 accumulation, BF16 output, and PDL path as
the existing tactic; only the output tile/staging configuration changes. The
golden replay must verify that the layer output and all post-step buffers stay
within the required tolerance.

### analysis.md

# Iteration 08 analysis

The profile was simulated as `p_c7687e585abe4b5fbacdc89caf0c9761` with
`.:iter_08`, then passed through operator, run-summary, iteration, optimality,
and kernel analyses. VibeSim's fixed before-state ladder still reports the
MXFP4 fused MoE at R0=379.160 us with `necessary_share=0.600686`, and MLA
decode at R0=358.836 us with R5=227.559 us and
`necessary_share=1.265660`. The kernel endpoint reports no cached alternative;
the profiler table confirms the two small-batch TGV projection launches.

The candidate changed the latent-up tactic, but B=1 graph latency was
261.600 us, equal to the baseline. The post-edit table shows one projection at
10.56 us and the other at 15.79 us, replacing approximately two 13.1 us
launches. This is a scheduling/tile trade with no critical-path gain, so the
candidate was rejected and the heuristic restored.

### result.json

```
{
  "iteration": 8,
  "status": "rejected_no_primary_gain",
  "before_us": {
    "B1_L1048576": 261.59998774528503,
    "B128_L8192": 486.8159890174866,
    "B16_L65536": 298.46400022506714
  },
  "after_us": {
    "B1_L1048576": 261.6640031337738,
    "B128_L8192": 484.8960041999817,
    "B16_L65536": 298.46400022506714
  },
  "checks": {
    "B1_L1048576": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B128_L8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B16_L65536": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "source_restored": true
}
```

(diff.patch: 9 lines, files: python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py)

## iter_09
### hypothesis.md

VibeSim ranks `unified.mla.moe.mxfp4_fused_moe` first by headroom
(R0=379.160 us, R5=8.573 us, R6/R7=227.756 us,
`necessary_share=0.600686`), and the profile confirms the TRT-LLM MXFP4
expert BMMs are the largest MoE launches. Their call currently enables PDL for
the one-token decode batch. Disable MoE PDL to test whether the dependency
setup/launch path is costing more than it hides at this shape.

The toggle changes only scheduling; routing ids/weights, MXFP8 activations,
MXFP4 weights/scales, accumulation, output buffers, and post-step state are
unchanged. The golden replay checks that assumption on all required points.

### analysis.md

# Iteration 09 analysis

The profile was simulated as `p_c7687e585abe4b5fbacdc89caf0c9761` with
`.:iter_09`; operator, run-summary, iteration, optimality, and kernel analyses
were then run. The fixed VibeSim result still ranks the MXFP4 fused MoE first
(R0=379.160 us, R5=8.573 us, R6/R7=227.756 us,
`necessary_share=0.600686`). The kernel endpoint has no cached alternative,
and the profiler table shows the actual TRT-LLM MXFP4 launches.

Disabling MoE PDL increased graph latency to 266.720 us at B=1, 489.920 us at
B=128, and 307.776 us at B=16, despite exact replay checks. The default PDL
policy was restored.

### result.json

```
{
  "iteration": 9,
  "status": "rejected_no_primary_gain",
  "before_us": {
    "B1_L1048576": 261.59998774528503,
    "B128_L8192": 486.8159890174866,
    "B16_L65536": 298.46400022506714
  },
  "after_us": {
    "B1_L1048576": 267.61600375175476,
    "B128_L8192": 493.120014667511,
    "B16_L65536": 306.68801069259644
  },
  "checks": {
    "B1_L1048576": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B128_L8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B16_L65536": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "source_restored": true
}
```

(diff.patch: 9 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_10
### hypothesis.md

The B=1 profile's largest non-attention launch is the merged-front TGV GEMM
(about 38 us), followed by two TGV projection launches. They are all compiled
with PDL enabled. Test PDL-disabled TGV execution for the small K3 decode GEMM
path to see whether the dependency machinery costs more than it hides in this
CUDA graph.

Only the launch dependency mode changes. Tile sizes, BF16 inputs/outputs,
FP32 accumulation, weights, and all tensor writes remain identical; replay
against the golden output and post-step state is mandatory.

### analysis.md

# Iteration 10 analysis

The post-edit profile was fetched with `.:iter_10` and simulated as
`p_c7687e585abe4b5fbacdc89caf0c9761`; operator, run-summary, iteration,
optimality, and kernel analyses were run afterward. VibeSim continues to rank
the MXFP4 fused MoE first (R0=379.160 us, `necessary_share=0.600686`) and
MLA decode second (R0=358.836 us, `necessary_share=1.265660`). The kernel
endpoint has no cached alternatives; the profiler confirmed the TGV launches.

Disabling PDL on the TGV GEMMs raised the B=1 graph replay to 267.680 us and
also regressed B=16. All checks were exact. Both TGV calls were restored to
PDL enabled.

### result.json

```
{
  "iteration": 10,
  "status": "rejected_no_primary_gain",
  "before_us": {
    "B1_L1048576": 261.59998774528503,
    "B128_L8192": 486.8159890174866,
    "B16_L65536": 298.46400022506714
  },
  "after_us": {
    "B1_L1048576": 267.67998933792114,
    "B128_L8192": 485.8880043029785,
    "B16_L65536": 302.5279939174652
  },
  "checks": {
    "B1_L1048576": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B128_L8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B16_L65536": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "source_restored": true
}
```

(diff.patch: 11 lines, files: python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py)

## iter_11
### hypothesis.md

The profiler's attention leaf is the TRT-LLM SM100 FMHA kernel, while
`TRTLLMMLABackend` omits the `backend` keyword when its configured backend is
`"trtllm-gen"`, relying on FlashInfer's `"auto"` dispatch. Pass the existing
backend explicitly so the decode call selects the measured TRT-LLM path
directly.

This is a dispatch-only change: query/KV tensors, page tables, scales,
softmax, output dtype, and KV writes are unchanged. The replay golden checks
the output and post-step state for every required point.

### analysis.md

# Iteration 11 analysis

The post-edit profile was simulated as
`p_c7687e585abe4b5fbacdc89caf0c9761` using `.:iter_11`, followed by operator,
run-summary, iteration, optimality, and kernel analyses. VibeSim remains
unchanged: MXFP4 fused MoE is R0=379.160 us with
`necessary_share=0.600686`; MLA decode is R0=358.836 us with
`necessary_share=1.265660`. The kernel endpoint reports no cached alternative.

Explicit `backend="trtllm-gen"` selected the same FMHA kernel and produced no
primary improvement: B=1 was 261.600 us. The dispatch expression was restored
to the original auto-compatible form.

### result.json

```
{
  "iteration": 11,
  "status": "rejected_no_primary_gain",
  "before_us": {
    "B1_L1048576": 261.59998774528503,
    "B128_L8192": 486.8159890174866,
    "B16_L65536": 298.46400022506714
  },
  "after_us": {
    "B1_L1048576": 261.6319954395294,
    "B128_L8192": 484.8960041999817,
    "B16_L65536": 298.43199253082275
  },
  "checks": {
    "B1_L1048576": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B128_L8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B16_L65536": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "source_restored": true
}
```

(diff.patch: 7 lines, files: python/sglang/srt/layers/attention/trtllm_mla_backend.py)
