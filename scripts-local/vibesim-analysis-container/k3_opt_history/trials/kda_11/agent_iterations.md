# KDA trial 11: agent iterations

## iter_00
### hypothesis.md

Hypothesis

The KDA decode backend excludes packed KDA decode whenever lower_bound is
set, even though fused_recurrent_kda_packed_decode implements the same safe
gate equation. Removing that exclusion should replace the generic
fused_sigmoid_gating_delta_rule_update launch with the KDA-specialized packed
kernel. The custom row-streaming CUDA alternative does not implement the safe
gate, so it must remain disabled when lower_bound is present.

The change preserves the safe-gate formula, BF16 recurrent state writeback,
Q/K normalization, beta sigmoid, cache indexing, and output layout. Only the
kernel dispatch changes; replay CHECK results must validate both output and
post-step conv/temporal state at all three points.

### analysis.md

Baseline profile and VibeSim analysis

- Prediction: p_38ccb83a6c4e44298f45b2251bd4cf54 (simulate prediction=.:before,
  run k3_kda, NVIDIA B200, 3 cases).
- Plain fixed-command graph latency: B128 403.008 us, B32 263.648 us,
  B1 138.752 us. The diagnostic profile used for this iteration had 26, 27,
  and 25 launches and graph latencies 370.208, 246.304, and 139.744 us.
- VibeSim run summary: 0.782952 ms modeled kernel time; B128 iteration is
  modeled as 417 us. The modeled B128 breakdown is attention 69 us and MoE
  348 us; mxfp4_fused_moe is 265 us (63.5%), merged_front 46 us (11.0%),
  and kda_recurrent_decode 30 us (7.2%).
- Optimality: mxfp4_fused_moe is the largest leaf, R0=412.401 us and
  R5=9.454 us (R0/R5=43.62), but kernels() reports only the
  sglang_trtllm_mxfp4 implementation and has_cached_alternative=false.
  The recurrent leaf has R0=42.495 us, R5=8.161 us (R0/R5=5.21), and is
  an editable sglang_triton path. The run-summary buckets identify batching
  as 29.0% and hardware gap as 61.1%; no excess-over-necessary or fusion
  bucket is modeled.
- Profiler confirmation for B128: the largest launches are the two MXFP4
  expert BMMs (125.510 and 66.412 us), followed by
  nvjet_sm100_tss (51.417 us), fused_sigmoid_gating_delta_rule_update_kernel
  (29.539 us), and the remaining KDA/MoE kernels. The 29.539 us launch is
  the KDA recurrent node identified above.

The VibeSim mxfp4 node is larger but has no cached alternative and is a
vendor kernel. This iteration targets the next editable node and its safe
gate dispatch, rather than changing routing or arithmetic.

(diff.patch: 30 lines, files: python/sglang/kernels/ops/attention/fla/fused_recurrent.py, python/sglang/srt/layers/attention/linear/kda_backend.py)

## iter_01
### hypothesis.md

Hypothesis

The row-streaming CUDA KDA packed decoder already accepts lower_bound and
use_lower_bound, and its device code implements
lower_bound * sigmoid(exp(A_log) * (a + dt_bias)). The Python caller passes
these values, but fused_recurrent_kda_packed_decode blocks the CUDA call when
lower_bound is non-None. Removing that stale condition should select the
CUDA row-streaming kernel for the fixed KDA safe-gate workload.

This changes only scheduling and implementation of the same fp32 recurrence;
the CUDA kernel updates the same BF16/FP32 state pool at the same indices and
writes the same BF16 output layout. Replay CHECK must validate output and
post-step state at B128, B32, and B1.

### analysis.md

Iteration 00 post-edit profile and VibeSim re-analysis

- Re-profiled the edited tree with the fixed diagnostic command. Graph
  latencies were B128 361.984 us, B32 242.176 us, and B1 137.696 us;
  profile sums were 411.698, 283.354, and 167.959 us.
- The VibeSim fetch was re-run and returned the fixed before-state prediction
  p_38ccb83a6c4e44298f45b2251bd4cf54. Its analysis is unchanged by the
  service contract: mxfp4_fused_moe remains the largest node, with
  R0/R5=43.62 and has_cached_alternative=false; merged_front is next, and
  kda_recurrent_decode is the editable leaf with R0/R5=5.21.
- Profiler confirmation after iteration 00: the recurrent launch changed to
  fused_recurrent_kda_packed_decode_kernel at 18.835 us for B128 (7.181 us
  at B32 and 4.214 us at B1). The MXFP4 BMMs remained 125.132 and 66.227 us,
  and the merged-front nvjet launch remained 51.533 us.

The current packed Triton route is faster and exact, but the JIT CUDA source
already supports the same safe lower-bound arguments. This iteration targets
that remaining dispatch restriction.

(diff.patch: 15 lines, files: python/sglang/kernels/ops/attention/fla/fused_recurrent.py)

## iter_02
### hypothesis.md

Hypothesis

The active fused_recurrent_kda_packed_decode_kernel uses BK=128, BV=32 and
num_warps=1 for every batch. At B128 the grid has 128*12*4 programs, so more
warps per program may distribute the 4096-element recurrent tile and reduce
register pressure or issue latency. Change only num_warps from 1 to 4 in the
packed Triton launch.

The program still performs the same safe-gate, normalization, beta, delta-rule
and BF16 state-store operations. Triton launch geometry does not change tensor
indices or state ownership; replay CHECK validates the allowed numerical
tolerance and post-step state for all required points.

### analysis.md

Iteration 01 post-edit profile and VibeSim re-analysis

- Fixed diagnostic profile: graph latency B128 361.984 us, B32 242.144 us,
  B1 137.728 us; kernel sums 411.992, 283.088, and 168.544 us.
- VibeSim was fetched again as prediction
  p_38ccb83a6c4e44298f45b2251bd4cf54 and all analysis verbs were rerun. The
  fixed prediction still ranks mxfp4_fused_moe first (R0/R5=43.62), then
  merged_front; kda_recurrent_decode remains the only profiled editable
  recurrent backend and has no cached alternative.
- The profiler still shows the active KDA leaf as
  fused_recurrent_kda_packed_decode_kernel: 18.835 us at B128, 7.181 us at
  B32, and 4.214 us at B1. The iteration 01 CUDA dispatch guard is inert for
  this workload because its CUDA wrapper requires FP32 state while the fixed
  state is BF16.

The next edit stays on the measured recurrent leaf and changes only Triton
launch geometry; the B128 kernel is a large [128, 32] FP32 tile launched with
one warp.

(diff.patch: 11 lines, files: python/sglang/kernels/ops/attention/fla/fused_recurrent.py)

## iter_03
### hypothesis.md

Hypothesis

Two warps may reduce the register/issue pressure of the [BV=32, BK=128]
packed KDA tile without the four-warp scheduling cost. The safe-gate math,
state indices, stores, and output layout remain unchanged; CHECK is required
at every workload point.

### analysis.md

Iteration 02 post-edit profile and VibeSim re-analysis

- The four-warp probe was profiled through the required replay: B128 was
  398.016 us, B32 259.616 us, and B1 136.704 us. CHECK passed at all
  points, but B128 regressed from the one-warp reference and had max_rel_err
  0.011555.
- VibeSim prediction p_38ccb83a6c4e44298f45b2251bd4cf54 was fetched and all
  analysis verbs were rerun. It still identifies mxfp4_fused_moe as the
  largest node (R0/R5=43.62, no cached alternative), with the measured KDA
  recurrent leaf as the best editable kernel family.
- The four-warp source edit was reverted after the smoke test. The current
  active kernel is again fused_recurrent_kda_packed_decode_kernel with the
  one-warp launch.

This iteration tests two warps as the remaining nearby launch configuration.

(diff.patch: 11 lines, files: python/sglang/kernels/ops/attention/fla/fused_recurrent.py)

## iter_04
### hypothesis.md

Hypothesis

The active FlashInfer TRT-LLM routed MXFP4 call uses
next_power_of_2(num_tokens) as tune_max_num_tokens. For the scored B128
decode, that ceiling is exactly 128; selecting the next tuning bucket (256)
may choose a better B128 tactic/launch schedule on B200. Apply the 256
ceiling only when num_tokens is at least 128, so B32 and B1 retain their
baseline kernels.

The MoE kernel still receives identical BF16/FP8-quantized activations,
weights, routing ids/weights, activation constants, and output buffer. This
is an implementation/tactic selection change, so replay CHECK must verify
the output and recurrent state at all points.

### analysis.md

Iteration 03 post-edit profile and VibeSim re-analysis

- The two-warp packed-KDA probe produced B128 396.832 us, B32 259.584 us,
  and B1 136.704 us, with CHECK passing at every point. It did not improve
  the one-warp reference, so the source was restored to one warp.
- VibeSim prediction p_38ccb83a6c4e44298f45b2251bd4cf54 was fetched and
  analyzed again. The mxfp4_fused_moe leaf remains the largest modeled node
  (R0/R5=43.62, no cached alternative), and the actual profile confirms its
  two B128 launches are 125.1 and 66.2 us.

The next experiment changes only the TRT-LLM MXFP4 tuning ceiling for the
B128 routed-MoE call, keeping B32/B1 on their existing buckets.

(diff.patch: 15 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_05
### hypothesis.md

Hypothesis

In KimiK3MoE._forward_fused, the shared gate-up activation and the routed
latent input are disjoint slices of the completed merged-front GEMM. For the
single-rank driver shape with TP1 and K3 AR fusion disabled, shared_down can
run on the existing MoE side stream while the routed MXFP4 path runs on the
current stream. Waiting on an event before the flat output reduction preserves
the exact data dependency and output ordering while hiding shared_down under
the dominant routed work.

The condition excludes TP>1 and AR-fusion paths, so collective ordering and
production multi-rank behavior are unchanged. No tensor values or state
updates are altered.

### analysis.md

Iteration 04 post-edit profile and VibeSim re-analysis

- The MXFP4 tuning-ceiling edit was profiled with the fixed diagnostic command:
  graph latencies were B128 362.048 us, B32 242.176 us, and B1 137.728 us.
  Replay remained exact at all points; the plain replay B128 result was
  395.808 us, only a noise-sized change from the one-warp reference.
- VibeSim prediction p_38ccb83a6c4e44298f45b2251bd4cf54 was simulated and
  analyzed again. The operator/optimality output still ranks
  mxfp4_fused_moe first (R0/R5=43.62, has_cached_alternative=false), with
  merged_front and shared_down as the next measured MoE leaves.
- The profiler still launches the same two MXFP4 BMMs and the same merged
  front/shared-down path; changing the autotuner ceiling did not move the
  kernel family or provide a meaningful latency reduction.

The next target is the serial dependency between the independent shared-down
GEMM and routed MXFP4 work in the TP1 driver shape.

## iter_06
### hypothesis.md

Hypothesis

Overlap shared_down with routed MXFP4 MoE on the existing side stream only
for TP1 and disabled K3 AR fusion. Both consume the completed merged-front
output, write disjoint slices, and are joined before any consumer can read the
flat latent/shared buffer. Numerical output, recurrent state, and all
multi-rank paths remain unchanged.

### analysis.md

Iteration 05 target selection

- The re-profiled tree after the MXFP4 tuning probe still launches the same
  dominant MXFP4 BMM pair and the serial shared_down GEMM.
- VibeSim prediction p_38ccb83a6c4e44298f45b2251bd4cf54 remains the fixed
  B200 KDA prediction. Its optimality ladder continues to identify
  mxfp4_fused_moe as the largest node and shared_down as a sizeable batching
  leaf; neither has a cached alternative.
- The next source-level change is a stream schedule for the already-present
  shared expert computation, with an event join before the output reduction.

(diff.patch: 31 lines, files: python/sglang/srt/models/kimi_k3.py)
