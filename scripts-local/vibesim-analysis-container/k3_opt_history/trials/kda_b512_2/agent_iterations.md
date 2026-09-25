# KDA_B512 trial 2: agent iterations

## agent log.md

iter_00: baseline; plain graph 677.344/482.816/378.464 us (B512/B256/B128); VibeSim ranked mxfp4_fused_moe first; golden captured.
iter_01: rejected large-batch BF16 MXFP4 activation path; B512/B256 had no TRT-LLM kernel, B128 exact; source restored and smoke passed.
iter_02: rejected route+pack+quant coverage extension; exact checks, no route_quant_fused launch, primary 677.344->679.392 us; source restored and smoke passed.
iter_03: rejected BF16 KDA 3-stage TMA at B>=256; exact checks, 677.344/482.816/378.464 -> 679.456/481.760/380.512 us; source restored and smoke passed.
iter_04: rejected routed-row MXFP4 tuning ceiling; exact checks, same TRT-LLM GEMMs, plain 677.376/484.864/380.416 -> 677.376/481.824/378.656 us; source restored and smoke passed.

## iter_01
### hypothesis.md

Iteration 01 hypothesis

Use the existing TRT-LLM MXFP4-weight / BF16-activation path for the K3 SM100
large-batch decode shapes (B>=256), while retaining MXFP8 activation
quantization for smaller decode batches. The warm-start history rejected this
path at B=1 because the installed TRT-LLM cubin had no matching SiTU tactic;
the new workload specifically makes B=256 and B=512 compute-bound enough that
the large-M tactic set must be re-measured.

The edit changes only the activation representation handed to the fused MoE
operator. It does not change the BF16 input tensor values before the branch,
router logits or top-k IDs/weights, packed MXFP4 weights/scales, SiTU
parameters, output allocation, KDA recurrent state, convolution state, or KV
cache writes. The golden replay is required to verify the numerical contract;
the B=128 guard preserves the already accepted small-batch path.

Prior trials used: `/workspace/opt_history/trials/kda_7` and
`/workspace/opt_history/trials/kda_25` rejected BF16 activation at small
decode M because the installed runner had no tactic, while
`/workspace/opt_history/TECHNIQUES.md` explicitly leaves activation format
open once large batches make MXFP4 compute-bound. This iteration re-tests that
dead end only at the new large-batch points.

### analysis.md

Iteration 01 analysis

VibeSim prediction: `p_1066e0f8b4d148de9d40979e8ca7f1e5`, built with
`simulate?prediction=.:before` for `k3_kda_b512`.

The required analyses ranked `unified.kda.moe.mxfp4_fused_moe` first. Across
the three decode cases, operator analysis assigns it 0.9613 ms / 57.15% of
predicted kernel time. Its optimality row is R0=961.328 us, R5=52.613 us,
R0/R5=18.27, with the fixed service leaving R6 and `necessary_share` null.
The run summary attributes 62.91% of the modeled excess to batching, and lists
this node as the largest batching contributor. The kernel drill-down identifies
the `sglang_trtllm_mxfp4` backend, shape 3584 hidden / 3072 intermediate / 112
local experts / top-2 / MXFP4 E2M1-UE8M0, with no cached alternative.

The real B=512 profile confirms the selected source path: the two fused
TRT-LLM MXFP4 BMM launches are 255.1 us and 117.3 us, while the standalone
per-token-group MXFP8 activation quantizer is 6.3 us. The mapped source is
`Mxfp4MoEMethod._apply_sm100_trtllm_gen()` in
`python/sglang/srt/layers/quantization/mxfp4.py`, which calls
`_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen()` in
`python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`.

The candidate targets only B>=256 in that runner. B=128 remains on the
existing default MXFP8 path so the known small-batch BF16-tactic failure is
not engaged by the mandatory smoke test.

Result: rejected. The first attempt reached the branch with FP32 routed input,
so it was corrected to cast the existing front output to BF16. The mandatory
smoke still passed, but replay then failed at both B=512 and B=256 with the
installed TRT-LLM error `No kernel found for mDtypeA: MxE2m1,
mDtypeB: Bfloat16, mDtypeC: Bfloat16` (fused SiTU). B=128 stayed exact on the
original path. The source was restored; there is no valid post-edit kernel
profile because the candidate fails before graph capture.

### result.json

```
{
  "iteration": 1,
  "status": "rejected",
  "reason": "TRT-LLM SM100 has no MXFP4-weight x BF16-activation fused SiTU kernel for the K3 shape at B=256 or B=512",
  "smoke_after_edit": {"point": "1,8192", "latency_us": 126.3, "pass": true},
  "replay": {
    "512,8192": {"ok": false, "error": "No kernel found for mDtypeA=MxE2m1, mDtypeB=Bfloat16, mDtypeC=Bfloat16"},
    "256,8192": {"ok": false, "error": "No kernel found for mDtypeA=MxE2m1, mDtypeB=Bfloat16, mDtypeC=Bfloat16"},
    "128,8192": {"latency_us": 378.368, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "smoke_after_restore": {"point": "1,8192", "latency_us": 126.5, "pass": true}
}
```

(diff.patch: 12 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_02
### hypothesis.md

Iteration 02 hypothesis

Raise the K3 fused route+pack+MXFP8-quant coverage from 64 to 512 tokens in
`python/sglang/kernels/ops/moe/moe_route_quant_fused.py`. This enables the
existing `route_quant_fused_kernel` at B=128, B=256, and B=512, replacing the
separate route radix, packed-ID, and per-token-group quant launches. The
large-batch profile shows those launches on the critical path immediately
before the dominant TRT-LLM MXFP4 BMM pair, so removing their launch and idle
SM gaps may lower graph replay latency without changing the BMM tactic.

The kernel already computes the same route radix selection, BF16/FP32 score
handling, packed top-k representation, MXFP8 E4M3 values, and UE8M0 scales;
only the launch covers more rows. It writes fresh routing and quantized output
buffers, so no KDA convolution/recurrent state or MLA KV row is touched. The
golden replay must prove output and post-step state equivalence at all points.

Prior work used: `/workspace/opt_history/TECHNIQUES.md` lever A9 and the
existing accepted route+quant implementation; `/workspace/opt_history/trials/kda_21`
rejected a different 112-expert/top-2 route specialization because it broke
small-batch numerics. This candidate keeps the existing 896-expert/top-16
kernel and exact output contract, changing only its batch coverage for the new
large-batch workload.

### analysis.md

Iteration 02 analysis

This iteration builds on the same VibeSim prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5` from `simulate?prediction=.:before`.
The required operator, run-summary, iteration, optimality, and kernel analyses
rank `unified.kda.moe.mxfp4_fused_moe` first: 961.328 us / 57.15% predicted
kernel share, R0=961.328 us, R5=52.613 us, R0/R5=18.27, and no cached
alternative. The real B=512 profile confirms the two fused MXFP4 BMMs at
255.1 us and 117.3 us, followed by small routing/quantization prep launches
of 6.5 us, 6.3 us, and about 4.8 us.

The BF16 activation candidate in iteration 01 was ruled out by the runtime's
missing tactic, so this iteration targets the remaining in-scope launches
around the same leaf. `moe_route_quant_fused.py` already contains a
correctness-preserving fused replacement for route radix + packed-ID creation
+ MXFP8 activation quantization, but its `_MAX_TOKENS=64` guard excludes every
new scored point. The source kernel is specialized to the exact K3 shape
(896 routing scores, top-16, 3584-wide rows) and its module documentation says
the produced IDs, weights, packed IDs, MXFP8 values, and scales are bit
identical to the three unfused launches.

The candidate extends only the coverage guard to the largest scored batch.
The profiler after the edit will confirm whether `route_quant_fused_kernel`
actually replaces the three baseline launches; VibeSim remains the basis for
selecting the dominant MoE leaf, while the measured table decides whether this
small schedule optimization clears the 0.5% speed threshold.

Post-edit result: rejected. The exact replay passed all three points with
`max_rel_err=0`, `state_ok=true`, but B=512 measured 679.392 us versus the
677.344 us baseline; B=256 and B=128 were unchanged. The diagnostic profile
still launched `routingCustom`, `per_token_group_quant_flat`, and
`_router_triton_kernel`; `route_quant_fused_kernel` was absent. A temporary
trace showed `_route_quant_fuse_eligible=True`, but the handoff helper was not
called by the live K3 grouped-topk dispatch, so increasing its coverage cannot
affect this layer. The trace and constant change were reverted, and the
restored smoke passed.

Post-profile VibeSim was rebuilt/fetched with the same prediction id and all
analysis verbs. It still ranks `unified.kda.moe.mxfp4_fused_moe` first at
961.328 us / 57.15%, R0/R5=18.27, no cached alternative; the fixed model's
worst batching node is unchanged.

### result.json

```
{
  "iteration": 2,
  "status": "rejected",
  "reason": "The existing route+quant handoff is not called by this K3 grouped-topk path; the coverage edit leaves the kernel table unchanged and the primary replay regresses",
  "candidate_latency_us": {
    "512,8192": 679.3919801712036,
    "256,8192": 482.81601071357727,
    "128,8192": 378.464013338089
  },
  "baseline_latency_us": {
    "512,8192": 677.344024181366,
    "256,8192": 482.81601071357727,
    "128,8192": 378.464013338089
  },
  "checks": {
    "512,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "profile": {
    "512,8192": "routingCustom + per_token_group_quant_flat + _router_triton_kernel; no route_quant_fused_kernel",
    "256,8192": "same unfused prep launches",
    "128,8192": "same unfused prep launches"
  },
  "smoke_after_restore": {"point": "1,8192", "pass": true}
}
```

(diff.patch: 5 lines, files: python/sglang/kernels/ops/moe/moe_route_quant_fused.py)

## iter_03
### hypothesis.md

Iteration 03 hypothesis

For aligned BF16 recurrent state, select `tma_stages=3` only when
`B * H >= 2048` in the KDA fused decode host wrapper. With K3 H=12 this
targets B=256 and B=512, reducing shared staging from four 32-row slabs / 64
KiB to three / 48 KiB and potentially increasing resident blocks at the large
decode grids. The kernel's row traversal, BF16 rounding, state stores,
convolution-state update, output normalization, and PDL dependency are
unchanged; only which preloaded stage buffer is reused changes.

B=128 remains on four stages because the warm-start trials `kda_24` and
`kda_25` found no gain from changing the old primary point. The prior
large-batch workload did not include B=256/B=512, so this is a scoped
re-measurement rather than assuming the earlier result transfers.

### analysis.md

Iteration 03 analysis

The fixed VibeSim prediction remains `p_1066e0f8b4d148de9d40979e8ca7f1e5`
(`k3_kda_b512`), rebuilt with `simulate?prediction=.:before`. Operator
analysis assigns `unified.kda.moe.mxfp4_fused_moe` 961.328 us / 57.15% and
the recurrent node 200.253 us / 11.90% of modeled kernel time. Optimality
places the recurrent leaf second at R0=200.253 us, R5=45.419 us,
R0/R5=4.41, with no cached alternative in the kernel drill-down. The real
B=512 profile confirms the selected BF16 fused KDA launch at 92.54 us; B=256
and B=128 launch the same template at 48.91 us and 27.43 us.

The current host dispatch in
`python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh` selects four
TMA stages for every BF16 state, using 64 KiB dynamic shared memory. Prior
trials `/workspace/opt_history/trials/kda_24` and `kda_25` rejected global
3-stage/2-chunk variants at the old B=128 point. This iteration isolates the
new large-grid regime: B>=256 uses the already-compiled 3-stage BF16 template
(48 KiB), while B=128 keeps the accepted 4-stage template. The post-edit
profile must verify the template names and the replay must verify all output
and state checks.

Post-edit result: the candidate did select the intended BF16 three-stage
template for B=256 and B=512, while B=128 retained four stages. Replay checks
passed with unchanged output and post-step state at every point, but graph
latency changed from 677.344/482.816/378.464 us to
679.456/481.760/380.512 us (B=512/B=256/B=128). The B=512 diagnostic profile
also showed the recurrent kernel at 838.798 us, versus 92.541 us for the
four-stage baseline, so the candidate is rejected and the source was restored.
The fixed VibeSim prediction and its MXFP4-first optimality ranking were
unchanged; no cached recurrent alternative was exposed.

### result.json

```
{
  "iteration": 3,
  "candidate": {
    "change": "BF16 KDA TMA stages: 3 when B*H >= 2048, otherwise 4",
    "latency_us": {
      "512,8192": 679.4559955596924,
      "256,8192": 481.7599952220917,
      "128,8192": 380.511999130249
    },
    "checks": {
      "512,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
      "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
      "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
    },
    "profile": {
      "512,8192": {"kda_template_stages": 3, "kda_kernel_us": 838.7978},
      "256,8192": {"kda_template_stages": 3, "kda_kernel_us": 48.544},
      "128,8192": {"kda_template_stages": 4, "kda_kernel_us": 26.992}
    }
  },
  "baseline_latency_us": {
    "512,8192": 677.344024181366,
    "256,8192": 482.81601071357727,
    "128,8192": 378.464013338089
  },
  "accepted": false,
  "reason": "B=512 regressed and the diagnostic B=512 recurrent kernel was anomalously slow; source restored."
}
```

(diff.patch: 10 lines, files: b/python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh)

## iter_04
### hypothesis.md

Iteration 04 hypothesis

The fixed VibeSim analysis and the B=512 kernel table identify the routed
MXFP4 MoE as the dominant node: its two routed GEMMs account for about 372 us
of the diagnostic B=512 kernel time and the simulated node has R0/R5=18.27.
The live standard routed call currently passes
`tune_max_num_tokens=next_power_of_2(x_quant.shape[0])` to FlashInfer. With
top_k=2, B=512 produces about 1024 expanded local rows, so the tuning ceiling
may be selecting a tactic for half the effective GEMM M. This re-evaluates the
warm-start dead end from `/workspace/opt_history/trials/kda_25` (B=128 -> 256
had no effect) and the analogous `/workspace/opt_history/trials/mla_4`
trial at the new large-batch regime.

For decode-sized standard routed inputs with at least 256 rows, pass the next
power of two of `num_tokens * routed_top_k`; retain the existing ceiling for
B=128 and smaller points. This changes only FlashInfer tactic/workspace
selection. Routing IDs and weights, quantized activations, MXFP4 weights,
activation type, output buffer, and all arithmetic inputs remain identical, so
the layer output and KDA/MLA state should be unchanged.

### analysis.md

Iteration 04 analysis

Baseline profile: `/workspace/opt_run/iter_03/profile.json` and the original
fixed profile show the same live launches. VibeSim was rebuilt with
`simulate?prediction=.:before` as prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5` (`k3_kda_b512`). Operator analysis assigns
`unified.kda.moe.mxfp4_fused_moe` 961.328 us / 57.15% of modeled kernel time.
Its optimality ladder is R0=961.328 us, R5=52.613 us, R0/R5=18.27, with no
cached alternative; the B=512 diagnostic table confirms the routed GEMM1 and
GEMM2 launches are the largest real kernels. This makes the FlashInfer
`tune_max_num_tokens` call a workload-specific candidate while retaining the
accepted KDA and shared-expert changes.

Post-edit VibeSim rebuilt/fetched the same prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5`; operator, run-summary, iteration,
optimality, and MXFP4 kernel-drilldown outputs were unchanged. The real
profile still launched the same MXFP4 GEMM1/GEMM2 kernels at every point, with
no new tactic or cached alternative. Replay checks were exact at all points.
The plain before/after replay was 677.376/484.864/380.416 us ->
677.376/481.824/378.656 us, but the diagnostic kernel table and graph timing
did not show a consistent large-batch change; the candidate is rejected and
the source was restored. The recorded before/after values are for this
iteration's identical plain command; the split/profile run is diagnostic only.

### result.json

```
{
  "iteration": 4,
  "candidate": {
    "change": "standard routed MXFP4 tuning ceiling uses tokens * routed_top_k for B>=256",
    "latency_us_before": {
      "512,8192": 677.3759722709656,
      "256,8192": 484.8639965057373,
      "128,8192": 380.41600584983826
    },
    "latency_us_after": {
      "512,8192": 677.3759722709656,
      "256,8192": 481.82401061058044,
      "128,8192": 378.6559998989105
    },
    "checks": {
      "512,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
      "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
      "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
    },
    "profile": {
      "512,8192": {"gemm1_us": 259.0842, "gemm2_us": 119.8176, "latency_us": 672.2240},
      "256,8192": {"gemm1_us": 183.6251, "gemm2_us": 94.7325, "latency_us": 491.1360},
      "128,8192": {"gemm1_us": 133.0428, "gemm2_us": 66.2684, "latency_us": 346.5920}
    }
  },
  "accepted": false,
  "reason": "The profile launched the same kernels and showed no reproducible primary-point gain; source restored."
}
```

(diff.patch: 10 lines, files: b/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_05
### hypothesis.md

Iteration 05 hypothesis

The dominant VibeSim node is still `unified.kda.moe.mxfp4_fused_moe`
(961.328 us / 57.15%, R0/R5=18.27, no cached alternative). The B=512
profile shows three independent preparation launches before its TRT-LLM
MXFP4 GEMM pair: routing, packed-ID preparation, and per-token-group MXFP8
quantization. The current handoff is specialized for 896 experts/top-16 and
does not cover this single-rank K3 harness shape of 112 experts/top-2.

This builds on `/workspace/opt_history/trials/kda_21` and its shared
implementation from `mla_20`, but narrows the prior rejected idea to the new
large decode regime. Add a 112-expert/top-2 route+quant kernel using the same
QuantTrait conversion, and invoke it only while the CUDA graph is being
captured for exactly B=256 or B=512. B=128 remains on the inherited router and
all eager execution, including chunked prefill, remains on the inherited
route/pack/quant chain. The route output, packed BF16 weights, MXFP8 values,
and UE8M0 scales are produced in the same formats consumed by FlashInfer; no
KDA recurrent/conv state or MLA KV state is touched. Replay against the
original golden checks whether the prior small-batch routing mismatch is
absent at these large graph shapes.

### analysis.md

Iteration 05 analysis

The current-tree baseline profile is `/workspace/opt_run/iter_04/profile.json`
and the plain pre-edit measurement is `/workspace/opt_run/iter_05/before.json`:
677.120/482.880/378.432 us at B=512/B=256/B=128. VibeSim was rebuilt with
`simulate?prediction=.:before` as prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5` (`k3_kda_b512`). Operator analysis puts
`unified.kda.moe.mxfp4_fused_moe` at 961.328 us / 57.15%; optimality gives
R0=961.328 us, R5=52.613 us, R0/R5=18.27, and no cached alternative. The
real B=512 table has MXFP4 GEMM1/GEMM2 at 255.1/117.3 us and the routing,
quantization, and router prep launches immediately around them. This is the
largest remaining source-level schedule opportunity after the tuning-ceiling
candidate left the same TRT-LLM kernels unchanged.

(diff.patch: 0 lines, files: )
