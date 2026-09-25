# KDA_B512 trial 3: agent iterations

## agent log.md

iter_00 baseline: VibeSim p_1066e0f8b4d148de9d40979e8ca7f1e5 selected routed MXFP4 MoE (57.15%, R0/R5=18.27, no cached alternative); plain replay 677.344/482.784/378.368 us for B=512/256/128, golden checks exact.
iter_01 rejected: B=512 tune ceiling 512->1024 kept the same 23 launches and `t128x8x512` BMMs; replay 677.344->677.248/482.784->482.784/378.368->378.368 us, all CHECKs exact.
iter_02 retained: local 112/top-2 route+quant fusion at B=512 removed one launch; replay 677.344->676.384/482.784->482.784/378.368->378.368 us, CHECK pass on all points, 0.14% primary gain below the 0.5% floor.
iter_03 rejected: bf16 KDA 4->3 TMA stages at B*heads>=4096 passed checks but replayed 676.480/482.816/378.400 us versus 676.384/482.784/378.368 us on the route-fusion stack; source restored.
iter_04 rejected: B=512 in-kernel TRT-LLM routing replayed 688.576 us versus 676.384 us and failed decode correctness (2 rows over tolerance, max_rel_err 0.315); B=256/128 stayed exact, source restored.
iter_05 invalid: B=512 MXFP4 x BF16 SiTU reached TRT-LLM but had no installed SM100 kernel (`MxE2m1 x Bfloat16`, routed SiTU); B=256/128 stayed exact and the source was restored after smoke.
iter_06 rejected: B=512-only TRT-LLM PDL disable passed exact checks but replayed 681.440 us versus 676.384 us; B=128 also regressed to 382.496 us, source restored.

## iter_00
### hypothesis.md

# Iteration 00 hypothesis

The baseline routed MXFP4 operation is compute-bound at B=512 and its two
expert launches use the `t128x8x512` family. FlashInfer receives a tuning
ceiling of `next_power_of_2(num_tokens)`, which is 512 for this point. Raising
that ceiling to 1024 only for the exact 512-token case may expose the larger
bucket's SM100 tactic/workspace schedule while leaving B=256 and B=128 on
their original paths.

This is a scheduling/autotuning change: the hidden activation, MXFP8 scales,
expert weights, routing IDs and weights, SiTU constants, output buffer, and
KDA state are unchanged. The replay must still match the captured output and
post-step conv/recurrent/MLA state at every required point.

### analysis.md

# Iteration 00: baseline

The required diagnostic profile is in `run.json` with per-point kernel tables in
`profile_B512_L8192.json`, `profile_B256_L8192.json`, and
`profile_B128_L8192.json`. The plain pre-edit replay reference is 677.344 us,
482.784 us, and 378.368 us for B=512, 256, and 128 respectively.

VibeSim was called after the profile with `simulate?prediction=.:before` and
returned `p_1066e0f8b4d148de9d40979e8ca7f1e5` (`k3_kda_b512`). Operator,
run-summary, iteration, optimality, and kernel analyses were run on that
prediction. The modeled leaf total is 1.682107 ms. The largest node is
`unified.kda.moe.mxfp4_fused_moe` at 0.961327 ms / 57.15%, followed by
`kda_recurrent_decode` at 0.200253 ms / 11.90% and `merged_front` at
0.195559 ms / 11.63%.

The optimality ladder ranks the routed MXFP4 leaf first: R0=961.328 us,
R5=52.613 us, R0/R5=18.27, with no cached alternative. KDA recurrent decode
is R0=200.253 us, R5=45.419 us, R0/R5=4.41, also without a cached
alternative. Run-summary analysis attributes 62.91% of the modeled gap to
batching and 17.27% to the hardware gap.

The profiler maps the routed leaf to the SM100 TRT-LLM MXFP4 kernels:
`bmm_MxE4m3_MxE2m1MxE4m3...t128x8x512...` and
`bmm_Bfloat16_MxE2m1MxE4m3...t128x8x512...`, which take 255.309 and
117.168 us at B=512. The live fused KDA source is the bf16-state
`kda_decode_fusion_many_heads_kernel`; the front and shared projections remain
ordinary dense kernels. The next experiment targets only the FlashInfer
large-batch tuning envelope, leaving all arithmetic, routing, state updates,
and output buffers unchanged.

## iter_01
### hypothesis.md

# Iteration 01 hypothesis

The B=512 routed MXFP4 calls use `tune_max_num_tokens=512`, while the live
expert work expands each token to two routed rows. Passing a 1024 tuning
ceiling only for the exact 512-token decode shape may make the SM100 autotuner
consider a schedule sized for the approximately 1024 local routed rows.

The edit changes only FlashInfer's tactic/workspace selection. It leaves the
MXFP8 activation and scales, expert weights, top-2 IDs and weights, SiTU
activation, output allocation, and all KDA conv/recurrent state untouched. The
prior tuning-bucket trials in `kda_11` and `kda_25` were neutral at smaller
decode points; this trial is specifically checking the new large-batch regime
identified by the B=512 profile.

### analysis.md

# Iteration 01 analysis

The candidate is based on iteration 00's VibeSim target
`unified.kda.moe.mxfp4_fused_moe`, which had R0=961.328 us, R5=52.613 us,
R0/R5=18.27, and no cached alternative in prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5`. The real profile showed the B=512
`t128x8x512` MXFP4 BMM pair as the dominant launched work.

Post-edit profile and VibeSim re-analysis will be appended after the smoke and
replay. If the kernel names and graph time remain unchanged, the source will
be restored and the trial recorded as neutral.

The smoke passed with `LAYER_SMOKE_OK`. Replay checks were exact at all three
points. Plain replay graph latency was 677.248 us, 482.784 us, and 378.368 us
for B=512, 256, and 128, respectively, versus 677.344 us, 482.784 us, and
378.368 us before the edit. The B=512 change is 0.014%, below the 0.5%
acceptance floor.

The post-edit diagnostic profile remains 23 launches at every point. Its
B=512 graph time is 673.152 us and the routed MoE split is 673.3 us; the
profiled table still contains the same `t128x8x512` BMM pair. After
`simulate?prediction=.:after`, VibeSim returned the fixed prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5`; operator, run-summary, iteration,
optimality, and `mxfp4_fused_moe` kernel analyses are unchanged. The candidate
is rejected as neutral and the source is restored.

### result.json

```
{
  "status": "rejected_neutral",
  "before_latency_us": {
    "512,8192": 677.343992,
    "256,8192": 482.784003,
    "128,8192": 378.367990
  },
  "after_latency_us": {
    "512,8192": 677.248001,
    "256,8192": 482.784003,
    "128,8192": 378.367990
  },
  "checks": {
    "512,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "prediction_id": "p_1066e0f8b4d148de9d40979e8ca7f1e5"
}
```

(diff.patch: 15 lines, files: /workspace/opt_run/iter_01/after/flashinfer_trtllm.py)

## iter_02
### hypothesis.md

# Iteration 02 hypothesis

The B=512 profile launches separate local routing, packed-ID preparation, and
MXFP8 activation-quantization work before the two dominant TRT-LLM MXFP4
BMMs. The existing route-plus-quant JIT already fuses those operations, and
the prior `mla_20`/`kda_23` work contains a corrected 112-local-expert,
top-2 specialization. I will reuse that implementation and enable it only
for exactly 512 rows in this K3 layer.

The fused kernel uses the same sigmoid/bias ranking, BF16/FP32 activation
quantizer, packed `(expert_id << 16) | bf16(weight)` representation, and
FlashInfer inputs as the unfused chain. It writes only temporary routing and
quantization buffers; KDA conv/recurrent state and MLA KV rows are not touched.
The B=128 and B=256 paths remain on the baseline route/quant chain to avoid
the small-batch neutral result recorded in `kda_23`.

### analysis.md

# Iteration 02 analysis

The post-iteration-01 VibeSim analysis still identifies
`unified.kda.moe.mxfp4_fused_moe` as the primary node at R0=961.328 us,
R5=52.613 us, R0/R5=18.27, with no cached alternative. The live B=512
profile maps the node to the two SM100 MXFP4 BMMs and also shows the separate
router/packing/quantization launches immediately before them.

This iteration targets only that measured preparation chain.

The reload smoke passed. The full replay activated
`void sglang::route_quant_fused_kernel<sglang::LocalRouterRadixTrait, true,
float, float>` at B=512 and retained the baseline route/quant kernels at
B=256 and B=128. B=512 graph latency was 676.384 us versus the iteration-00
677.344 us baseline; B=256 and B=128 were 482.784 us and 378.368 us. The
primary improvement is 0.14%, below the judge's 0.5% floor, so this is kept as
a non-regressing stack for the next experiment rather than counted as a final
speed win.

The B=512 check passed with `max_rel_err=0.008299`, zero rows over tolerance,
`state_ok=true`, and no NaNs. B=256 and B=128 were bit-exact with
`state_ok=true`. The diagnostic profile has 22 launches at B=512 (one fewer
than baseline) and 23 at the smaller points. It replaces `_router_triton_kernel`
and the standalone MXFP8 quant launch with the local fused route/quant kernel,
while FlashInfer's internal routing-index launch remains.

After the profile, VibeSim `simulate?prediction=.:after` returned the fixed
prediction `p_1066e0f8b4d148de9d40979e8ca7f1e5`. Operator, run-summary,
iteration, optimality, and both relevant kernel drill-downs remained unchanged:
the routed-MXFP4 node is still 57.15% with R0/R5=18.27 and no cached
alternative; recurrent KDA is R0/R5=4.41 with no cached alternative.

### result.json

```
{
  "status": "retained_non_regressing",
  "candidate": "112-local-expert top-2 route+quant fusion at B=512",
  "prediction_id": "p_1066e0f8b4d148de9d40979e8ca7f1e5",
  "before_latency_us": {
    "512,8192": 677.343992,
    "256,8192": 482.784003,
    "128,8192": 378.367990
  },
  "after_latency_us": {
    "512,8192": 676.383972,
    "256,8192": 482.784003,
    "128,8192": 378.367990
  },
  "checks": {
    "512,8192": {"max_rel_err": 0.008298755, "state_ok": true, "pass": true},
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  }
}
```

(diff.patch: 435 lines, files: /workspace/opt_run/iter_02/after/moe_route_quant_fused.py, /workspace/opt_run/iter_02/after/route_quant_fused.cuh, /workspace/opt_run/iter_02/after/route_radix.cuh, /workspace/opt_run/iter_02/after/topk.py)

## iter_03
### hypothesis.md

# Iteration 03 hypothesis

Iteration 02 removed the separate K3 local route/quantizer launches at B=512,
but the plain replay gain was only 0.14%. The remaining editable attention
leaf is the bf16-state fused KDA decode kernel. Its B=512 grid has 6,144
head-token CTAs, where four TMA staging buffers consume 64 KiB of dynamic
shared memory. The already compiled three-stage variant uses 48 KiB and may
raise occupancy at this large grid.

Select three TMA stages only when `B * heads >= 4096` for bf16 state. B=256
and B=128 retain four stages. The stage count changes only state staging and
barrier scheduling; per-element recurrence arithmetic, bf16 rounding, state
addresses, conv state, output values, and MLA KV writes are unchanged. This
trial builds on the accepted bf16 KDA port and the `kda_23` three-stage
experiment, which was neutral at smaller B=128 but did not test this large
grid.

### analysis.md

# Iteration 03 analysis

The active stack is iteration 02's B=512-only local route/quant fusion. VibeSim
was re-run after its profile and still ranks
`unified.kda.moe.mxfp4_fused_moe` first at R0=961.328 us, R5=52.613 us,
R0/R5=18.27, with no cached alternative. The next directly editable leaf is
`unified.kda.attention.kda_recurrent_decode` at R0=200.253 us, R5=45.419 us,
R0/R5=4.41; its kernel drill-down confirms the bf16-state 12x128x128 fused
KDA path.

The smoke passed. Replay checks passed at every point with the same B=512
relative error as iteration 02 (`max_rel_err=0.008299`, `state_ok=true`).
Plain replay was 676.480 us, 482.816 us, and 378.400 us for B=512, 256,
and 128. This is 0.096 us slower at B=512 than the iteration-02 stack and
does not clear the baseline 0.5% floor.

The post-edit diagnostic profile still has 22/23/23 launches and the same
fused local route kernel. VibeSim `simulate?prediction=.:after` returned the
fixed prediction `p_1066e0f8b4d148de9d40979e8ca7f1e5`; all analysis levels,
optimality, and the recurrent leaf drill-down remained unchanged. The
three-stage bf16 KDA variant is rejected and the source is restored to four
stages for bf16 state.

### result.json

```
{
  "status": "rejected",
  "before_latency_us": {
    "512,8192": 676.3840,
    "256,8192": 482.7840,
    "128,8192": 378.3680
  },
  "after_latency_us": {
    "512,8192": 676.4800,
    "256,8192": 482.8160,
    "128,8192": 378.4000
  },
  "checks": {
    "512,8192": {"max_rel_err": 0.008298755, "state_ok": true, "pass": true},
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "prediction_id": "p_1066e0f8b4d148de9d40979e8ca7f1e5"
}
```

(diff.patch: 11 lines, files: /workspace/opt_run/iter_03/after/kda_fused_decode.cuh)

## iter_04
### hypothesis.md

# Iteration 04 hypothesis

At exactly 512 rows, temporarily select `TopKOutputFormat.BYPASSED` for the
K3 fused-front routed call.  `TopK.forward_cuda` then returns the existing
`BypassedTopKOutput` carrier instead of materializing top-k IDs and weights;
the already-wired SM100 SiTU runner calls
`trtllm_fp4_block_scale_moe`, which performs sigmoid, correction-bias ranking,
top-2 selection, renormalization, activation quantization, and the two expert
GEMMs in its fused routing path.  The standard pre-routed path remains active
for B=256/128 and for every other model/backend shape.

The change does not modify model weights, routing logits, KDA convolution or
recurrent state, MLA KV rows, expert weights, or the output buffer contract.
The top-k configuration is restored immediately after the synchronous Python
dispatch setup, and CUDA graph replay contains only the captured kernels.  The
replay golden check is required to validate that FlashInfer's routing arithmetic
and all post-step state remain within the task tolerance.

Outcome: rejected.  The in-kernel routing path was both slower at B=512 and
numerically invalid for the decode rule, despite preserving the KDA state.

### analysis.md

# Iteration 04 analysis

The fresh diagnostic profile of the retained iteration-02 stack launches 22
kernels at B=512 and 23 at B=256/128.  The B=512 profile has the local
112-expert/top-2 route+quant kernel active, followed by the two dominant
FlashInfer MXFP4 expert BMMs; the measured diagnostic graph values are
671.232/491.328/346.624 us for B=512/256/128.  The profile is diagnostic only;
the iteration-02 plain replay reference is 676.384/482.784/378.368 us.

After `simulate?prediction=.:after`, VibeSim returned fixed prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5` (`vibesim:k3_kda_b512`).  The required
operator, run-summary, iteration, optimality, and kernel analyses were run.
The selected node remains `unified.kda.moe.mxfp4_fused_moe` at R0=961.328 us,
R5=52.613 us, R0/R5=18.27, with no cached alternative.  Its kernel drill-down
is the measured K3 shape: hidden=3584, intermediate=3072, 112 local experts,
top-k=2, BF16 input, MXFP4 E2M1/UE8M0 weights, group size 32, SiTU, and
DeepSeekV3 sigmoid routing.  The next nodes are KDA recurrent decode
(R0/R5=4.41) and merged front (R0/R5=1.67).

Prior trial `kda_8` tested the same in-kernel routing contract at B=128/32/1
and rejected it because FlashInfer's internal routing-index path and an added
BF16 conversion outweighed the removed router launch.  The present workload
has B=512 and roughly 1024 local routed rows, so this iteration tests the
contract only at exactly 512 rows while retaining the standard path at 256
and 128.

The cache-cleared smoke passed after the edit.  The plain replay then rejected
the candidate: B=512 was 688.576 us versus the retained-stack 676.384 us, and
the decode check reported `max_rel_err=0.315353`, two rows over tolerance, and
`state_ok=true`.  B=256 and B=128 were bit-exact at 482.752 and 378.368 us.
The source was restored and the required post-restore smoke passed.

### result.json

```
{
  "status": "rejected",
  "candidate": "B=512 K3 in-kernel TRT-LLM DeepSeekV3 routing",
  "prediction_id": "p_1066e0f8b4d148de9d40979e8ca7f1e5",
  "before_latency_us": {
    "512,8192": 676.383972,
    "256,8192": 482.784003,
    "128,8192": 378.367990
  },
  "after_latency_us": {
    "512,8192": 688.575983,
    "256,8192": 482.751995,
    "128,8192": 378.367990
  },
  "checks": {
    "512,8192": {
      "max_rel_err": 0.315353,
      "rows_over_tol": 2,
      "state_ok": true,
      "pass": false
    },
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "smoke_after_restore": true
}
```

(diff.patch: 82 lines, files: /workspace/opt_run/iter_04/after/kimi_k3.py)

## iter_05
### hypothesis.md

# Iteration 05 hypothesis

For exactly 512 routed rows, pass `flashinfer_mxfp4_moe_precision="bf16"`
to the SM100 TRT-LLM MXFP4 runner.  This keeps the same packed MXFP4 weights,
correction-bias routing, top-k IDs and weights, SiTU parameters, expert output
buffer, and KDA state, while removing the MXFP8 activation quantization and
scale tensor from the critical path.  The current route+quant handoff is
disabled for this one shape because BF16 mode cannot consume its MXFP8 result.
The default MXFP8 path remains unchanged at B=256/128 and for all other
backends/shapes.

The hypothesis is explicitly re-tested despite prior `kda_5`/`kda_7` failures,
because those trials were at smaller m and the current measured B=512 MoE has
roughly 1024 local routed rows, where the history predicts a different tactic
regime.  Correctness and the post-step state remain acceptance gates.

Outcome: invalid.  The target B=512 shape has no BF16 SiTU TRT-LLM tactic in
the installed FlashInfer/SM100 cubin, so no latency or correctness result can
be accepted from this branch.

### analysis.md

# Iteration 05 analysis

The restored retained stack was profiled with the fixed diagnostic command:
22 launches at B=512 and 23 at B=256/128, with diagnostic graph values
671.232/491.904/346.432 us.  The B=512 table still contains the two
`bmm_*MxE2m1MxE4m3*` TRT-LLM MXFP4 expert GEMMs and the local fused
route+quant launch.

The fresh `simulate?prediction=.:after` returned fixed VibeSim prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5`.  Operator, run-summary, iteration,
optimality, and MXFP4 kernel drill-down analyses were rerun.  The selected
node remains `unified.kda.moe.mxfp4_fused_moe` at 57.15%, R0=961.328 us,
R5=52.613 us, R0/R5=18.27, and no cached alternative.  The drill-down is
the 3584 x 3072, 112-local-expert, top-2, BF16-input, MXFP4 E2M1/UE8M0,
group-32, SiTU shape.  Run-summary assigns 62.91% of the modeled time to
batching and 17.27% to hardware gap, so changing the activation tactic is a
reasonable experiment but its graph replay must be measured.

Prior trials `kda_5` and `kda_7` rejected BF16 activation because the installed
SM100 kernel set had no MXFP4-weight x BF16 SiTU tactic at their small decode
shape.  `TECHNIQUES.md` explicitly identifies B>=512 and prefill as the
remaining regime to re-evaluate, so this iteration changes only exact B=512.

The first full replay exposed the FP32 merged-front input contract; the
candidate was corrected to cast only B=512 routed rows to BF16.  The corrected
replay reached TRT-LLM and failed with `No kernel found` for
`MxE2m1 x Bfloat16`, routed SiTU (`mActType=2`, `mRouteAct=1`).  The branch is
therefore invalid on this installed B200 kernel set.  B=256/128 remained exact,
and the source was restored with a passing smoke.

### result.json

```
{
  "status": "invalid_no_kernel",
  "candidate": "B=512 MXFP4 x BF16 TRT-LLM SiTU activation",
  "prediction_id": "p_1066e0f8b4d148de9d40979e8ca7f1e5",
  "error": "No kernel found for MxE2m1 x Bfloat16, routed SiTU",
  "checks": {
    "512,8192": {"pass": false, "latency_us": null, "state_ok": null},
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "smoke_after_restore": true
}
```

(diff.patch: 70 lines, files: /workspace/opt_run/iter_05/after/kimi_k3.py, /workspace/opt_run/iter_05/after/mxfp4.py)

## iter_06
### hypothesis.md

# Iteration 06 hypothesis

Disable TRT-LLM programmatic dependent launch only when the routed token count
is exactly 512.  PDL changes launch scheduling and kernel selection metadata,
but not routing, MXFP4 weights, activation quantization, SiTU arithmetic,
expert output, KDA state, or MLA KV rows.  B=256, B=128, and all other token
counts keep the existing PDL policy.  The live kernel names contain `schPd`,
so the hypothesis targets a real launched option rather than an unused code
path.

The prior `kda_7`/`kda_10` PDL-off trials regressed their smaller points; this
large-batch trial is retained only if B=512 improves and the smaller points do
not regress.

Outcome: rejected.  PDL remains enabled at all measured shapes.

### analysis.md

# Iteration 06 analysis

The current post-rollback profile is the iteration-05 restored-stack table:
22 launches at B=512 and 23 at B=256/128, with the B=512 expert BMMs using
the `schPd` TRT-LLM PDL family.  VibeSim prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5` was fetched and its operator,
run-summary, iteration, optimality, and kernel analyses were already rerun for
this unchanged stack.  They rank `unified.kda.moe.mxfp4_fused_moe` first at
57.15%, R0=961.328 us, R5=52.613 us, R0/R5=18.27, with no cached alternative.
The KDA recurrent node is second at R0/R5=4.41.

The profile maps the dominant node to the standard routed
`trtllm_fp4_block_scale_routed_moe` call in
`flashinfer_trtllm.py:1108`; its `enable_pdl` argument is supplied by
`trtllm_moe_enable_pdl`.  Prior trials `kda_7` and `kda_10` disabled PDL at
smaller decode shapes and regressed, so this is a bounded re-test at the
large B=512 compute regime where dependent-launch setup could have a different
tradeoff.  The change cannot alter tensor values or state.

The cache-cleared smoke passed.  The plain replay passed all state/output
checks, but measured 681.440/481.760/382.496 us for B=512/256/128 versus the
retained-stack 676.384/482.784/378.368 us.  The candidate is rejected and the
helper was restored, followed by another passing smoke.

### result.json

```
{
  "status": "rejected",
  "candidate": "disable TRT-LLM PDL at exactly 512 tokens",
  "prediction_id": "p_1066e0f8b4d148de9d40979e8ca7f1e5",
  "before_latency_us": {
    "512,8192": 676.383972,
    "256,8192": 482.784003,
    "128,8192": 378.367990
  },
  "after_latency_us": {
    "512,8192": 681.439996,
    "256,8192": 481.759995,
    "128,8192": 382.495999
  },
  "checks": {
    "512,8192": {"max_rel_err": 0.008299, "state_ok": true, "pass": true},
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "smoke_after_restore": true
}
```

(diff.patch: 15 lines, files: /workspace/opt_run/iter_06/after/flashinfer_trtllm.py)
