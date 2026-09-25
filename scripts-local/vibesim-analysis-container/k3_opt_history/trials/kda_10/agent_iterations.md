# KDA trial 10: agent iterations

## agent log.md

iter_00: pristine profile + VibeSim p_38ccb83a6c4e44298f45b2251bd4cf54; plain graph baseline 402.944/263.584/138.720 us (B=128/32/1); selected KDA packed-dispatch guard experiment.
iter_01: removed the lower_bound exclusion from KDA packed decode; replay CHECK passed exactly; plain graph 402.944->394.944/263.584->259.552/138.720->136.832 us.
iter_02: attempted scoped SM100 K3 BF16-activation dispatch; reverted after the actual fused-front input failed the BF16 contract; post-revert smoke passed.
iter_03: BF16 fused KDA candidate passed all replay CHECKs but cost 111.3 us at B=128 and regressed graph latency; reverted, leaving iter_01 active.
iter_04: tested the MXFP4 BF16 dispatch and then its FP32-to-BF16 producer cast; the former was a no-op and the latter had no B=128 kernel; reverted, replay/smoke evidence retained.
iter_05: raised the fused route-quant token cap 64->128; replay CHECK passed but no fused route kernel launched and latency was unchanged; reverted.
iter_06: disabled TRTLLM PDL above 64 tokens; replay CHECK passed but B=128 regressed 394.944->398.912 us; reverted, smoke passed.

## iter_00
### hypothesis.md

Baseline iteration. The profile and VibeSim analysis identify the TRT-LLM MXFP4
MoE as the primary modeled bottleneck. A practical source-level candidate is the
KDA recurrent decode leaf because its existing packed implementation already
supports the fixed BF16 state and lower-bounded K3 gate.

### analysis.md

Prediction: p_38ccb83a6c4e44298f45b2251bd4cf54 (vibesim:k3_kda), built with prediction=.:before.

VibeSim operator analysis: total modeled kernel time 0.782953 ms. The largest node was
unified.kda.moe.mxfp4_fused_moe at 0.412401 ms / 52.67%, followed by merged_front at
0.120183 ms / 15.35% and kda_recurrent_decode at 0.042496 ms / 5.43%.

The optimality ladder ranked mxfp4_fused_moe first: R0=412.401 us, R5=9.454 us,
R0/R5=43.62, with no R6 estimate. Its kernel drill-down was backend
sglang_trtllm_mxfp4 with has_cached_alternative=false. The actual profiler table
confirmed the two TRT-LLM MXFP4 BMM launches (125.142 us + 66.563 us at B=128).

The direct MoE backend has no cached alternative in this fixed workload. The next
actionable measured node is kda_recurrent_decode: R0=42.495 us, R5=8.161 us,
R0/R5=5.21, backend sglang_triton, state_dtype=bf16, lower_bound=-5. Its source
contains an existing packed KDA kernel with the same lower-bound formula, so the
first experiment targets the dispatch guard rather than changing arithmetic.

Other relevant VibeSim leaf ladders: kda_conv_decode R0/R5=15.88 and
kda_gated_norm R0/R5=43.57; both also had no cached alternatives. Run-summary
optimality ratio was 0.07679, with 61.14% hardware gap and 29.01% batching gap.

### result.json

```
{"B":128,"seq_len":8192,"latency_us":402.94399857565657,"latency_mode":"graph","source":"plain baseline"}
{"B":32,"seq_len":8192,"latency_us":263.5839879512787,"latency_mode":"graph","source":"plain baseline"}
{"B":1,"seq_len":8192,"latency_us":138.72000575065613,"latency_mode":"graph","source":"plain baseline"}
```

(diff.patch: 1 lines, files: )

## iter_01
### hypothesis.md

Remove the stale `lower_bound is None` condition from KDA decode's packed-path
guard. The Triton packed KDA kernel accepts `lower_bound`, computes the same
K3 safe-gate expression, accumulates in FP32, stores BF16 state in place, and
returns the same [1, B, H, V] layout. This should replace the general
`fused_sigmoid_gating_delta_rule_update_kernel` launch with the lower-overhead
packed kernel while preserving output and post-step state within the driver's
check tolerance at B=128, B=32, and B=1.

### analysis.md

Prediction before edit: p_38ccb83a6c4e44298f45b2251bd4cf54 (vibesim:k3_kda),
simulated from prediction=.:before. The optimality ladder ranked
unified.kda.moe.mxfp4_fused_moe first (R0=412.401 us, R5=9.454 us, R0/R5=43.62,
necessary_share unavailable, no cached alternative). The selected experiment is
the next actionable measured leaf, unified.kda.attention.kda_recurrent_decode
(R0=42.495 us, R5=8.161 us, R0/R5=5.21, no cached alternative): its source
dispatch guard excludes packed decode solely when lower_bound is present, while
the packed KDA kernel already implements that lower-bound gate and BF16 state.

The profiler confirmed the baseline general recurrent launch as
fused_sigmoid_gating_delta_rule_update_kernel (29.408 us at B=128), alongside
_causal_conv1d_update_kernel (6.650 us) and the separate gated norm. The edit
only changes path selection; it does not change the convolution, gate equation,
state indices, state update, or downstream output norm.

Post-edit diagnostic profile: the B=128 graph profile was 362.080 us with
411.6 us kernel sum, and the B=32/B=1 profiles were 242.272/137.760 us. The
new profiler table shows the same launch count and the expected packed KDA path;
the plain replay is the timing reference because the split/profile run perturbs
graph timing. VibeSim `prediction=.:after` resolves to the same fixed before-state
prediction (the service explicitly reports this), so its operator, run-summary,
and ladder outputs remain unchanged; the measured profile is the source of truth
for this edit's small recurrent-path gain.

The required replay passed exactly at all points (`max_rel_err=0`, `state_ok=true`).
Because the primary plain timing gain is only 1.99%, this change alone is not
enough for the objective; retain it as a correctness-preserving base while
targeting the larger mxfp4 fused-MoE node next.

### result.json

```
{"B":128,"seq_len":8192,"latency_us":394.9440121650696,"latency_mode":"graph","check":{"max_abs_err":0.0,"max_rel_err":0.0,"state_ok":true,"pass":true}}
{"B":32,"seq_len":8192,"latency_us":259.552001953125,"latency_mode":"graph","check":{"max_abs_err":0.0,"max_rel_err":0.0,"state_ok":true,"pass":true}}
{"B":1,"seq_len":8192,"latency_us":136.83199882507324,"latency_mode":"graph","check":{"max_abs_err":0.0,"max_rel_err":0.0,"state_ok":true,"pass":true}}
```

(diff.patch: 17 lines, files: python/sglang/srt/layers/attention/linear/kda_backend.py)

## iter_02
### hypothesis.md

In the SM100 TRT-LLM MXFP4 wrapper, select FlashInfer's existing BF16-input
variant only for the exact K3 shape (hidden=3584, intermediate=3072) when the
fixed default precision is requested. This removes the runtime activation
quantization work and uses the same packed MXFP4 expert weights, top-k routing,
SiTU parameters, output buffer, and downstream combine path. The KDA state is
not touched by this MoE-only change; replay CHECKs will verify the full layer
output and post-step state.

### analysis.md

Prediction before edit: p_38ccb83a6c4e44298f45b2251bd4cf54 (vibesim:k3_kda),
re-fetched with prediction=.:after after iter_01. The service states that this
container serves a fixed before-state prediction, so operator/optimality outputs
are unchanged: `unified.kda.moe.mxfp4_fused_moe` is R0=412.401 us, R5=9.454 us,
R0/R5=43.62, 52.67% of modeled time, backend `sglang_trtllm_mxfp4`, and
`has_cached_alternative=false`.

The iter_01 profiler still showed the two TRT-LLM MXFP4 BMM launches as the
largest real kernels (B=128: 125.142 us and 66.563 us in the baseline table),
with only the KDA recurrent dispatch changed. This iteration targets the source
wrapper for that top node. FlashInfer 0.6.18 exposes an existing `bf16` branch
for the same routed SiTU MXFP4 operation; for K3's exact hidden/intermediate
shape it skips runtime MXFP8 activation quantization while preserving expert
weights, routing IDs, top-k, activation type, and BF16 output storage.

The mandatory smoke rejected this hypothesis before timing: the BF16 branch
asserted because the actual fused-front routed input was not BF16. The override
was reverted immediately, the cache-cleared smoke passed with `LAYER_SMOKE_OK`,
and no invalid profile or replay result is treated as a performance result.

### result.json

```
{"B":1,"seq_len":8192,"latency_us":null,"ok":false,"error":"AssertionError: FlashInfer BF16 branch received non-BF16 hidden_states","pass":false}
{"smoke_after_revert":true,"marker":"LAYER_SMOKE_OK"}
```

(diff.patch: 3 lines, files: )

## iter_03
### hypothesis.md

Extend `kda_fused_decode` to accept BF16 recurrent state and fuse the causal
convolution, KDA recurrence, and output norm for the production K3 shape. The
state load/store specialization converts BF16 state to FP32 shared/register
values and rounds recurrent writes back to BF16; the q/k/v convolution output
also needs the same BF16 store/reload boundary as the reference chain. This
preserves the post-step state contract and was checked against the golden.

The experiment was numerically valid after the explicit BF16 qkv boundary,
but the fused implementation was slower at the scored batch, so it is not
retained in the active source.

### analysis.md

VibeSim prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`, built with
`simulate?prediction=.:after_iter03`. The service reports a fixed before-state
prediction, so the post-edit analysis remains the same: total modeled time
0.782953 ms; `unified.kda.moe.mxfp4_fused_moe` is 0.412401 ms (52.67%), with
R0/R5=43.62 and the largest headroom. The KDA recurrent node is 0.042496 ms
(5.43%), R0/R5=5.21, so it was a plausible secondary target but not the
primary roofline candidate.

The iter_03 profile confirmed the BF16 fused kernel was actually launched. At
B=128 it consumed 111.30 us by itself and the diagnostic graph step was
448.06 us; the plain replay was also slower than the iter_01 reference. The
candidate passed its replay CHECKs after adding the BF16 qkv round-trip, but
it did not meet the latency objective, so the BF16 fused specialization was
reverted. The active tree remains on the validated iter_01 path.

### result.json

```
{
  "candidate": "BF16 fused KDA decode (reverted)",
  "replay": [
    {"B": 128, "seq_len": 8192, "latency_us": 482.84798860549927, "max_rel_err": 0.014705881364310497, "state_ok": true, "pass": true},
    {"B": 32, "seq_len": 8192, "latency_us": 280.0320088863373, "max_rel_err": 0.007476075982692748, "state_ok": true, "pass": true},
    {"B": 1, "seq_len": 8192, "latency_us": 159.13599729537964, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "smoke_after_revert": true
}
```

(diff.patch: 422 lines, files: python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh, python/sglang/kernels/ops/attention/kda_fused_decode.py)

## iter_04
### hypothesis.md

The dominant MXFP4 node appeared to have an available BF16 activation route,
so first make that route's dtype guard explicit, then make the K3 fused-front
producer satisfy it for the scored batch. The route would preserve the same
router outputs, weights, expert tensors, and output accumulation if a valid
FlashInfer kernel existed. Runtime dispatch disproved that assumption for the
installed SM100 kernel set; the producer cast was therefore not retained.

### analysis.md

VibeSim prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`, fetched with
`simulate?prediction=.:before_iter04`, then analyzed at operator,
run_summary, optimality, and kernel levels. The model ranks
`unified.kda.moe.mxfp4_fused_moe` first at 0.412401 ms / 52.67%, with
R0/R5=43.62, the largest headroom, and no cached alternative. Its kernel
shape is K3's 3584 x 3072, 112-local-expert MXFP4 SiTU path.

The first edit changed the FlashInfer BF16 branch predicate to also require
`x.dtype == torch.bfloat16`. The measured fused-front input remained FP32 at
B=128, so the dispatch and kernel table were unchanged. Its replay was
numerically exact: B=128/32/1 measured 396.832/259.584/136.800 us with
`max_rel_err=0`, `state_ok=true`, and `pass=true` at every point.

The follow-up producer edit cast the B>64 routed rows to BF16 to force that
branch. The mandatory B=1 smoke passed because it does not enter the branch,
but the full replay rejected B=128 with FlashInfer's no-kernel error for
MXFP4 x BF16 -> BF16 SiTU. B=32 and B=1 remained exact. Both edits were
reverted; the post-revert smoke passed. The profiled no-op B=128 step was
362.080 us with 412.427 us of kernel time and 26 launches.

### result.json

```
{
  "candidate_bf16_predicate": {
    "replay": [
      {"B": 128, "seq_len": 8192, "latency_us": 396.8320125770569, "max_rel_err": 0.0, "state_ok": true, "pass": true},
      {"B": 32, "seq_len": 8192, "latency_us": 259.5840096473694, "max_rel_err": 0.0, "state_ok": true, "pass": true},
      {"B": 1, "seq_len": 8192, "latency_us": 136.79999113082886, "max_rel_err": 0.0, "state_ok": true, "pass": true}
    ],
    "outcome": "no-op: actual fused-front input was not BF16"
  },
  "candidate_producer_cast": {
    "B128": {"pass": false, "error": "FlashInfer TRTLLM no kernel found for MXFP4 x BF16 -> BF16 SiTU"},
    "B32": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "B1": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "outcome": "reverted"
  },
  "smoke_after_revert": true
}
```

(diff.patch: 14 lines, files: candidate/flashinfer_trtllm.py, candidate/kimi_k3.py)

## iter_05
### hypothesis.md

The route-quant fused kernel was guarded by a small token cap. Raising the
cap only for the existing path should have fused routing/quantization without
changing routing values or expert math. The profiler showed that this layer
did not enter that kernel, so the cap change had no useful effect and was
reverted.

### analysis.md

VibeSim prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`, fetched with
`simulate?prediction=.:before_iter05`, followed by all four analysis verbs.
The result is the fixed before-state model: MXFP4 fused MoE remains the top
node at 0.412401 ms / 52.67%, R0/R5=43.62, no cached alternative; routing
and route-quant are leaves inside that node rather than a separate modeled
roofline target.

The experiment raised `moe_route_quant_fused._MAX_TOKENS` from 64 to 128,
expecting the B=128 decode shape to use the fused route-quant kernel and
remove launch overhead. Replay stayed exact at 395.776/259.616/136.896 us
for B=128/32/1, with zero relative error and valid post-step state at all
points. The post-edit B=128 profile still had 26 launches and showed
`_router_triton_kernel` (3.831 us), routing indices (3.674 us), and per-token
FP8 quantization (2.582 us); no `moe_route_quant_fused` kernel appeared.
The cap did not cover this path's other alignment/coverage guards, so there
was no meaningful speed change. The edit was reverted and the smoke passed.

### result.json

```
{
  "candidate": "moe_route_quant_fused._MAX_TOKENS 64 -> 128 (reverted)",
  "replay": [
    {"B": 128, "seq_len": 8192, "latency_us": 395.77600359916687, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 32, "seq_len": 8192, "latency_us": 259.6159875392914, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 1, "seq_len": 8192, "latency_us": 136.89599931240082, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "smoke_after_revert": true
}
```

(diff.patch: 5 lines, files: candidate/moe_route_quant_fused.py)

## iter_06
### hypothesis.md

PDL dependency setup may hide producer/consumer latency for small decode
batches but add overhead once the MXFP4 routed token count reaches 128. Keep
the existing PDL path through 64 tokens and disable it at the scored shape.
This changes only scheduling, not tensor values, routing, expert math, or
state, so replay CHECKs should remain exact. Measurement showed the policy
was slower at B=128 and it was reverted.

### analysis.md

VibeSim prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`, fetched with
`simulate?prediction=.:after_iter06`, then run through operator,
run_summary, optimality, and kernel analysis. The service reports a fixed
before-state prediction: MXFP4 fused MoE is still 0.412401 ms / 52.67%,
R0/R5=43.62, with no cached alternative. The modeled hardware gap is 61.14%
and batching gap 29.01%, so a dispatch policy change needed to be validated
against the real kernel table.

The experiment disabled TRTLLM PDL above 64 tokens. Replay CHECK passed at
all points (B=128/32/1: 398.912/259.584/142.880 us, zero relative error,
state valid), but B=128 regressed against the active iter_01 reference
394.944 us. The post-edit profile measured 366.048 us graph time, 406.848 us
kernel time, and 26 launches; the large MXFP4 GEMMs remained the same path.
The change was reverted after the required cache-cleared B=1 smoke passed.

### result.json

```
{
  "candidate": "TRTLLM PDL disabled for num_tokens > 64 (reverted)",
  "replay": [
    {"B": 128, "seq_len": 8192, "latency_us": 398.9120125770569, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 32, "seq_len": 8192, "latency_us": 259.5840096473694, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 1, "seq_len": 8192, "latency_us": 142.87999272346497, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "smoke_after_revert": true
}
```

(diff.patch: 13 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)
