# MLA_B512 trial 2: agent iterations

## iter_00
### analysis.md

# Iteration 00 Baseline Analysis

Prediction: `p_6b53ccf740d742fd8779d087584820d7` (`vibesim:k3_mla_b512`), fetched with
`GET /api/v1/simulate?prediction=.:before`. The prediction was followed by the
`analyze` operator, run-summary, and iteration verbs, `optimality?scope=iter`, and
the `kernels` drill-downs.

VibeSim ranks `unified.mla.moe.mxfp4_fused_moe` first: R0 = 776.872 us, R5 =
46.037 us, R6 = 245.662 us, R0/R5 = 16.875, and `necessary_share` = 0.3162.
Its `kernels` entry is the K3 shape (112 local experts, top-k 2, hidden 3584,
intermediate 3072, BF16 input, MXFP4 E2M1/UE8M0 weights, SiTU) with backend
`sglang_trtllm_mxfp4`, no cached alternative. The run summary is R0 = 1.9368 ms,
R5/R6 = 0.4718 ms, with batching as the largest gap bucket (47.86%).

The next measured node is `unified.mla.attention.mla_decode_attention` (R0 =
686.445 us, R5 = 502.730 us, R0/R5 = 1.365, cached alternative available),
already covered by the inherited TRT-LLM MLA dispatch. The actual B=512 profile
confirms the MXFP4 BMM pair at 259.442 us and 119.318 us, plus the 385.228 us
MLA kernel; B=256 and mixed B=16 use the same expected families.

Plain fixed-flag baseline: B=512 978.432 us, B=256 638.432 us, mixed B=16
271.904 us. Diagnostic profile latencies were 984.608, 647.776, and 276.864 us.

## iter_01
### hypothesis.md

# Iteration 01 Hypothesis

Change the K3 TRT-LLM MXFP4 routed-MoE call in
`python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` so the exact
`x_quant.shape[0] == 512` decode bucket passes `tune_max_num_tokens=1024`.
FlashInfer's installed B200 tuning table contains separate 512- and 1024-token
entries, and the top-k-2 dispatch expands the primary batch into about 1024
expert rows. This tests whether the current ceiling chooses a tactic optimized
for the input-token count instead of the expanded expert workload.

The edit changes only autotuner bucket/tactic selection. Routing IDs and weights,
activation quantization, MXFP4 weights/scales, activation type, output buffer,
KDA/MLA state, and KV writes are unchanged. B=256, mixed B=16, and other
shapes use the original bucket expression. Prior `kda_25` history rejected the
same idea only at B=128; it did not measure this B=512 regime.

### analysis.md

# Iteration 01 Candidate Analysis

The post-edit profile was routed through `GET /api/v1/simulate?prediction=.:before`
and then the operator, run-summary, iteration, optimality, and MXFP4 `kernels`
verbs. The fixed before-state prediction remained `p_6b53ccf740d742fd8779d087584820d7`
(`vibesim:k3_mla_b512`). It still ranks
`unified.mla.moe.mxfp4_fused_moe` first with R0 = 776.872 us, R5 = 46.037 us,
R6 = 245.662 us, R0/R5 = 16.875, `necessary_share` = 0.3162, and no cached
alternative. The attention leaf remains cached and already uses the inherited
TRT-LLM path.

The candidate profile still launched the same 28/29 kernel families and did not
replace the routed MXFP4 BMM pair. Diagnostic graph latencies were 984.704,
647.680, and 276.992 us. The plain golden replay was numerically exact with
`state_ok=true`, but measured 978.304, 638.528, and 273.856 us against the
fixed baseline 978.432, 638.432, and 271.904 us. The primary gain was 0.013%,
below the 0.5% floor, so the tuning-ceiling edit was rejected and reverted.

### result.json

```
{
  "candidate": "K3 MXFP4 tune_max_num_tokens 512 -> 1024",
  "status": "rejected_no_primary_gain",
  "before_latency_us": {
    "512,8192": 978.432,
    "256,8192": 638.432,
    "16,65536,mix": 271.904
  },
  "after_latency_us": {
    "512,8192": 978.304,
    "256,8192": 638.528,
    "16,65536,mix": 273.856
  },
  "checks": [
    {"point": "512,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "256,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "16,65536,mix", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "vibesim_prediction": "p_6b53ccf740d742fd8779d087584820d7"
}
```

(diff.patch: 19 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_02
### hypothesis.md

# Iteration 02 Hypothesis

Raise only the K3 local 112-expert/top-2 route+quant fused-kernel coverage from
64 to 512 input rows in `python/sglang/kernels/ops/moe/moe_route_quant_fused.py`.
At B=256/512 the profile shows separate routing and MXFP8 quantization launches;
the accepted `mla_20` local specialization already produces the same route IDs,
weights, packed IDs, and UE8M0 scales for this exact shape, but its cap prevents
it from being used. The 896-expert/top-16 specialization retains its original
64-row limit.

This changes only preparation launch fusion. The route radix and per-token-group
quant device functions are unchanged; expert weights, activation scales, MoE
GEMMs, attention, KV writes, KDA state, and output buffers remain the same.
Golden replay must verify the output and post-step state at B=512, B=256, and
mixed B=16 before retaining the edit.

### analysis.md

# Iteration 02 Candidate Analysis

The post-edit diagnostic profile was followed by `simulate?prediction=.:before`,
operator/run-summary/iteration analysis, `optimality?scope=iter`, and the
MXFP4 `kernels` drill-down. VibeSim remained fixed at prediction
`p_6b53ccf740d742fd8779d087584820d7`: the MXFP4 node is R0 = 776.872 us,
R5 = 46.037 us, R6 = 245.662 us, R0/R5 = 16.875, necessary share 0.3162,
and has no cached alternative. The profile confirmed the source candidate by
launching `route_quant_fused_kernel<LocalRouterRadixTrait>` at 6.57 us for
B=512, while the remaining `routingIndicesClusterKernel` and MXFP4 BMM pair
were unchanged.

Golden replay checks passed the harness tolerance and state check, but fused
routing changed the output accumulation: max relative errors were 0.008032 at
B=512 and 0.008772 at B=256 (zero rows over tolerance, `state_ok=true`). Plain
latencies were 976.480, 637.504, and 273.888 us versus 978.432, 638.432, and
271.904 us. The primary improvement is only 0.20% and the mixed point regresses,
so the local cap extension is rejected and reverted.

### result.json

```
{
  "candidate": "local 112-expert route+quant cap 64 -> 512",
  "status": "rejected_primary_below_floor_and_mixed_regression",
  "before_latency_us": {
    "512,8192": 978.432,
    "256,8192": 638.432,
    "16,65536,mix": 271.904
  },
  "after_latency_us": {
    "512,8192": 976.480,
    "256,8192": 637.504,
    "16,65536,mix": 273.888
  },
  "checks": [
    {"point": "512,8192", "max_rel_err": 0.008032128, "rows_over_tol": 0, "state_ok": true, "pass": true},
    {"point": "256,8192", "max_rel_err": 0.008771929, "rows_over_tol": 0, "state_ok": true, "pass": true},
    {"point": "16,65536,mix", "max_rel_err": 0.0, "rows_over_tol": 0, "state_ok": true, "pass": true}
  ],
  "vibesim_prediction": "p_6b53ccf740d742fd8779d087584820d7"
}
```

(diff.patch: 33 lines, files: python/sglang/kernels/ops/moe/moe_route_quant_fused.py)

## iter_03
### hypothesis.md

# Hypothesis

The initial per-kernel profile puts the B512 critical path in the standard SiTU TRT-LLM FP4 routed-MoE call. Its MXFP4 GEMM launches are compute-bound and the profile shows PDL-enabled schedules. Prior history rejected globally disabling PDL at smaller decode batches, but that does not settle the large-m regime: at M=512, PDL producer/consumer coordination may cost more than it hides. This trial disables PDL only for the standard routed call when `x_quant.shape[0] == 512`; all other paths and token counts retain the inherited setting.

The change is scheduling-only. It does not alter routing, weights, arithmetic, output buffers, KDA state, or MLA KV writes, so numerical output and post-step state should be identical.

### analysis.md

# Analysis

The B512 VibeSim `k3_mla_b512` prediction (`p_6b53ccf740d742fd8779d087584820d7`) was rebuilt before this trial. Operator analysis predicts 1.936793953 ms total: `unified.mla.moe.mxfp4_fused_moe` 0.776872 ms (40.111%) and `unified.mla.attention.mla_decode_attention` 0.686446 ms (35.442%). The MXFP4 leaf has R0=776.872 us, R5=46.037 us, R6/R7=245.662 us, R0/R5=16.875, and `necessary_share=0.316219`; it has no cached alternative. The per-kernel table maps that node to the standard SiTU TRT-LLM FP4 routed-MoE call in `flashinfer_trtllm.py`.

This is an experiment on the large-M PDL schedule because the earlier PDL result was measured at smaller batches. The post-edit profile and replay determine whether this is useful.

### result.json

```
{
  "status": "pending",
  "prediction": "p_6b53ccf740d742fd8779d087584820d7",
  "change": "disable PDL for standard routed FP4 call at x_quant.shape[0] == 512"
}
```

(diff.patch: 16 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_07
(diff.patch: 15 lines, files: /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_08
(diff.patch: 65 lines, files: /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)
