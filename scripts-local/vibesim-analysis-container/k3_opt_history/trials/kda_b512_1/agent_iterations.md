# KDA_B512 trial 1: agent iterations

## agent log.md

iter_00 baseline: VibeSim ranked mxfp4_fused_moe first (R0 961.328 us, R5 52.613 us, R0/R5 18.27, no cached alternative); plain graph 677.408 / 482.784 / 378.400 us for B=512/256/128, golden captured.
iter_01 rejected: large-batch TRT-LLM tuner ceiling (2x routed rows) kept the same MXFP4 dynamic-BMM family; plain graph 677.408 -> 679.424 / 482.784 -> 480.800 / 378.400 -> 378.368 us, exact CHECKs.
iter_02 rejected/inactive: local 112/top-2 route+quant JIT did not match the driver’s global routing shape; profile retained separate router+quant launches, plain graph 677.344 / 482.752 / 378.368 us, exact CHECKs.
iter_03 rejected: raising the existing fused route+quant token cap to 512 was numerically exact but still inactive for the live 112/top-2 path; plain graph 677.312 / 482.784 / 378.368 us.

## iter_00
### hypothesis.md

# Iteration 00 hypothesis

No source change is made in the baseline iteration. The next experiment will
test whether the TRT-LLM MXFP4 autotuner benefits from a request-token ceiling
that matches the routed local-row count at the large points: use a 512-row
bucket for B=256 and a 1024-row bucket for B=512, while retaining the current
ceiling for B=128. The experiment builds on the prior KDA trials that found the
same node dominant but only tested a larger ceiling at B=128; this workload
specifically exercises the new large-m regime.

### analysis.md

# Iteration 00: seeded baseline

VibeSim was called in the required order: `workspace-info`, then
`simulate?prediction=.:before`, which built prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5` for the fixed before-state run
`k3_kda_b512`.

The analysis handle accepted by this service is
`/opt/vibesim/repo/logs/k3_kda_b512`. The operator analysis modeled
`1.682107 ms` of leaf time. The largest node is
`unified.kda.moe.mxfp4_fused_moe` at `0.961327 ms` / `57.15%`, followed by
`unified.kda.attention.kda_recurrent_decode` at `0.200253 ms` / `11.90%` and
`unified.kda.moe.merged_front` at `0.195559 ms` / `11.63%`.

The run summary reports optimality ratio `0.1825`, necessary ratio `0.1825`,
batching gap `62.91%`, hardware gap `17.27%`, and imbalance `1.57%`.
`optimality?scope=iter` ranks the MXFP4 node first with R0 `961.328 us`, R5
`52.613 us`, and R0/R5 `18.27`. KDA recurrent decode is R0 `200.253 us`,
R5 `45.419 us`, R0/R5 `4.41`; merged front is R0 `195.560 us`, R5
`117.096 us`, R0/R5 `1.67`. Per-node R6/necessary-share fields are null in
this fixed prediction.

Kernel drill-down maps the dominant node to backend
`sglang_trtllm_mxfp4`, shape hidden `3584`, intermediate `3072`, `112` local
experts, top-k `2`, MXFP4 E2M1/UE8M0 weights, and reports
`has_cached_alternative=false`. Dense alternatives exist for the merged front
(`has_cached_alternative=true`), but the live profiler confirms the dominant
source-level leaf is the TRT-LLM MXFP4 BMM.

The exact diagnostic profile was written to `profile_B*.json`, and the
combined driver record is `run.json`. The live B=512 table has the routed
MXFP4 BMM at `769.556 us`, KDA fused decode at `93.937 us`, and the front
bf16 GEMM at `118.516 us`. The B=256 and B=128 profiles use the same routed
MXFP4 family.

Plain fixed-flag graph references, which are the timing references for this
iteration, are B=512 `677.408 us`, B=256 `482.784 us`, and B=128 `378.400 us`.
The golden output and post-step state were captured in
`/tmp/golden_B{512,256,128}_L8192.pt`.

### result.json

```
{
  "mode": "baseline",
  "plain_graph_latency_us": {
    "512,8192": 677.40797996521,
    "256,8192": 482.7840030193329,
    "128,8192": 378.3999979496002
  },
  "golden": [
    "/tmp/golden_B512_L8192.pt",
    "/tmp/golden_B256_L8192.pt",
    "/tmp/golden_B128_L8192.pt"
  ],
  "checks": "golden captured; no post-edit replay"
}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# Iteration 01 hypothesis

For K3's exact local-top-2, 112-local-expert, hidden-3584,
intermediate-3072 routed shape, set the TRT-LLM MXFP4 autotuner ceiling to
`2 * next_power_of_2(B)` when B is at least 256. The routed expert work sees
approximately 512 rows at B=256 and 1024 rows at B=512, so the larger bucket
could choose a better large-m tactic. B=128 retains its existing ceiling, and
all non-K3 shapes and prefill paths retain their existing behavior. This is a
tactic/workspace selection change only; routing tensors, expert weights,
activation values, output storage, and KDA state updates are unchanged.

### analysis.md

# Iteration 01: large-batch tuner ceiling

The candidate was selected from the initial VibeSim target
`unified.kda.moe.mxfp4_fused_moe` and from the live routed BMM profile. It
builds on `/workspace/opt_history/trials/kda_11` (which only tested a larger
ceiling at B=128) and applies the same idea to this task's new B=256/512
large-row regime.

After the diagnostic profile, `simulate?prediction=.:after` returned the same
fixed prediction `p_1066e0f8b4d148de9d40979e8ca7f1e5`. The required
`analyze` levels, `optimality?scope=iter`, and the MXFP4 `kernels` drill-down
were rerun. VibeSim still models MXFP4 fused MoE at `961.328 us` / `57.15%`
with R0/R5 `18.27` and `has_cached_alternative=false`; recurrent decode is
still R0/R5 `4.41`. The fixed service reports no post-edit model delta.

The profiler confirms the candidate still launches the same
`bmm_MxE4m3_MxE2m1MxE4m3...t128x8x512...dynB...` and
`bmm_Bfloat16_MxE2m1MxE4m3...t128x8x512u2...dynB...` kernels for every point.
The tuning-ceiling edit therefore did not expose a new kernel implementation;
the diagnostic measurements remain noisy/perturbed as expected and are not
used as the timing reference.

### result.json

```
{
  "before_plain_graph_latency_us": {
    "512,8192": 677.40797996521,
    "256,8192": 482.7840030193329,
    "128,8192": 378.3999979496002
  },
  "after_plain_graph_latency_us": {
    "512,8192": 679.423987865448,
    "256,8192": 480.80000281333923,
    "128,8192": 378.36799025535583
  },
  "checks": [
    {"point": "512,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "256,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "accepted": false,
  "reason": "B=512 regressed; B=256 gain was below 0.5%; same dynamic-batch kernel family remained active."
}
```

(diff.patch: 33 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_02
### hypothesis.md

# Iteration 02 hypothesis

Use the accepted local EP8 route+quant implementation from `mla_20` for the
K3 local shape, but cap it to B=512 so B=256 and B=128 stay on the seeded
path. The local JIT uses the same radix and quantization device code and
therefore produces the same route ids, fp32 weights, TRT-LLM packed ids, FP8
activations, and UE8M0 scales; it does not touch expert weights, layer output,
or KDA recurrent/conv/MLA state.

### analysis.md

# Iteration 02: local route+quant specialization probe

This iteration built on accepted `/workspace/opt_history/trials/mla_20`, which
added a 112-expert/top-2 route trait, and ruled out the wrong-shape KDA probe
in `/workspace/opt_history/trials/kda_23` (896 experts/top-2, never active).

After the profile, `simulate?prediction=.:after` returned the fixed prediction
`p_1066e0f8b4d148de9d40979e8ca7f1e5`. Operator analysis still models
`unified.kda.moe.mxfp4_fused_moe` at `961.328 us` / `57.15%`; the iteration
ladder remains R0 `961.328 us`, R5 `52.613 us`, R0/R5 `18.27`, with no
per-node necessary-share estimate and no cached alternative. The run summary
remains optimality/necessary ratio `0.1825`, batching gap `62.91%`, and
hardware gap `17.27%`. The MXFP4 kernel drill-down is unchanged.

The actual profile is decisive: B=512, B=256, and B=128 still launch
`_router_triton_kernel` plus `per_token_group_quant_flat_kernel`; no
`route_quant_fused_kernel` appears. The driver reaches the global routing
shape, so the local 112/top-2 wrapper is not eligible in this path.

Plain replay was B=512 `677.344 us`, B=256 `482.752 us`, and B=128
`378.368 us`, versus `677.408 / 482.784 / 378.400 us` before. All checks
were exact, but the differences are below noise and the intended kernel was
inactive.

### result.json

```
{
  "before_plain_graph_latency_us": {
    "512,8192": 677.40797996521,
    "256,8192": 482.7840030193329,
    "128,8192": 378.3999979496002
  },
  "after_plain_graph_latency_us": {
    "512,8192": 677.344024181366,
    "256,8192": 482.7519953250885,
    "128,8192": 378.36799025535583
  },
  "checks": [
    {"point": "512,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "256,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "accepted": false,
  "reason": "The live profile stayed on the unfused router/quant chain; no route_quant_fused kernel launched."
}
```

(diff.patch: 422 lines, files: python/sglang/kernels/jit/csrc/moe/route_quant_fused.cuh, python/sglang/kernels/jit/csrc/moe/route_radix.cuh, python/sglang/kernels/ops/moe/moe_route_quant_fused.py, python/sglang/srt/layers/moe/topk.py)

## iter_03
### hypothesis.md

The previous local 112/top-2 route+quant specialization was based on the
measured shape and preserved the expert preparation outputs, but the profile
showed that it never launched. This iteration tested the narrower prerequisite
that the existing fused wrapper was simply rejecting the larger decode token
count. The cap was raised to 512, retaining the original 896/top-16 kernel and
all other guards. Since the live shape is not that wrapper's expert/top-k
specialization, the change was expected to be neutral unless the generic
handoff could consume it; the profile showed that it could not.

This trial builds on the route+quant work documented in prior `kda_23` and
`mla_20` trials and rules out token-cap selection as the missing activation
condition.

### analysis.md

Iteration 03 measured the existing K3 fused route+quant wrapper with its token
cap raised from 64 to 512. VibeSim was run after the diagnostic profile with
`prediction=.:after`; the fixed prediction was
`p_1066e0f8b4d148de9d40979e8ca7f1e5` at
`/opt/vibesim/repo/logs/k3_kda_b512`.

The VibeSim operator breakdown is unchanged: `unified.kda.moe.mxfp4_fused_moe`
is 0.961327 ms (57.15%), followed by KDA recurrent decode at 0.200253 ms
(11.90%) and merged front at 0.195559 ms (11.63%). Its iteration ladder is
R0=961.328 us, R5=52.613 us, R0/R5=18.27; the run summary reports
R0=1.682106 ms, R5=R6=0.307007 ms, necessary share 0.182513, with 62.91%
batching gap and 17.27% hardware gap. The kernel drill-down has no cached
alternative for the routed MXFP4 leaf.

The per-kernel table is decisive for this trial: all three points still launch
the generic router and `per_token_group_quant_flat_kernel`; no
`route_quant_fused_kernel` appears. The source wrapper's 896-expert/top-16
coverage therefore does not match the live 112-expert/top-2 rank-local shape.
Raising only `_MAX_TOKENS` did not change dispatch and is rejected.

### result.json

```
{
  "before_plain_graph_latency_us": {
    "512,8192": 677.40797996521,
    "256,8192": 482.7840030193329,
    "128,8192": 378.3999979496002
  },
  "after_plain_graph_latency_us": {
    "512,8192": 677.3120164871216,
    "256,8192": 482.7840030193329,
    "128,8192": 378.36799025535583
  },
  "checks": [
    {"point": "512,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "256,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "accepted": false,
  "reason": "The live profile stayed on the generic router and quantization chain; no fused route+quant kernel launched."
}
```

(diff.patch: 0 lines, files: )
