# KDA trial 24: agent iterations

## iter_00
### hypothesis.md

# Iteration 00 Hypothesis

Route Kimi K3's fused `[q, k, v, g]` BF16 projection through the existing
CuTe-TGV BF16 GEMM for the production per-rank shape `[N=6144, K=7168]` and
decode batches up to 128.

The TGV kernel computes the same dense BF16 linear operation with FP32
accumulation and BF16 output. The edit is guarded by BF16 dtype, contiguous
input, the exact K3 weight shape, CUDA execution, and the small decode batch
range. It does not change routing, MXFP4 expert weights, KDA gate math,
recurrent-state writes, or any tensor shape. The replay golden will verify both
layer output and post-step state at B=128, 32, and 1.

### analysis.md

# Iteration 00 Analysis

Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54` (`vibesim:k3_kda`), built with
`simulate?prediction=.:before`.

VibeSim predicted 417 us for B=128, 245 us for B=32, and 121 us for B=1.
The run summary was 0.78295 ms of measured kernel time, with a 0.06012 ms
hardware necessary floor and an optimality ratio of 0.0768. Its largest gap
bucket was hardware gap (61.1%), followed by batching (29.0%).

The iteration optimality ladder, sorted by `R0-R5`, selected:

| node | R0 measured | R5 limit | R0/R5 | necessary share |
| --- | ---: | ---: | ---: | ---: |
| `moe.mxfp4_fused_moe` | 412.4 us | 9.45 us | 43.62x | null (not segmented) |
| `attention.qkvbfg` | 51.86 us | 17.32 us | 2.99x | null (not segmented) |
| `attention.kda_recurrent_decode` | 42.50 us | 8.16 us | 5.21x | null (not segmented) |
| `moe.merged_front` | 120.18 us | 91.11 us | 1.32x | null (not segmented) |

The MXFP4 node is the largest, but `kernels` reports only
`sglang_trtllm_mxfp4` and `has_cached_alternative=false`; its implementation is
the external FlashInfer TRT-LLM kernel. The qkv node reports
`has_cached_alternative=true`, maps to the BF16 `fused_qkvg_proj` in
`python/sglang/srt/models/kimi_k3.py`, and its B=128 profiler entry is the
51.542 us `nvjet_sm100_tss_240x64_64x9...` kernel. That is the first actionable
source-level alternative with meaningful ladder headroom. The custom KDA
recurrent kernel is already present and measured at 27.207 us in the profiler,
so it is not the first target despite its generic VibeSim estimate.

Baseline plain graph reference (identical fixed flags, no diagnostic phases):

| point | latency_us |
| --- | ---: |
| B=128, L=8192 | 379.4240 |
| B=32, L=8192 | 242.2080 |
| B=1, L=8192 | 126.4320 |

The split/profile invocation is retained in `profile.json` and the
`profile_B*_L*.json` files; it is diagnostic only.

After the qkv candidate profile, VibeSim was simulated again with the same
idempotent before-state prediction and the same analysis verbs. The prediction
remained `p_38ccb83a6c4e44298f45b2251bd4cf54`; the operator ladder and cached
alternative report were unchanged. The corrected profiler table showed the
TGV qkv kernel active: the `TgvGemm...cta64x128...` entry count increased from
1 to 2 at B=128 (and likewise at smaller batches). This verified dispatch, but
the B=128 graph critical path did not move.

Candidate decision: reject. The primary point was 379.424 us before versus
380.288 us after in identical plain replay commands; B=32 and B=1 improved to
239.040 us and 124.384 us, respectively, but the primary requirement was not
met. All three CHECK records passed, including `state_ok=true`.

### result.json

```
{
  "candidate": "K3 fused qkv BF16 CuTe-TGV dispatch",
  "accepted": false,
  "before_latency_us": {
    "128,8192": 379.42400574684143,
    "32,8192": 242.20800399780273,
    "1,8192": 126.43200159072876
  },
  "after_latency_us": {
    "128,8192": 380.2880048751831,
    "32,8192": 239.04000222682953,
    "1,8192": 124.38400089740753
  },
  "checks": [
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "32,8192", "max_rel_err": 0.019138754515693437, "state_ok": true, "pass": true},
    {"point": "1,8192", "max_rel_err": 0.005813952947539261, "state_ok": true, "pass": true}
  ],
  "profile": {
    "B128_tgv_kernel_count": 2,
    "B128_tgv_kernel_avg_us": 22.41120000000003,
    "B128_total_kernel_us": 414.3738000000004,
    "B128_graph_profile_us": 347.5840091705322
  }
}
```

(diff.patch: 53 lines, files: iter_00/candidate/kimi_k3.py)

## iter_01
### hypothesis.md

# Iteration 01 Hypothesis

Use the existing CuTe-TGV BF16 GEMM for Kimi K3's fused MoE front when its
output is requested as FP32 and the weight is the exact decode front shape
`[15984, 7168]` (also accept the production `16768`-row shape). The front
already computes `hidden_states @ weight.T` with BF16 inputs and FP32 output;
the TGV path preserves both dtypes and the same output shape, so router logits,
MXFP4 activation quantization, routing decisions, and all post-step state
contracts remain unchanged except for ordinary GEMM accumulation ordering.

The guard is limited to CUDA SM100, contiguous BF16 inputs, and decode batches
through 128. All other shapes and output dtypes use the existing dispatch.

The follow-up tactic-15 variant selected a 1-CTA 128x128 tile for the front.
It preserved outputs and state but regressed all measured graph points,
especially B=1, so it is rejected independently.

### analysis.md

# Iteration 01 Analysis

Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54` (`vibesim:k3_kda`), rebuilt by
`simulate?prediction=.:before` after the fresh baseline profile.

The prediction is the fixed Kimi-K3 KDA B200 layer model. Its operator shares
are unchanged from iter_00: MXFP4 routed MoE is 52.67% of predicted kernel
time, `merged_front` is 15.35%, qkv is 6.62%, and recurrent decode is 5.43%.
The run summary gives `R0=0.78295 ms`, hardware necessary `R5=0.06012 ms`,
optimality ratio 0.0768, with hardware gap 61.1% and batching gap 29.0%.

The optimality ladder reports:

| node | R0 measured | R5 limit | R0/R5 | necessary share |
| --- | ---: | ---: | ---: | ---: |
| `moe.mxfp4_fused_moe` | 412.4 us | 9.45 us | 43.62x | null |
| `attention.qkvbfg` | 51.86 us | 17.32 us | 2.99x | null |
| `attention.kda_recurrent_decode` | 42.50 us | 8.16 us | 5.21x | null |
| `moe.merged_front` | 120.18 us | 91.11 us | 1.32x | null |

`kernels?kernel_set=unified.kda.moe.merged_front` maps to
`sglang_bf16_auto`, shape `n=16768, k=7168, dtype=bf16`, and reports
`has_cached_alternative=true`. The actual driver profile uses the local
`[15984, 7168]` front weight and launches the `nvjet_sm100_tss...` GEMM at
51.606 us for B=128. The external MXFP4 node has no cached alternative, so
the merged-front dispatch is the next source-level candidate.

Fresh plain graph baseline (fixed timing flags, no diagnostic phases):

| point | latency_us |
| --- | ---: |
| B=128, L=8192 | 379.3920 |
| B=32, L=8192 | 243.1360 |
| B=1, L=8192 | 126.4320 |

## Post-edit measurement and re-analysis

After the front dispatch edit, the diagnostic profile launched the expected
FP32 TGV kernel for the fused front: `cta64x128x128_2cta1...outfloat32`,
averaging 50.896 us at B=128 versus 51.606 us for the baseline NVJet
kernel. The full profile summed 415.1 us of kernels and reported a 345.6 us
diagnostic graph replay. The VibeSim prediction was rebuilt/fetched again
with `simulate?prediction=.:before`, then `analyze`/`optimality`/`kernels`
were rerun; it still reports `merged_front` as 120.18 us predicted time,
R0/R5=1.32x, and a cached alternative. The larger MXFP4 node remains the
dominant predicted node, but it has no cached alternative.

The clean plain replay with the front dispatch measured 378.368, 241.120,
and 124.384 us for B=128, 32, and 1 respectively. All three replay checks
passed with max relative error 0 and `state_ok=true`. The primary gain was
0.27%, below the 0.5% acceptance threshold, so this candidate is rejected.

The additional tactic-15 experiment was also numerically exact but measured
380.448, 245.216, and 160.256 us; its B=1 regression confirms that tactic
must not be retained.

### result.json

```
{
  "candidate": "K3 fused front FP32 CuTe-TGV dispatch",
  "accepted": false,
  "before_latency_us": {
    "128,8192": 379.39199805259705,
    "32,8192": 243.13600361347198,
    "1,8192": 126.43200159072876
  },
  "after_latency_us": {
    "128,8192": 378.36799025535583,
    "32,8192": 241.11999571323395,
    "1,8192": 124.38400089740753
  },
  "checks": [
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "32,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "1,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "tactic15_variant": {
    "accepted": false,
    "latency_us": {
      "128,8192": 380.44801354408264,
      "32,8192": 245.2159970998764,
      "1,8192": 160.25599837303162
    },
    "checks_passed": true
  },
  "profile": {
    "B128_front_tgv_avg_us": 50.8958,
    "B128_baseline_nvjet_avg_us": 51.606,
    "B128_front_profile_graph_us": 345.6000089645386,
    "B128_front_profile_kernel_sum_us": 415.1
  }
}
```

(diff.patch: 20 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_02
### hypothesis.md

# Iteration 02 Hypothesis

For BF16 KDA state, select the existing 3-stage TMA specialization when
`B * H >= 512`, matching the established float-state occupancy threshold.
At the production K3 shape H=12 this changes only B=128 (1536 blocks) from
64 KiB to 48 KiB of dynamic state staging, which should improve resident
block capacity; B=32 and B=1 retain the current four-stage path.

The template changes only state staging and synchronization. It does not
change the recurrence arithmetic, state addresses, output stores, conv-cache
updates, or any model-level ordering, so output and post-step state should
remain within the existing BF16 tolerance. The source guard is limited to
the already-selected K3 fused decode wrapper and the BF16 state path.

### analysis.md

# Iteration 02 Analysis

Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54` (`vibesim:k3_kda`). The
fresh diagnostic profile was obtained with the fixed workload and then routed
through `simulate?prediction=.:before`; `analyze` was run at operator and
run-summary levels, followed by `optimality?scope=iter` and the recurrent
`kernels` drill-down.

VibeSim reports total predicted kernel time 0.782953 ms. The dominant
MXFP4 fused MoE is 0.412401 ms (52.67%, R0/R5=43.62x), but its kernel detail
has no cached alternative. The KDA recurrent node is 0.0424959 ms (5.43%,
R0/R5=5.21x), with a 0.008161 ms hardware limit and no cached alternative;
the ladder therefore identifies an implementation-efficiency opportunity.
The run summary reports a 0.060122 ms hardware-necessary floor, 61.14%
hardware gap, and 29.01% batching gap.

The B=128 profile confirms the source mapping. It launches
`kda_decode_fusion_many_heads_kernel<..., 4, true, __nv_bfloat16>` once at
27.0944 us. The K3 host wrapper in
`python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh` selects four
TMA state stages for every BF16 state, even though the same wrapper already
selects three stages for a sufficiently large float-state grid. The measured
front GEMM and MoE kernels are larger, but the front dispatch experiment in
iter_01 did not clear the primary acceptance threshold, so the recurrent
kernel is the next direct source-level target.

Baseline plain graph replay captured before editing:

| point | latency_us |
| --- | ---: |
| B=128, L=8192 | 379.3920 |
| B=32, L=8192 | 243.1040 |
| B=1, L=8192 | 126.4000 |

## Post-edit measurement and re-analysis

The edit selected the intended BF16 3-stage specialization at B=128, while
B=32 and B=1 remained on the 4-stage specialization. VibeSim was rebuilt with
`simulate?prediction=.:before` after the profile and all analysis verbs were
rerun; its fixed roofline prediction is unchanged, so the recurrent node
remains an efficiency opportunity at R0/R5=5.21x with no cached alternative.

The recurrent kernel profile measured 27.155 us for 3 stages at B=128,
slightly slower than the 27.094 us 4-stage baseline. Plain replay measured
378.368, 242.144, and 126.400 us for B=128, 32, and 1. All checks passed
with max relative error 0, no NaN, and `state_ok=true`, but the primary
improvement was only 0.27%, below the required 0.5% threshold. Reject this
staging choice and retain the original four-stage BF16 path.

### result.json

```
{
  "candidate": "KDA BF16 recurrent decode 3-stage TMA at B*H >= 512",
  "accepted": false,
  "before_latency_us": {
    "128,8192": 379.39199805259705,
    "32,8192": 243.1039959192276,
    "1,8192": 126.39999389648438
  },
  "after_latency_us": {
    "128,8192": 378.36799025535583,
    "32,8192": 242.14400351047516,
    "1,8192": 126.39999389648438
  },
  "checks": [
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "32,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "1,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "profile": {
    "B128_baseline_recurrent_us": 27.0944,
    "B128_candidate_recurrent_us": 27.1550,
    "B128_candidate_template_stages": 3,
    "B32_candidate_template_stages": 4,
    "B1_candidate_template_stages": 4
  }
}
```

(diff.patch: 11 lines, files: python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh)

## iter_03
### hypothesis.md

# Iteration 03 Hypothesis

Reduce the KDA fused decode block from 256 to 128 threads. The grid remains
one block per `(decode token, value head)`, and the existing warp-strided
loops cover all 32 state rows with the new `kRowsPerWarp` value. The change
should reduce per-block thread and register resources while preserving every
BF16 state load/store, reduction, output, and cache update.

The measured profile disproved the performance assumption: the smaller block
does more work per warp and is slower at every tested batch. All checks still
passed, so the variant is rejected on latency only.

### analysis.md

# Iteration 03 Analysis

Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54` (`vibesim:k3_kda`). The
fresh baseline and candidate profiles were each followed by
`simulate?prediction=.:before`, operator/run-summary analysis, the iteration
optimality ladder, and the recurrent kernel drill-down. VibeSim continues to
identify the recurrent node as a necessary-work efficiency opportunity
(R0/R5=5.21x, no cached alternative), while the MXFP4 MoE remains the larger
overall node but is not source-swappable through the provided alternatives.

The baseline source uses `kThreads=256` in
`python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh`. This
iteration changed only that compile-time block size to 128. The candidate
replay was numerically exact, but plain graph latency changed from the fresh
baseline 380.320, 243.104, 126.432 us to 381.440, 244.128, 126.464 us for
B=128, 32, and 1. The candidate profile measured the fused recurrent kernel
at 29.638, 14.192, and 8.870 us respectively, versus the 256-thread baseline
27.094, 12.406, and 7.766 us in the comparable profile. The candidate is
rejected and 256 threads must remain.

### result.json

```
{
  "candidate": "KDA fused decode kThreads 256 to 128",
  "accepted": false,
  "before_latency_us": {
    "128,8192": 380.3200125694275,
    "32,8192": 243.1039959192276,
    "1,8192": 126.43200159072876
  },
  "after_latency_us": {
    "128,8192": 381.44001364707947,
    "32,8192": 244.1280037164688,
    "1,8192": 126.46399438381195
  },
  "checks": [
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "32,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "1,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "profile": {
    "recurrent_us": {
      "before": {"128,8192": 27.0944, "32,8192": 12.4064, "1,8192": 7.7664},
      "after": {"128,8192": 29.6381, "32,8192": 14.1920, "1,8192": 8.8702}
    }
  }
}
```

(diff.patch: 11 lines, files: python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh)

## iter_04
### hypothesis.md

# Iteration 04 Hypothesis

Set `kUseActiveQkReduction=true` in the K3 fused decode configuration. The
active implementation reduces exactly the four warps covering the 128 Q/K
elements and supplies zero for no additional warps; the full reduction
currently contributes only zeros from those remaining warps. This changes
reduction scheduling, not the arithmetic inputs, recurrence, state stores, or
output/state semantics. The expected gain is a shorter Q/K normalization
setup with identical BF16 outputs and post-step state.

### analysis.md

# Iteration 04 Analysis

Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54` (`vibesim:k3_kda`). The
current baseline profile was routed through `simulate?prediction=.:before`,
then operator/run-summary analysis, `optimality?scope=iter`, and the
recurrent kernel drill-down. The relevant node remains
`unified.kda.attention.kda_recurrent_decode`: R0=42.50 us, R5=8.16 us,
R0/R5=5.21x, no cached alternative. The overall predicted dominant node is
the 412.4 us MXFP4 fused MoE, also with no cached alternative.

The B=128 profiler confirms the current K3 recurrent launch is the 256-thread,
four-stage BF16 fused kernel at 27.09 us. In its Q/K normalization, only the
first four of eight warps have nonzero inputs (`kDimK=128`, `kThreads=256`),
but the selected `block_reduce_sum2` still performs warp reductions for all
eight. The source already contains `block_reduce_sum2_active_for`, so the
next hypothesis is a compile-time selection of the four active warps.

## Post-edit measurement and re-analysis

After the edit, the profile was routed through `simulate?prediction=.:before`
and the full analysis set again. The candidate kernel name contains
`kUseActiveQkReduction=true` and measured 27.0976 us at B=128 versus 27.1100
us for the baseline profile, a 0.0124 us difference. Plain replay measured
378.496, 243.136, and 126.432 us for B=128, 32, and 1. All checks passed
with zero relative error and `state_ok=true`, but the primary change was only
0.48% against the fresh 380.320 us run and is not a robust 0.5% gain. Reject
the specialization and keep the baseline reduction setting.

### result.json

```
{
  "candidate": "KDA active four-warp QK reduction",
  "accepted": false,
  "before_latency_us": {
    "128,8192": 380.3200125694275,
    "32,8192": 243.1039959192276,
    "1,8192": 126.43200159072876
  },
  "after_latency_us": {
    "128,8192": 378.495991230011,
    "32,8192": 243.13600361347198,
    "1,8192": 126.43200159072876
  },
  "checks": [
    {"point": "128,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "32,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": "1,8192", "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "profile": {
    "B128_baseline_recurrent_us": 27.1100,
    "B128_candidate_recurrent_us": 27.0976,
    "candidate_template_active_qk": true
  }
}
```

(diff.patch: 11 lines, files: python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh)
