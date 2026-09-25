# MLA trial 26: agent iterations

## iter_00
### hypothesis.md

The next iteration will specialize the local EP8 route+quant kernel for
112 experts and top-k 2 with a 128-thread router block. The quant half will
use 56 two-lane pairs, each processing two independent 32-element groups.
The same radix comparisons, tie handling, sigmoid values, normalization,
MXFP8 quantization, scales, and packed routing stores remain in place, so
the layer output and KV post-step state should be unchanged. Only B <= 64
uses this fused local path; B=128 continues through its existing path.

### analysis.md

Baseline VibeSim analysis

Measurement
- Fixed diagnostic command: Kimi-K3 MLA, FlashInfer MXFP4, CuteDSL MLA,
  12 heads, FP8 E4M3 KV, 112 experts, EP8, local top-k 2, CUDA graph,
  40 iterations and 10 warmups.
- Graph latency from the plain fixed command: B=1/L=1048576 253.472 us,
  B=128/L=8192 476.608 us, B=16/L=65536 290.240 us.
- Diagnostic profile graph latency: 254.464/470.592/291.360 us for the
  same points. Profile kernel sums were 320.1/548.8/356.5 us.

VibeSim
- simulate selector: prediction=.:before
- prediction_id: p_c7687e585abe4b5fbacdc89caf0c9761
- API detail: this service accepts the returned identifier as
  prediction_id for analyze/optimality/kernels.
- operator analysis: aggregate predicted kernel time is 1.158900 ms.
  The largest operators are mxfp4_fused_moe 0.379161 ms (32.72%),
  mla_decode_attention 0.358836 ms (30.96%), and merged_front
  0.158805 ms (13.70%).
- run summary: R0=1.158899 ms, R5=0.147162 ms, R6=0.147162 ms,
  optimality/necessary ratio=0.126985. The aggregate gaps are batching
  32.79%, hardware gap 36.88%, and imbalance 17.64%.
- iteration optimality, largest leaves:
  - unified.mla.moe.mxfp4_fused_moe: R0=379.160 us, R5=8.573 us,
    R6/R7=227.756 us, necessary_share=0.600686, no cached alternative.
  - unified.mla.attention.mla_decode_attention: R0=358.836 us,
    R5=227.559 us, R6/R7=454.164 us, necessary_share=1.265660,
    cached alternative present.
  - unified.mla.moe.merged_front: R0=158.805 us, R5=121.067 us,
    R6/R7=30.048 us, necessary_share=0.189216.
- Per-leaf kernel drill-down confirms both candidate families. MLA decode
  has backend sglang_cutedsl_mla and has_cached_alternative=true. MXFP4
  fused MoE has backend sglang_trtllm_mxfp4 and no cached alternative.

Source/profiler mapping
- The B=1 profiler launches
  fmhaSm100fKernel_* (132.686 us), the local
  route_quant_fused_kernel<LocalRouterRadixTrait,...> (8.121 us), and
  the FP8 set_mla_kv_concat_q kernel (6.659 us). B=16 launches the same
  local fused route/quant kernel (4.208 us); B=128 uses the external
  routing/finalize path because the local fusion token cap is exceeded.
- The local route+quant launch is produced by
  kernels/ops/moe/moe_route_quant_fused.py and
  kernels/jit/csrc/moe/route_quant_fused.cuh. Its 224-thread block has
  112 active routing threads plus 112 threads needed only because the
  quant half assigns one two-lane pair to each of 112 groups.

Decision

The MLA leaf is large but its R6 necessary estimate is already above R0,
and its alternative is an external cached backend. The local fused
route+quant path is a source-controlled launch with redundant inactive
router threads and affects the scored B=1 shape without touching the
B=128 path. The next hypothesis preserves every per-expert and per-group
operation and only changes independent thread scheduling.

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

Hypothesis

Specialize the local EP8 route+quant launch for the measured 112-expert,
top-k-2 shape. Use a 128-thread CTA for radix selection, where 112 lanes
are valid experts and 16 participate only in the existing block barriers.
Use the 64 available two-lane quant pairs as 56 active pairs, with each
active pair quantizing two independent groups (0..55 and 56..111).

The radix key/activation loads, four radix rounds, deterministic tie
handling, sigmoid values, top-k renormalization, FP8 values/scales, and
packed indices are unchanged. Quant groups do not depend on one another,
so processing the second group in the same pair cannot change any result.
The dispatch remains gated to N=112/K=2 and M<=64; the B=128 workload
continues through its original backend path.

### analysis.md

Iteration 01 analysis

- Prediction: p_c7687e585abe4b5fbacdc89caf0c9761 from
  simulate?prediction=.:before. The service is a fixed before-state model,
  so its roofline values are unchanged from iter_00.
- The selected VibeSim leaf remains
  unified.mla.moe.mxfp4_fused_moe (R0=379.160 us, R5=8.573 us,
  R6/R7=227.756 us, necessary_share=0.600686, no cached alternative).
  MLA decode remains second (R0=358.836 us, R5=227.559 us,
  R6/R7=454.164 us, necessary_share=1.265660, cached alternative).
- The profiler confirmed the changed local route_quant_fused kernel still
  launched at B=1 and B=16, while B=128 stayed on its original routing
  path. In the diagnostic B=1 profile its measured time was 7.99 us,
  only slightly below the 8.12 us baseline; the graph replay did not
  benefit from that small kernel change.

Replay and decision

- CHECK passed exactly at every point: max_abs_err=0, max_rel_err=0,
  nan=false, state_ok=true.
- Plain replay latency changed 253.472 -> 261.504 us at B=1/L=1048576,
  476.608 -> 476.704 us at B=128/L=8192, and
  290.240 -> 288.256 us at B=16/L=65536.
- Reject this hypothesis. The 128-thread router/paired quant schedule is
  numerically safe but slower on the scored long-context replay, so the
  source edit will be reverted before the next iteration.

### result.json

```
{
  "iteration": "01",
  "hypothesis": "128-thread local route+quant CTA with two quant groups per active pair",
  "baseline_latency_us": {
    "1,1048576": 253.4720003604889,
    "128,8192": 476.608008146286,
    "16,65536": 290.23998975753784
  },
  "replay_latency_us": {
    "1,1048576": 261.50399446487427,
    "128,8192": 476.7040014266968,
    "16,65536": 288.2559895515442
  },
  "checks": [
    {
      "point": [
        1,
        1048576
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        128,
        8192
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        16,
        65536
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    }
  ],
  "accepted": false,
  "reason": "B=1 long-context replay increased by 3.17%; exact numerics passed.",
  "diagnostic_profile_latency_us": {
    "1,1048576": 253.4720003604889,
    "128,8192": 487.87200450897217,
    "16,65536": 291.26399755477905
  },
  "diagnostic_profile_route_quant_us": {
    "1,1048576": 7.9935,
    "16,65536": null
  },
  "prediction_id": "p_c7687e585abe4b5fbacdc89caf0c9761"
}
```

(diff.patch: 34 lines, files: b/python/sglang/kernels/jit/csrc/moe/route_quant_fused.cuh)

## iter_02
### hypothesis.md

Hypothesis

Use 512 threads per block for the Kimi-K3 MLA output-gate kernel instead
of 256. The kernel processes eight BF16 values per thread, so B=1 changes
from four blocks to two blocks for the same 7168-element output. B=16 and
B=128 retain the same arithmetic and grid work shape modulo block packing.

This is only a launch geometry change. Each output element still computes
the same FP32 sigmoid, rounds it to BF16, multiplies in FP32, and rounds
the result to BF16; no layer input, KV write, or post-step state is changed.

### analysis.md

Iteration 02 analysis

- Prediction: p_c7687e585abe4b5fbacdc89caf0c9761 from
  simulate?prediction=.:before. The fixed VibeSim prediction still ranks
  mxfp4_fused_moe first (R0=379.160 us, R6/R7=227.756 us,
  necessary_share=0.600686) and MLA decode second (R0=358.836 us,
  R6/R7=454.164 us, necessary_share=1.265660).
- The real profile launched mla_output_gate_kernel<512,true> at every
  point. B=1's four blocks were reduced to two, but the graph replay did
  not improve.
- CHECK passed exactly at every point: max_abs_err=0, max_rel_err=0,
  nan=false, state_ok=true.
- Plain replay latency changed 253.472 -> 261.600 us at B=1/L=1048576,
  476.608 -> 476.736 us at B=128/L=8192, and
  290.240 -> 288.288 us at B=16/L=65536. The diagnostic profile was
  within normal run-to-run variation (253.440/473.536/291.232 us).

Decision

Reject and revert the 512-thread geometry. The block-count reduction is
not useful on the scored graph path; the exact baseline kernel is retained.

### result.json

```
{
  "iteration": "02",
  "hypothesis": "512-thread MLA output-gate blocks",
  "baseline_latency_us": {
    "1,1048576": 253.4720003604889,
    "128,8192": 476.608008146286,
    "16,65536": 290.23998975753784
  },
  "replay_latency_us": {
    "1,1048576": 261.59998774528503,
    "128,8192": 476.73600912094116,
    "16,65536": 288.2879972457886
  },
  "checks": [
    {
      "point": [
        1,
        1048576
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        128,
        8192
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        16,
        65536
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    }
  ],
  "diagnostic_profile_latency_us": {
    "1,1048576": 253.4399926662445,
    "128,8192": 473.53601455688477,
    "16,65536": 291.23198986053467
  },
  "output_gate_kernel": "mla_output_gate_kernel<512,true>",
  "accepted": false,
  "reason": "The plain scored replay showed no improvement and B=1 was 3.20% slower; diagnostic profile was within noise.",
  "prediction_id": "p_c7687e585abe4b5fbacdc89caf0c9761"
}
```

(diff.patch: 13 lines, files: b/python/sglang/kernels/ops/kimi_k3/mla_output_gate.py)

## iter_03
### hypothesis.md

Hypothesis

Run the Kimi-K3 MLA fused-A projection with the tile_n=16 variant even for
the B=1 decode bucket. The current dispatcher chooses tile_n=8 for 1..8
tokens and tile_n=16 above that; the profiler shows the B=1 kernel as
fused_a_gemm_kernel<..., tile_n=8, stage_cnt=16>, while B=16 uses
tile_n=16, stage_cnt=12. The larger output tile keeps the same BF16 MMA
K-loop and accumulation order but may reduce tile/pipeline overhead on
Blackwell.

Only the fused qkv_a projection dispatch changes. The output is still
written by the same BF16 GEMM kernel, with the same input/weight values and
no effect on KV writes or recurrent/cache state. B=128 is outside the
1..16 fused-A coverage and remains unchanged.

### analysis.md

Iteration 03 analysis

- Prediction: p_c7687e585abe4b5fbacdc89caf0c9761 from
  simulate?prediction=.:before. The VibeSim ranking remains
  mxfp4_fused_moe first (R0=379.160 us, R6/R7=227.756 us,
  necessary_share=0.600686) and MLA decode second (R0=358.836 us,
  R6/R7=454.164 us, necessary_share=1.265660).
- The profiler confirmed B=1 now launches
  fused_a_gemm_kernel<1,2112,7168,16,16,256,12>, the same tile used by
  B=16, instead of the baseline tile_n=8/stage_cnt=16 kernel.
- CHECK passed exactly at every point: max_abs_err=0, max_rel_err=0,
  nan=false, state_ok=true.
- Plain replay latency changed 253.472 -> 255.488 us at B=1/L=1048576,
  476.608 -> 476.800 us at B=128/L=8192, and
  290.240 -> 288.256 us at B=16/L=65536.

Decision

Reject and revert. The tile_n=16 schedule helps the B=16 diagnostic
slightly but regresses the scored B=1 graph by 0.80%.

### result.json

```
{
  "iteration": "03",
  "hypothesis": "Use fused-A tile_n=16 for B=1..16 instead of tile_n=8 for B<=8",
  "baseline_latency_us": {
    "1,1048576": 253.4720003604889,
    "128,8192": 476.608008146286,
    "16,65536": 290.23998975753784
  },
  "replay_latency_us": {
    "1,1048576": 255.48800826072693,
    "128,8192": 476.79999470710754,
    "16,65536": 288.2559895515442
  },
  "checks": [
    {
      "point": [
        1,
        1048576
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        128,
        8192
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        16,
        65536
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    }
  ],
  "diagnostic_profile_latency_us": {
    "1,1048576": 255.48800826072693,
    "128,8192": 471.6799855232239,
    "16,65536": 291.3280129432678
  },
  "profiled_kernels": {
    "1,1048576": "fused_a_gemm_kernel<1,2112,7168,16,16,256,12>",
    "16,65536": "fused_a_gemm_kernel<1,2112,7168,16,16,256,12>"
  },
  "accepted": false,
  "reason": "B=1 long-context replay increased by 0.80%; exact numerics passed, but the scored point regressed.",
  "prediction_id": "p_c7687e585abe4b5fbacdc89caf0c9761"
}
```

(diff.patch: 18 lines, files: b/python/sglang/kernels/jit/csrc/gemm/dsv3_fused_a_gemm.cuh)

## iter_04
### hypothesis.md

Hypothesis

Disable the TRT-LLM programmatic dependent-launch flag only for the exact
Kimi-K3 FP8 MLA decode layout used here (12 heads, 128 no-PE dimension,
512 latent rank, 64 rope dimension). The layer's CUDA graph already
captures a fixed dependency order and the MLA attention is on the critical
path after query/KV preparation; removing the attention kernel's PDL wait/
trigger may save control overhead without changing any arithmetic.

The fused query/KV preparation kernels retain their existing PDL chain, and
all other TRT-LLM MLA shapes keep `_ENABLE_PDL`. This changes no tensor
values or state writes.

### analysis.md

Iteration 04 analysis

- Prediction: p_c7687e585abe4b5fbacdc89caf0c9761 from
  simulate?prediction=.:before. VibeSim still identifies the aggregate
  mxfp4_fused_moe leaf as the largest (R0=379.160 us,
  R6/R7=227.756 us, necessary_share=0.600686) and MLA decode as second
  (R0=358.836 us, R6/R7=454.164 us, necessary_share=1.265660).
- The profiler ran the same 28/29/28-launch graph with the exact K3
  layout, but the attention PDL removal did not reduce graph time.
- CHECK passed exactly at every point: max_abs_err=0, max_rel_err=0,
  nan=false, state_ok=true.
- Plain replay latency changed 253.472 -> 257.568 us at B=1/L=1048576,
  476.608 -> 478.816 us at B=128/L=8192, and
  290.240 -> 288.288 us at B=16/L=65536.

Decision

Reject and revert. PDL is beneficial for this kernel's graph scheduling;
the original enabled path is retained.

### result.json

```
{
  "iteration": "04",
  "hypothesis": "Disable attention PDL for exact K3 FP8 MLA shape",
  "baseline_latency_us": {
    "1,1048576": 253.4720003604889,
    "128,8192": 476.608008146286,
    "16,65536": 290.23998975753784
  },
  "replay_latency_us": {
    "1,1048576": 257.56800174713135,
    "128,8192": 478.8160026073456,
    "16,65536": 288.2879972457886
  },
  "checks": [
    {
      "point": [
        1,
        1048576
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        128,
        8192
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        16,
        65536
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    }
  ],
  "diagnostic_profile_latency_us": {
    "1,1048576": 257.53599405288696,
    "128,8192": 473.6959934234619,
    "16,65536": 291.3280129432678
  },
  "accepted": false,
  "reason": "Disabling PDL regressed B=1 by 1.62% and B=128 by 0.46%; exact numerics passed.",
  "prediction_id": "p_c7687e585abe4b5fbacdc89caf0c9761"
}
```

(diff.patch: 26 lines, files: b/python/sglang/srt/layers/attention/trtllm_mla_backend.py)

## iter_05
### hypothesis.md

Hypothesis

Use FlashInfer's native `cute-dsl` MLA decode implementation inside the
`cutedsl_mla` backend instead of forcing the parent wrapper to dispatch
`trtllm-gen`. VibeSim reports a cached MLA alternative, and the benchmark
selects this backend explicitly; the two implementations have the same
paged FP8 MLA math and shape contract.

This is an implementation/backend selection change only. The query,
latent KV cache append, page table, sequence lengths, and output/state
interfaces remain unchanged. The replay CHECK will reject any numerical
drift beyond the harness tolerance.

### analysis.md

Iteration 05 analysis

- Prediction: p_c7687e585abe4b5fbacdc89caf0c9761 from
  simulate?prediction=.:before. The VibeSim optimality output still puts
  mxfp4_fused_moe first (R0=379.160 us, R6/R7=227.756 us,
  necessary_share=0.600686) and MLA decode second (R0=358.836 us,
  R6/R7=454.164 us, necessary_share=1.265660).
- Native CuteDSL compiled and launched a 29-kernel graph, but the primary
  and secondary graph replays were slower: 296.352/484.864/294.336 us.
- CHECK passed the harness threshold at every point, but only B=1 was
  bit-identical. B=128 had max_rel_err=0.013761 and B=16 had
  max_rel_err=0.004852, so the implementation changes numerical output.

Decision

Reject and revert. The TRT-LLM implementation is both faster and the
numerically stable path for this workload.

### result.json

```
{
  "iteration": "05",
  "hypothesis": "Use native cute-dsl MLA decode implementation internally",
  "baseline_latency_us": {
    "1,1048576": 253.4720003604889,
    "128,8192": 476.608008146286,
    "16,65536": 290.23998975753784
  },
  "replay_latency_us": {
    "1,1048576": 296.35199904441833,
    "128,8192": 484.8639965057373,
    "16,65536": 294.3359911441803
  },
  "checks": [
    {
      "point": [
        1,
        1048576
      ],
      "max_abs_err": 0,
      "max_rel_err": 0,
      "mean_rel_err": 0,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        128,
        8192
      ],
      "max_abs_err": 0.1875,
      "max_rel_err": 0.01376146687989234,
      "mean_rel_err": 0.0009116552537307142,
      "nan": false,
      "state_ok": true,
      "pass": true
    },
    {
      "point": [
        16,
        65536
      ],
      "max_abs_err": 0.06640625,
      "max_rel_err": 0.0048515978190613466,
      "mean_rel_err": 0.00035660158027894795,
      "nan": false,
      "state_ok": true,
      "pass": true
    }
  ],
  "diagnostic_profile_latency_us": {
    "1,1048576": 294.40000653266907,
    "128,8192": 479.775995016098,
    "16,65536": 297.5040078163147
  },
  "accepted": false,
  "reason": "Native CuteDSL was slower at every point and produced nonzero numerical drift at B=128/B=16.",
  "prediction_id": "p_c7687e585abe4b5fbacdc89caf0c9761"
}
```

(diff.patch: 13 lines, files: b/python/sglang/srt/layers/attention/cutedsl_mla_backend.py)

## iter_06
### hypothesis.md

Hypothesis

Select the existing fused-A GEMM tile_m=32 implementation for the exact
Kimi-K3 MLA projection shape (hd_in=7168, hd_out=2112). The current
tile_m=16 kernel launches 132 feature tiles for one token; tile_m=32 halves
that grid while using the same BF16 MMA instruction sequence and FP32
accumulation.

The selector remains shape-gated, B=128 is outside fused-A coverage, and
the kernel's output contract and all later KV/query operations are
unchanged. Numerical output and state must pass the golden replay.

### analysis.md

Iteration 06 analysis

VibeSim prediction: p_c7687e585abe4b5fbacdc89caf0c9761

The fixed before-state prediction reports 1.158900 ms at R0, 0.147162 ms at R5/R6,
and an optimality ratio of 0.126985. The largest node remains
`unified.mla.moe.mxfp4_fused_moe`: R0=379.160 us, R5=8.573 us,
R6/R7=227.756 us, necessary_share=0.600686. MLA decode attention is next by
necessary share (R0=358.836 us, R5=227.559 us, R6/R7=454.164 us,
necessary_share=1.265660), and its kernel set has a cached alternative. The
fused-A projection is not a top VibeSim node, so this experiment was a narrow
kernel-table probe rather than a claim that it was the global bottleneck.

The post-edit kernel table launched
`fused_a_gemm_kernel<1,2112,7168,32,8,256,9>` for B=1 and the corresponding
tile_m=32 kernel for B=16. The B=1 fused-A kernel fell from 13.341 us to
12.490 us in the diagnostic table, but graph replay moved only from
253.472 us to 253.408 us (-0.025%), below the 0.5% acceptance floor. B=128
was outside this selector but measured 484.832 us versus 476.608 us in the
fixed baseline, and B=16 measured 286.240 us versus 290.240 us. Because the
primary improvement is not statistically meaningful and a required workload
point regressed, the change is rejected.

All VibeSim artifacts for this iteration are in this directory; the prediction
was simulated before the operator, run-summary, iteration, optimality, and
kernel analyses were read.

### result.json

```
{
  "iteration": "06",
  "hypothesis": "Use tile_m=32 for the exact Kimi-K3 fused-A projection shape.",
  "baseline_latency_us": {
    "1,1048576": 253.4720003604889,
    "128,8192": 476.608008146286,
    "16,65536": 290.23998975753784
  },
  "replay_latency_us": {
    "1,1048576": 253.4080147743225,
    "128,8192": 484.8319888114929,
    "16,65536": 286.24001145362854
  },
  "checks": [
    {"point": [1, 1048576], "max_abs_err": 0, "max_rel_err": 0, "mean_rel_err": 0, "nan": false, "state_ok": true, "pass": true},
    {"point": [128, 8192], "max_abs_err": 0, "max_rel_err": 0, "mean_rel_err": 0, "nan": false, "state_ok": true, "pass": true},
    {"point": [16, 65536], "max_abs_err": 0, "max_rel_err": 0, "mean_rel_err": 0, "nan": false, "state_ok": true, "pass": true}
  ],
  "diagnostic_profile_latency_us": {
    "1,1048576": 253.4080147743225,
    "128,8192": 471.71199321746826,
    "16,65536": 297.2800135612488
  },
  "accepted": false,
  "reason": "The primary change was below the required 0.5% gain and B=128 regressed in replay.",
  "prediction_id": "p_c7687e585abe4b5fbacdc89caf0c9761"
}
```

(diff.patch: 10 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/gemm/dsv3_fused_a_gemm.cuh)

## iter_07
### hypothesis.md

Hypothesis

The VibeSim MoE node is the largest necessary-share node, and the live kernel
table shows the small-batch shared-down BF16 GEMM on the MoE critical path.
The generic TGV selector chooses tactic 18 for both `(M,N,K)=(1,7168,6144)`
and `(16,7168,6144)`. A direct CUDA-event sweep of the same TGV kernels found
tactic 6 fastest at M=1 and tactic 4 fastest at M=16, while the latent-up
`(7168,3584)` shape still prefers the existing tactic 18.

Add exact shape and batch guards to select those two measured tactics. The
change only selects an existing BF16 GEMM implementation; it does not change
the operation, weights, accumulation type, output layout, or any state update.
The B=128 path is outside the K3 small-batch TGV guard and should remain
unchanged. Golden replay must verify output and post-step state at all points.
