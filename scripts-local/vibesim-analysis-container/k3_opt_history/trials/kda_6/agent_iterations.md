# KDA trial 6: agent iterations

## agent log.md

iter_00 baseline: VibeSim MoE leaf 53.96% / necessary_share 0.427; plain graph 556.480 -> 310.720 -> 179.584 us for B=128/32/1; golden captured.
iter_01 rejected: BF16-input TRT-LLM SiTU variant has no B200 kernel; rollback smoke passed; no timing/CHECK.
iter_02 rejected: PDL-off replay 564.704 -> 316.864 -> 185.792 us, all CHECKs pass but every point regresses; rollback smoke passed.
iter_03 accepted base: fused-front shared_down overlapped routed MXFP4 BMM; plain graph 556.480 -> 548.288, 310.720 -> 309.024, 179.584 -> 179.488 us; all replay CHECKs pass.

## iter_00
### hypothesis.md

# Hypothesis

The first experiment will test the existing TRT-LLM MXFP4 `bf16` activation
mode for the exact Kimi-K3 SM100 path. It removes the mxfp8 activation
quantization/packing work and may select a faster BF16xMXFP4 expert kernel.
The expert weights, routing ids/weights, SiTU constants, output layout, and
KDA state path remain unchanged. Replay against the captured output and
post-step state is the acceptance test; revert the experiment if the tolerance
or any batch-size guard fails.

### analysis.md

# Iteration 00: baseline

Initial profile was captured with the required diagnostic command. The VibeSim
prediction was built with `prediction=.:before` and resolved to
`p_38ccb83a6c4e44298f45b2251bd4cf54` (`.:k3_kda` for analysis).

VibeSim operator analysis ranked `unified.kda.moe.mxfp4_fused_moe` first at
0.542877 ms / 53.96% of predicted kernel time. Its iteration-2 (B=128) ladder
was R0=0.542878 ms, R5=0.009836 ms, R6=0.231980 ms, with
`necessary_share=0.427316`; this is the largest measured headroom and a
kernel-efficiency candidate. The next nodes were `qkvbfg_a_proj` (14.37%,
necessary_share 0.078250) and `kda_recurrent_decode` (4.22%, 0.374439).

The VibeSim kernel drill-down for the MoE leaf reports backend
`sglang_trtllm_mxfp4`, shape hidden=3584/intermediate=3072/local experts=112,
top-k=16, and `has_cached_alternative=false`.

The profiler confirms that the live B=128 step launches the two large TRT-LLM
MXFP4 BMMs at 292.315 us and 147.303 us, plus the router/quantization and
finalize work. The source mapping is
`kimi_k3.py` fused-front MoE -> `mxfp4.py` ->
`flashinfer_trtllm.py::_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen`.

The exact plain graph reference was B=128 556.480 us, B=32 310.720 us, and
B=1 179.584 us. The diagnostic profile (which includes `--split` and kernel
profiling) reported 628.192, 364.992, and 178.656 us respectively and is not
used as the timing reference.

### result.json

```
{
  "mode": "baseline",
  "plain_graph": {
    "128,8192": 556.4799904823303,
    "32,8192": 310.7199966907501,
    "1,8192": 179.58399653434753
  },
  "checks": "golden captured at /tmp/golden_B{128,32,1}_L8192.pt"
}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# Hypothesis

Use FlashInfer's supported BF16 activation precision to avoid MXFP8 activation
quantization and potentially improve the dominant routed-MoE leaf. The source
guard was limited to SM100 with configured precision `default`; numerical
behavior was intended to be checked by replay.

Outcome: rejected before replay because the installed TRT-LLM SM100 kernel
set has no BF16-input SiTU kernel for this shape.

### analysis.md

# Iteration 01: BF16 expert-input experiment

This iteration used the same initial VibeSim target: `unified.kda.moe.mxfp4_fused_moe`.
No post-edit profile or VibeSim re-analysis was valid because the edited layer
could not launch the selected expert kernel.

The smoke reached `Mxfp4MoEMethod._apply_sm100_trtllm_gen` and FlashInfer
rejected the BF16-input combination at B=1 with:
`No kernel found ... mDtypeA: MxE2m1, mDtypeB: Bfloat16, mDtypeC: Bfloat16,
mActType: 2 ...`.

The source was restored and the required reload smoke then passed on the
original MXFP8 (`default`) path at B=1. This experiment is rejected for the
full workload and has no timing or CHECK result.

### result.json

```
{
  "status": "rejected",
  "smoke": {
    "point": [1, 8192],
    "ok": false,
    "error": "No kernel found for mDtypeA MxE2m1, mDtypeB Bfloat16, mDtypeC Bfloat16, SiTU"
  },
  "rollback_smoke": "pass",
  "timing": null,
  "checks": null
}
```

(diff.patch: 12 lines, files: python/sglang/srt/layers/quantization/mxfp4.py)

## iter_02
### hypothesis.md

# Hypothesis

CUDA graph replay already provides launch ordering, so disabling programmatic
dependent launch on the large pre-routed TRT-LLM MXFP4 expert operation might
remove dependency overhead while preserving all values and state.

Outcome: rejected. PDL is beneficial for all three measured points.

### analysis.md

# Iteration 02: TRT-LLM PDL experiment

The target remained `unified.kda.moe.mxfp4_fused_moe`, selected from the
initial VibeSim ladder and live B=128 BMM table. This edit only changed the
`enable_pdl` argument of the pre-routed FlashInfer TRT-LLM call; no VibeSim
post-edit prediction was built because the full replay rejected the edit on
latency and the source was restored immediately.

PDL-off replay results were B=128 564.704 us, B=32 316.864 us, and B=1
185.792 us. Compared with the plain baseline 556.480/310.720/179.584 us,
PDL-off regressed every point. All three checks passed with max_rel_err=0,
state_ok=true, and no NaNs, confirming the knob is numerically neutral but not
an optimization for this graph.

### result.json

```
{
  "status": "rejected",
  "latency_us": {
    "128,8192": 564.7040009498596,
    "32,8192": 316.864013671875,
    "1,8192": 185.7919991016388
  },
  "baseline_latency_us": {
    "128,8192": 556.4799904823303,
    "32,8192": 310.7199966907501,
    "1,8192": 179.58399653434753
  },
  "checks": {
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "32,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "1,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  }
}
```

(diff.patch: 5 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_03
### hypothesis.md

# Hypothesis

In the TP1/EP-rank-shaped harness, `_forward_fused` unnecessarily serializes
the shared-expert down projection before the routed MXFP4 expert operation.
Launching the shared GEMM on the already-provided side stream should hide most
of its bandwidth cost under the routed BMMs. The final synchronization preserves
the original data dependencies, addition order, output values, and KDA state.

### analysis.md

# Iteration 03: shared-down overlap

The post-edit diagnostic profile was captured in `profile_B*.json` and
`run.json`; `profile.json` is the run summary copy. VibeSim was re-simulated
with `prediction=.:after`, which intentionally resolves to the fixed
before-state prediction `p_38ccb83a6c4e44298f45b2251bd4cf54`. Its operator,
iteration, optimality, and leaf-kernel analyses therefore remain the same:
the MXFP4 leaf is 53.96% with R0=0.542878 ms, R5=0.009836 ms, R6=0.231980
ms, `necessary_share=0.427316`, and no cached alternative.

The source change targets the next most useful executable dependency exposed by
the same analysis: `shared_down` is an independent 50.179 us predicted GEMM.
The fused front already produces `gate_up` and `routed_input` separately. On
the single-rank driver, the shared down GEMM and routed expert operation write
disjoint output regions, so the shared GEMM is issued on the existing MoE side
stream and joined before the unchanged final add. The post-edit profile still
launches 26/27/25 kernels at B=128/32/1; the graph profile is diagnostic and
shows the expected stream overlap.

The plain fixed-flags reference after the edit was B=128 548.288 us, B=32
309.024 us, and B=1 179.488 us. The replay CHECK passed at every point with
`max_rel_err=0`, `state_ok=true`, and no NaNs.

### result.json

```
{
  "status": "accepted_as_base",
  "plain_graph_latency_us": {
    "128,8192": 548.2879877090454,
    "32,8192": 309.02400612831116,
    "1,8192": 179.48800325393677
  },
  "replay_latency_us": {
    "128,8192": 548.3840107917786,
    "32,8192": 309.6959888935089,
    "1,8192": 181.72800540924072
  },
  "checks": {
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "32,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "1,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "profile": {
    "128,8192": {"latency_us": 624.0959763526917, "launches": 26},
    "32,8192": {"latency_us": 360.9920144081116, "launches": 27},
    "1,8192": {"latency_us": 181.66400492191315, "launches": 25}
  }
}
```

(diff.patch: 27 lines, files: python/sglang/srt/models/kimi_k3.py)
