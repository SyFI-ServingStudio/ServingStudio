# KDA trial 5: agent iterations

## agent log.md

iter_00: baseline graph latency 558.6 -> 558.6 us (B128), 312.7 -> 312.7 us (B32), 181.7 -> 181.7 us (B1); VibeSim selected mxfp4_fused_moe; pending overlap experiment.
iter_01: side-stream shared/routed overlap; replay 558.6 -> 550.3 us (B128), 312.7 -> 309.7 us (B32), 181.7 -> 183.8 us (B1); all CHECK pass, insufficient and regresses B1.
iter_02: gated overlap for B>=32; replay 558.6 -> 550.4 us (B128), 312.7 -> 309.7 us (B32), 181.7 -> 181.7 us (B1); all CHECK pass, still insufficient.
iter_03: BF16 MXFP4 activation experiment rejected by required smoke (no SM100 MxE2m1 x Bfloat16 SiTU kernel); reverted, rollback smoke passed.
iter_04: disabled TRT-LLM PDL for B>=32; replay 558.6 -> 556.7 us (B128), 312.7 -> 316.9 us (B32), 181.7 -> 185.8 us (B1); all CHECK pass, rejected.
iter_05: native JIT fused gate for B>=64/local112; replay 558.6 -> 552.4 us (B128), 312.7 -> 308.7 us (B32), 181.7 -> 181.7 us (B1); checks pass, but slower than iter_02 and rejected.

## iter_00
### hypothesis.md

# Iteration 00 hypothesis

Use the Kimi K3 MoE side stream for the plain TP1 fused-front path. Launch `_forward_shared(gate_up, shared_output)` on `self.alt_stream` after the current stream's front GEMM completes, run `_forward_routed(..., latent)` on the current stream concurrently, then wait for the side stream before latent normalization and the final add.

The shared and routed branches write disjoint slices of `buf`, read immutable inputs/weights, and have no ordering dependency until the tail. The arithmetic and state writes therefore remain unchanged; only stream scheduling changes. The existing side-stream pattern and join are already used by the validated K3 overlap paths, and the driver constructs the MoE side stream during layer creation.

### analysis.md

# Iteration 00 baseline

VibeSim was queried in the required order:

- `workspace-info` first: run `k3_kda`, recipe `kimi_k3_sglang`, GPU `NVIDIA B200`; the reading guide says R6 near R5 is necessary work with batching/launch headroom, R6 near zero is unnecessary work, and a large R0/R5 with nonzero R6 is an inefficient implementation of necessary work.
- `simulate?prediction=.:before`: prediction id `p_38ccb83a6c4e44298f45b2251bd4cf54`.
- `analyze` levels `operator`, `run_summary`, and `iteration` using `prediction_id=p_38ccb83a6c4e44298f45b2251bd4cf54`.
- `optimality?prediction_id=p_38ccb83a6c4e44298f45b2251bd4cf54&scope=iter`.
- `kernels?prediction_id=p_38ccb83a6c4e44298f45b2251bd4cf54&kernel_set=unified.kda.moe.mxfp4_fused_moe`.

The operator breakdown modeled 1.0061 ms of kernel work. The top node was `unified.kda.moe.mxfp4_fused_moe` at 0.542878 ms / 53.96%, backend `sglang_trtllm_mxfp4`. Its optimality ladder was R0 0.542878 ms, R5 0.0098356 ms, R6 0.231980 ms, R7 0.231980 ms, R0/R5 55.19, and `necessary_share` 0.4273. The node had no cached alternative (`has_cached_alternative=false`), so the viable source-level opportunity is dataflow overlap around the routed expert kernel.

The profiler confirms the node mapping: the B=128 graph launches include MXFP4 expert GEMMs at 293.2092 us and 147.6031 us. The independent shared-expert GEMM is about 51.36 us and is currently serialized on the main stream in the TP1 driver path. The full per-kernel tables are in the three `profile_B*_L8192.json` files.

## iter_01
### hypothesis.md

# Iteration 01 hypothesis

The first overlap is useful only at larger batches and its stream synchronization costs more than it saves at B=1. Keep the same arithmetic-preserving schedule but enable it only when `num_tokens >= 32` (the scored B=128 case and the required B=32 case); leave B=1 on the original sequential path to remove its regression. Then pursue the remaining MXFP4 fused-MoE cost through its activation/runner path.

### analysis.md

# Iteration 01 overlap result

After the source edit I called `simulate?prediction=.:after` (the service returns the fixed before-state prediction id `p_38ccb83a6c4e44298f45b2251bd4cf54`) and reran `analyze` at `operator`, `run_summary`, and `iteration`, `optimality?prediction_id=...&scope=iter`, and the `kernels` drill-down for `unified.kda.moe.mxfp4_fused_moe`.

The VibeSim result is unchanged by design: the top operator remains `unified.kda.moe.mxfp4_fused_moe` at 0.542878 ms / 53.96%. Its ladder remains R0 0.542878 ms, R5 0.0098356 ms, R6/R7 0.231980 ms, R0/R5 55.19, and necessary share 0.4273. The kernel drill-down still reports backend `sglang_trtllm_mxfp4` and `has_cached_alternative=false`.

The post-edit diagnostic profile still launches the same 26/27/25 kernels, so the measured change is the intended side-stream schedule. The unperturbed replay measured 550.336 us / 309.728 us / 183.776 us for B=128/32/1 versus 558.560 / 312.704 / 181.696 us at baseline. B=128 improved only 1.47%, and B=1 regressed 1.15%, so this edit is not sufficient as the final result. All replay checks passed with max relative error 0 and `state_ok=true`.

### result.json

```
{
  "replay": [
    {"B": 128, "seq_len": 8192, "latency_us": 550.3360033035278, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}},
    {"B": 32, "seq_len": 8192, "latency_us": 309.7279965877533, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}},
    {"B": 1, "seq_len": 8192, "latency_us": 183.77600610256195, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}}
  ]
}
```

(diff.patch: 24 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_02
### hypothesis.md

# Iteration 02 hypothesis

The measured top node is the SM100 TRT-LLM MXFP4 runner. Its `default` mode first creates MXFP8 activations and packed routing, while the same runner exposes a supported `bf16` activation mode. Selecting `bf16` for the K3 MXFP4 expert method should remove the activation quantization/packing launch and may choose a faster B200 GEMM schedule. The expected numerical difference is only the removal of FP8 activation rounding; it will be checked against the captured output and recurrent state at every required batch.

### analysis.md

# Iteration 02 gated-overlap result

After the edit, `simulate?prediction=.:after` returned prediction id `p_38ccb83a6c4e44298f45b2251bd4cf54`; I reran operator analysis, run-summary analysis, iteration analysis, optimality, and the `unified.kda.moe.mxfp4_fused_moe` kernel drill-down.

The fixed VibeSim model still selects `unified.kda.moe.mxfp4_fused_moe`: R0 0.542878 ms, R5 0.0098356 ms, R6/R7 0.231980 ms, R0/R5 55.19, necessary share 0.4273, and no cached alternative. The diagnostic profiler still shows the same routed MXFP4 kernels and launch count; only the B=1 scheduling branch is now serialized.

Plain replay was 550.368 us / 309.728 us / 181.696 us for B=128/32/1. Checks passed exactly, but the scored point remains only 1.47% below baseline, so more headroom is required.

### result.json

```
{
  "replay": [
    {"B": 128, "seq_len": 8192, "latency_us": 550.3680109977722, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}},
    {"B": 32, "seq_len": 8192, "latency_us": 309.7279965877533, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}},
    {"B": 1, "seq_len": 8192, "latency_us": 181.69599771499634, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}}
  ]
}
```

(diff.patch: 24 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_03
### hypothesis.md

# Iteration 03 hypothesis

Test BF16 activations to eliminate MXFP8 quantization work. This hypothesis was rejected at build/runtime because the installed SM100 FlashInfer kernel set does not cover the required BF16 SiTU combination.

### analysis.md

# Iteration 03 failed experiment

The VibeSim before-state prediction was already available as `p_38ccb83a6c4e44298f45b2251bd4cf54`; its prior operator/optimality/kernel analysis still identifies `unified.kda.moe.mxfp4_fused_moe` (R0 0.542878 ms, R5 0.0098356 ms, R6 0.231980 ms, necessary share 0.4273, no cached alternative).

The attempted BF16 activation selection was scoped to the measured K3 shape. The required cache-cleared smoke failed inside FlashInfer TRT-LLM with `No kernel found` for `mDtypeA: MxE2m1, mDtypeB: Bfloat16, mActType: 2, mFusedAct: 1`. The edit was removed immediately and the smoke was rerun successfully on the original MXFP8 default path. No timing or correctness result from the invalid branch is used.

### result.json

```
{
  "status": "invalid",
  "smoke": {"pass": false, "error": "No kernel found for MxE2m1 x Bfloat16 SiTU"},
  "rollback_smoke": {"pass": true}
}
```

(diff.patch: 1 lines, files: )

## iter_04
### hypothesis.md

# Iteration 04 hypothesis

VibeSim classifies the dominant MXFP4 node as an inefficient implementation of necessary work, and the profiler shows the two large expert GEMMs are launched with dynamic-batch (`dynB`) tactics. TRT-LLM PDL is enabled for every measured batch by `trtllm_moe_enable_pdl`; for this captured decode graph, PDL may add dependent-launch coordination overhead. Disable PDL only for batches of 32 or more while retaining the default B=1 behavior. PDL changes launch scheduling only, so output and recurrent state arithmetic should be identical.

### analysis.md

# Iteration 04 setup

The candidate remains the VibeSim-selected `unified.kda.moe.mxfp4_fused_moe` node (prediction `p_38ccb83a6c4e44298f45b2251bd4cf54`; R0/R5 55.19; necessary share 0.4273; no cached alternative). This iteration tests the runner's PDL launch policy after the BF16 alternative was rejected by the smoke.

VibeSim was rerun after the profile: operator, run-summary, iteration, optimality, and kernel drill-down all still select the same node and report no cached alternative. Plain replay was 556.672 us / 316.864 us / 185.792 us at B=128/32/1, all with exact output/state checks, so disabling PDL is rejected and will be reverted.

### result.json

```
{
  "replay": [
    {"B": 128, "seq_len": 8192, "latency_us": 556.6719770431519, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}},
    {"B": 32, "seq_len": 8192, "latency_us": 316.864013671875, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}},
    {"B": 1, "seq_len": 8192, "latency_us": 185.7919991016388, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}}
  ]
}
```

(diff.patch: 15 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_05
### hypothesis.md

# Iteration 05 hypothesis

The actual local 112-expert router misses the 896-expert radix specialization and launches a Triton top-k kernel. The existing native JIT fused gate supports this local shape; use it only at B>=64 after making the strided fused-front logits contiguous. This should preserve the selected top-k semantics within the required tolerance and reduce routing overhead.

### analysis.md

# Iteration 05 native-router result

The VibeSim prediction was rebuilt/fetched after profiling as `p_38ccb83a6c4e44298f45b2251bd4cf54`; operator, run-summary, iteration, optimality, and kernel drill-down again selected `unified.kda.moe.mxfp4_fused_moe` with R0 0.542878 ms, R5 0.0098356 ms, R6/R7 0.231980 ms, necessary share 0.4273, and no cached alternative.

The profiler confirmed an extra contiguous router-input copy and one additional launch. Plain replay was 552.416 us / 308.704 us / 181.696 us at B=128/32/1. All checks passed; B=128 output max relative error was 0.002778 and state matched, but latency was worse than the gated-overlap result, so this candidate is rejected.

### result.json

```
{
  "replay": [
    {"B": 128, "seq_len": 8192, "latency_us": 552.4160265922546, "check": {"max_abs_err": 0.00048828125, "max_rel_err": 0.00277776197539854, "mean_rel_err": 0.0000032982284210447688, "nan": false, "state_ok": true, "pass": true}},
    {"B": 32, "seq_len": 8192, "latency_us": 308.7039887905121, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}},
    {"B": 1, "seq_len": 8192, "latency_us": 181.69599771499634, "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}}
  ]
}
```

(diff.patch: 28 lines, files: python/sglang/srt/layers/moe/topk.py)
