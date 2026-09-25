# KDA trial 25: agent iterations

## iter_00
### hypothesis.md

No source change was made in the baseline iteration. The first candidate selected for iter_01 is the K3 plain-TP fused-front MoE stream schedule: keep the shared-down GEMM on the side stream, but delay its event wait until immediately before `_add3`. The routed latent norm and up projection are independent of `shared_output`; only the final add consumes both. This preserves exact branch arithmetic and leaves KDA recurrent state untouched.

### analysis.md

Baseline VibeSim prediction

- Prediction built with `simulate?prediction=.:before`: `p_38ccb83a6c4e44298f45b2251bd4cf54`.
- Analysis handle: `prediction=.:k3_kda` (the service resolves the returned run as `k3_kda`).
- Measured plain graph replay: B=128 `378.528 us`, B=32 `243.072 us`, B=1 `126.464 us`.
- Diagnostic profile (split/profile phases): B=128 graph `346.656 us`, 23 launches, kernel sum `417.781 us`; B=32 graph `225.856 us`, 24 launches, kernel sum `289.472 us`; B=1 graph `125.504 us`, 23 launches, kernel sum `175.1 us`.

VibeSim `analyze?level=operator` modeled 782.953 us across the three decode cases. The largest node was `unified.kda.moe.mxfp4_fused_moe` at 412.401 us / 52.67%, followed by `merged_front` at 120.183 us / 15.35% and `kda_recurrent_decode` at 42.496 us / 5.43%.

`optimality?scope=iter` ranked MXFP4 first: R0 `412.401 us`, R5 `9.454 us`, R0/R5 `43.62`; the per-node R6/necessary-share fields were null in this fixed prediction, while the run summary gave R6/R0 `7.68%`. The recurrent node was R0 `42.495 us`, R5 `8.161 us`, R0/R5 `5.21`. `kernels` with `kernel_set=mxfp4_fused_moe` and `kernel_set=kda_recurrent_decode` reported `has_cached_alternative=false` for both.

The actual B=128 kernel table agrees with the MoE ranking: the TRT-LLM MXFP4 GEMM1 and GEMM2 launches took `132.835 us` and `66.240 us`. The source therefore cannot gain from selecting a cached replacement. The selected source-level opportunity is schedule overlap around this node: defer the already-recorded shared-down completion wait until the final add so the independent latent norm/up tail can overlap it. This changes no arithmetic, tensor contents, or recurrent state writes.

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

Defer the plain-TP shared-down side-stream event wait from immediately after routed MXFP4 dispatch to immediately before the final `_add3`, allowing the independent latent norm/up tail to overlap shared-down. The event still precedes the only read of `shared_output`, so arithmetic, output values, and all KDA/MoE state remain unchanged.

Result: correctness and state checks passed exactly at all points, but B=128 graph latency was `379.360 us` versus `378.528 us` baseline, with no diagnostic profile improvement. This candidate is rejected and is removed from the final source.

### analysis.md

Post-edit VibeSim routing

- Re-ran `simulate` after the profile. The service reports its fixed before-state prediction `p_38ccb83a6c4e44298f45b2251bd4cf54`; analysis handle remains `prediction=.:k3_kda`.
- Re-ran operator, run-summary, iteration, optimality, and kernel analyses. They remain unchanged: MXFP4 fused MoE is 412.401 us / 52.67% modeled time with R0/R5 `43.62`; KDA recurrent decode is 42.496 us with R0/R5 `5.21`; both `has_cached_alternative=false`.
- The post-edit diagnostic profile still has 23/24/23 launches and graph times B=128 `346.496 us`, B=32 `225.888 us`, B=1 `125.472 us`. The actual top-node work did not move.

The deferred event wait is therefore not a measurable improvement for this graph. The fixed VibeSim prediction also provides no post-edit delta to justify keeping it.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "ok": true, "latency_us": 379.35999035835266, "latency_mode": "graph", "us_step": 1142.1760320663452, "us_step_graph": 379.35999035835266, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 32, "seq_len": 8192, "ok": true, "latency_us": 243.13600361347198, "latency_mode": "graph", "us_step": 1097.1519947052002, "us_step_graph": 243.13600361347198, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [32, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 8192, "ok": true, "latency_us": 126.43200159072876, "latency_mode": "graph", "us_step": 1110.5279922485352, "us_step_graph": 126.43200159072876, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [1, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
```

(diff.patch: 39 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_02
### hypothesis.md

Change `kChunkV` from 32 to 64 in the fused KDA decode kernel and cap TMA stages at the new `kNumChunks`. This reduces the BF16 recurrent-state loop from four chunks/barrier waits to two while keeping the same 128 rows, per-row dot-product order, BF16 rounding points, and state write addresses. The host dispatch remains valid for BF16 and FP32 layouts, and all non-KDA/MoE paths are untouched.

Result: `CHECK pass:true` with zero output/state error at B=128, 32, and 1, but the scored B=128 graph latency increased from `378.528 us` to `390.688 us` and B=32 increased from `243.072 us` to `245.216 us`. The change is removed from the final source.

### analysis.md

Baseline VibeSim routing for the KDA-kernel experiment

- Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`, built by `simulate?prediction=.:before`; analysis handle `prediction=.:k3_kda`.
- `analyze?level=operator`: MXFP4 fused MoE `412.401 us` / `52.67%`; KDA recurrent decode `42.496 us` / `5.43%`.
- `optimality?scope=iter`: KDA recurrent R0 `42.495 us`, R5 `8.161 us`, R0/R5 `5.21`; per-node R6/necessary-share is null in this fixed prediction, and run-summary R6/R0 is `7.68%`.
- `kernels?kernel_set=kda_recurrent_decode`: backend `sglang_triton`, no cached alternative. The actual profile launches the fused `kda_decode_fusion_many_heads_kernel` once per step, at `27.181 us` for B=128.

The MXFP4 node remains the largest modeled node but has no source-selectable cached alternative. The fused KDA kernel is the largest directly editable specialized kernel. Its BF16 state path uses 4 TMA chunks of 32 value rows with 4 barriers and 32 KiB shared staging; testing 2 chunks of 64 rows retains the same row-wise arithmetic and state stores while reducing synchronization and staging-loop overhead.

Post-edit VibeSim was re-run after profiling and returned the same fixed prediction and ladder. The actual profile showed the fused kernel change was not beneficial: graph diagnostics were B=128 `357.920 us`, B=32 `229.824 us`, B=1 `125.472 us`, with kernel sums `431.9/292.3/174.8 us`. The replay was exact at every point, but plain graph timing regressed to B=128 `390.688 us` and B=32 `245.216 us`; the candidate is rejected.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "ok": true, "latency_us": 390.6880021095276, "latency_mode": "graph", "us_step": 1268.3520317077637, "us_step_graph": 390.6880021095276, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 32, "seq_len": 8192, "ok": true, "latency_us": 245.2159970998764, "latency_mode": "graph", "us_step": 1172.3840236663818, "us_step_graph": 245.2159970998764, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [32, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 8192, "ok": true, "latency_us": 126.43200159072876, "latency_mode": "graph", "us_step": 1262.2400522232056, "us_step_graph": 126.43200159072876, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [1, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
```

(diff.patch: 33 lines, files: python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh)

## iter_03
### hypothesis.md

Extend `_k3_bf16_gemm`'s fixed-shape TGV fast path to the K3 merged front weight `(15984, 7168)` when the requested output is FP32. This changes only the dense GEMM implementation; the output dtype, tensor shape, router logits, expert routing API, recurrent state, and all later arithmetic stay the same. The golden replay will verify the resulting accumulated values remain within the required tolerance.

### analysis.md

Baseline VibeSim routing for the merged-front GEMM experiment

- Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`, from `simulate?prediction=.:before`; analysis handle `prediction=.:k3_kda`.
- `analyze?level=operator`: `unified.kda.moe.merged_front` is `120.183 us` / `15.35%` across the three cases, behind MXFP4 fused MoE at `412.401 us` / `52.67%`.
- `optimality?scope=iter`: merged-front R0 `120.183 us`, R5 `91.108 us`, R0/R5 `1.32`; per-node R6/necessary-share is null in the fixed prediction. `kernels?kernel_set=single_gemm` needs an exact leaf selector; the real B=128 profile shows the front as the `51.414 us` NVJet GEMM, and the code path is `_k3_bf16_gemm` in `srt/models/kimi_k3.py`.

The front emits FP32 because its router-logit slice must retain the routing contract. CUTEDSL TGV accepts FP32 output, so the experiment dispatches only the fixed K3 `(15984, 7168)` front shape through TGV while preserving FP32 output allocation.

## iter_04
### hypothesis.md

Select TGV tactic 15 for the fixed K3 merged-front shape `(n, k)=(15984, 7168)`. The kernel implements the same BF16-input, FP32-accumulate GEMM and writes the same FP32 output; only CTA tiling changes. The attention projection and all other GEMM shapes retain their existing tactic selection.

Result: output and state checks passed exactly at all three points, but graph latency regressed at all three points, especially B=1. The selector change is removed.

### analysis.md

Baseline VibeSim routing for the front-tactic experiment

- Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`, analysis handle `prediction=.:k3_kda`.
- The post-iter_03 profile still maps the merged front to the editable TGV path; B=128 front launch was `50.982 us` with the selected `cta64x128x128_2cta` kernel.
- VibeSim remains fixed at MXFP4 fused MoE `412.401 us` / `52.67%`, merged front `120.183 us` / `15.35%`, with merged-front R0/R5 `1.32` and no cached alternative reported for the dominant MXFP4 node.

The TGV extension has a supported single-CTA `cta128x128x128` configuration (tactic 15). For the fixed `(15984, 7168)` front, it reduces the B=128 grid from 250 two-CTA blocks to 125 one-CTA blocks; this experiment changes only dispatch geometry and keeps FP32 output.

Post-edit VibeSim was re-run and returned the same fixed prediction/ladders. The diagnostic graph timings were B=128 `347.648 us`, B=32 `227.808 us`, B=1 `160.224 us`; the replay was exact, but plain graph timings regressed to `380.416/245.248/151.040 us`. The tactic is rejected.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "ok": true, "latency_us": 380.41600584983826, "latency_mode": "graph", "us_step": 1331.7760229110718, "us_step_graph": 380.41600584983826, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 32, "seq_len": 8192, "ok": true, "latency_us": 245.2480047941208, "latency_mode": "graph", "us_step": 1353.8559675216675, "us_step_graph": 245.2480047941208, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [32, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 8192, "ok": true, "latency_us": 151.0400027036667, "latency_mode": "graph", "us_step": 1335.0080251693726, "us_step_graph": 151.0400027036667, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [1, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
```

(diff.patch: 13 lines, files: python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py)

## iter_05
### hypothesis.md

Set the K3 MXFP4 MoE method's `flashinfer_mxfp4_moe_precision` to `bf16` before weight processing. This removes the default activation quantization/preparation kernel and uses the runner's BF16 activation path; weights, expert IDs/weights, SiTU constants, output dtype, and KDA state are unchanged. The golden replay must verify whether the different activation representation remains within tolerance.

Result: rejected. The installed FlashInfer/TRT-LLM build does not provide the required BF16 activation kernel, so the mandatory smoke failed. The source was restored and the restored smoke passed.

### analysis.md

Baseline VibeSim routing for the BF16 MXFP4-activation experiment

- Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`, analysis handle `prediction=.:k3_kda`.
- `analyze?level=operator`: `unified.kda.moe.mxfp4_fused_moe` is `412.401 us` / `52.67%`; merged front is `120.183 us` / `15.35%`.
- `optimality?scope=iter`: MXFP4 R0 `412.401 us`, R5 `9.454 us`, R0/R5 `43.62`, with null per-node R6/necessary-share and run-summary necessary ratio `7.68%`; `kernels?kernel_set=mxfp4_fused_moe` reports no cached alternative.

The source path maps this node to `Mxfp4MoEMethod` and the SM100 TRT-LLM runner. The runner supports a BF16 activation mode in addition to the default MXFP8 activation-quantized mode. This experiment selects BF16 activations only for this K3 SiTU layer while retaining the MXFP4 weights and routing contract.

The mandatory smoke failed before timing: the installed TRT-LLM runner has no kernel for MXFP4 weights with BF16 activations (`MxE2m1 x Bfloat16 -> Bfloat16`). The change was removed immediately; the restored tree smoke passed. No valid post-edit profile or VibeSim delta exists for this candidate.

### result.json

```
{
  "candidate": "bf16 MXFP4 activation mode",
  "smoke": "failed",
  "error": "No kernel found for the given options: mDtypeA: MxE2m1, mDtypeB: Bfloat16, mDtypeC: Bfloat16, mUseDeepSeekFp8: 0",
  "restored_tree_smoke": "pass"
}
```

(diff.patch: 16 lines, files: /workspace/opt_run/iter_05/candidate/kimi_k3.py)

## iter_06
### hypothesis.md

For the K3 SM100 routed MXFP4 path at B=128, pass `tune_max_num_tokens=256` instead of 128 to FlashInfer's TRT-LLM runner. This affects only tactic selection/workspace sizing; routing inputs, MXFP4 weights/scales, activation type, output dtype, and state are unchanged. B=32 and B=1 retain their existing tuning ceilings.

Result: exact output/state checks at all points, no B=128 latency gain. The tuning override is removed.

### analysis.md

Baseline VibeSim routing for the MXFP4 tuning-ceiling experiment

- Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`, analysis handle `prediction=.:k3_kda`.
- The modeled bottleneck remains `unified.kda.moe.mxfp4_fused_moe` at `412.401 us` / `52.67%`, R0/R5 `43.62`, with null per-node R6/necessary-share and no cached alternative.
- The real B=128 profile has the TRT-LLM MXFP4 GEMM1/GEMM2 launches at about `130.3 us` and `66.4 us`. Their source call uses `tune_max_num_tokens=next_power_of_2(x_quant.shape[0])`; this experiment raises only the K3 B=128 ceiling to 256 to test a different FlashInfer tactic.

Post-edit VibeSim returned the same fixed prediction and ladder. The diagnostic graph profile was B=128 `345.632 us`, B=32 `223.872 us`, B=1 `125.408 us`; the replay checks were exact, but plain graph timing was unchanged at B=128 (`378.400 us`) and the candidate provided no reproducible scored gain. It is rejected.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "ok": true, "latency_us": 378.3999979496002, "latency_mode": "graph", "us_step": 1320.5440044403076, "us_step_graph": 378.3999979496002, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 32, "seq_len": 8192, "ok": true, "latency_us": 241.15200340747833, "latency_mode": "graph", "us_step": 1304.1599988937378, "us_step_graph": 241.15200340747833, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [32, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 8192, "ok": true, "latency_us": 124.41600114107132, "latency_mode": "graph", "us_step": 1454.1120529174805, "us_step_graph": 124.41600114107132, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [1, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
```

(diff.patch: 19 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_07
### hypothesis.md

Hypothesis: force TGV tactic 15 for the K3 merged-front shape `(N,K)=(15984,7168)` when `M>=64`, based on a direct B200 microbenchmark. This was expected to reduce the B=128 front GEMM while preserving FP32 output and all arithmetic. The layer replay disproved the latency assumption: exact checks passed, but B=128 regressed, so the edit is not retained.

### analysis.md

VibeSim prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`; analysis handle: `prediction=.:k3_kda`.

The required post-edit profile was routed through `simulate`, `analyze`, and `optimality` before interpretation. The service is a fixed before-state model, so its operator results remain: `unified.kda.moe.mxfp4_fused_moe` is 412.401 us / 52.67% with R0/R5=43.622 and no cached alternative; `unified.kda.moe.merged_front` is 120.183 us / 15.35% with R0/R5=1.319. The post-edit profile showed the forced tactic-15 front kernel at about 50.98 us for B=128, while the graph rose to 347.712 us in the diagnostic run.

The exact replay preserved output and state at all points, but B=128 latency changed from the retained iter_03 result 378.368 us to 380.416 us. B=32 was 241.088 us and B=1 was 124.384 us. Since the scored point regressed, this hypothesis is rejected and the source is restored.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "ok": true, "latency_us": 380.41600584983826, "latency_mode": "graph", "us_step": 1206.7519426345825, "us_step_graph": 380.41600584983826, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 32, "seq_len": 8192, "ok": true, "latency_us": 241.08800292015076, "latency_mode": "graph", "us_step": 1199.3600130081177, "us_step_graph": 241.08800292015076, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [32, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 8192, "ok": true, "latency_us": 124.38400089740753, "latency_mode": "graph", "us_step": 1202.623963356018, "us_step_graph": 124.38400089740753, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [1, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
```

(diff.patch: 10 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py)

## iter_08
### hypothesis.md

Hypothesis: use 3 TMA stages for the BF16 KDA recurrent kernel when `B*H>=512`, matching the existing FP32 occupancy heuristic. This reduces dynamic shared memory from 64 KB to 48 KB and was expected to improve the B=128 KDA leaf without changing arithmetic, state addresses, or synchronization semantics. The full replay showed no scored latency improvement, so the candidate is not retained.

### analysis.md

VibeSim prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54`; analysis handle: `prediction=.:k3_kda`.

The post-edit profile was routed through `simulate`, `analyze`, and `optimality`. The fixed service reports the same roofline ranking: `unified.kda.moe.mxfp4_fused_moe` is 412.401 us / 52.67% with R0/R5=43.622 and no cached alternative; `unified.kda.attention.kda_recurrent_decode` is 42.496 us / 5.43% with R0/R5=5.207 and no cached alternative. The candidate profile did launch the 3-stage BF16 KDA template, but its diagnostic graph was 345.664 us at B=128 and the plain replay was unchanged at 378.368 us.

All three replay points had max_abs_err=0, max_rel_err=0, state_ok=true, and pass=true. Because the scored B=128 latency did not drop, the stage change is rejected and the BF16 selection is restored to 4 stages.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "ok": true, "latency_us": 378.36799025535583, "latency_mode": "graph", "point": [128, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 32, "seq_len": 8192, "ok": true, "latency_us": 241.11999571323395, "latency_mode": "graph", "point": [32, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 8192, "ok": true, "latency_us": 124.41600114107132, "latency_mode": "graph", "point": [1, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true}}
```

(diff.patch: 9 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh)
