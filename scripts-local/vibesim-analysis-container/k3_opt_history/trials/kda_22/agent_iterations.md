# KDA trial 22: agent iterations

## agent log.md

iter_00 baseline: graph 384.512 -> 249.312 -> 130.528 us (B=128/32/1); VibeSim selected unified.kda.moe.mxfp4_fused_moe as the dominant necessary-work node.
iter_01 rejected routed-topk prepack: checks passed, but B128 improved only 0.28% and B32/B1 regressed; source reverted.
iter_02 accepted BF16 KDA fused decode with BF16x2 state stores: 384.512 -> 378.432, 249.312 -> 243.168, 130.528 -> 126.464 us; all final checks passed.

## iter_00
### analysis.md

# Iteration 00

Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54` (`vibesim:k3_kda`), built with `simulate?prediction=.:before`.

VibeSim operator analysis modeled 782.953 us of leaf kernel time. The largest operator was `unified.kda.moe.mxfp4_fused_moe` at 412.401 us / 52.67%, followed by `merged_front` at 120.183 us / 15.35% and KDA recurrent decode at 42.495 us / 5.43%. The B=128 iteration predicted 417 us total, with the MXFP4 expert leaf at 265 us / 63.5%.

The optimality ladder ranked `unified.kda.moe.mxfp4_fused_moe` first: R0=412.401 us, R5=9.454 us, `r0/r5=43.62`, and headroom=402.947 us. R6/R7 were unavailable for this fixed leaf, so the reading guide classifies it as the largest necessary-work implementation headroom candidate. `merged_front` was R0=120.183 us, R5=91.108 us, ratio 1.32; KDA recurrent decode was R0=42.495 us, R5=8.161 us, ratio 5.21.

The actual B=128 profile has 26 launches and 600.068 us of kernel time. The two dominant MXFP4 expert launches are 240.727 us and 133.751 us. `fused_recurrent_kda_packed_decode_kernel` is already present at 19.114 us. The kernel drill-down for `unified.kda.moe.mxfp4_fused_moe` reports backend `sglang_trtllm_mxfp4`, R0=412.401 us, R5=9.454 us, and `has_cached_alternative=false`.

The next change targets dispatch overhead around the selected expert node, not its arithmetic: FlashInfer's pre-routed API repacks the standard `(topk_ids, topk_weights)` pair into BF16-weight packed IDs internally. The existing SGLang Triton pack helper produces the same packed bit representation in one graph-capturable launch, allowing the runner to pass the packed form directly.

## iter_01
### hypothesis.md

# Hypothesis

In the SM100 MXFP4 routed-MoE runner, pre-pack standard top-k routing with the existing `fused_pack_topk` Triton kernel before calling `trtllm_fp4_block_scale_routed_moe`. FlashInfer's own pre-routed conversion is defined as `(expert_id << 16) | bf16(weight_bits)`, which is exactly the helper's output; expert IDs, BF16 routing weights, activation quantization, expert GEMMs, finalization, recurrent state, and all layer arithmetic remain unchanged. The expected gain is removal of the runner's repeated torch conversion/packing work from the CUDA graph critical path.

### analysis.md

# Iteration 01

Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54` (`vibesim:k3_kda`). I ran `simulate?prediction=.:after` and then the operator, run-summary, iteration, optimality, and kernel analysis verbs. The service reported that this container exposes a fixed before-state prediction, so the prediction id and all VibeSim values remained unchanged.

The analysis still ranked `unified.kda.moe.mxfp4_fused_moe` first (R0=412.401 us, R5=9.454 us, ratio=43.62, headroom=402.947 us, no cached alternative). The routed-MXFP4 kernel drill-down remained the same SM100 shape: hidden 3584, intermediate 3072, 112 local experts, top-k 2, BF16 activations, MXFP4 E2M1/UE8M0 weights, group size 32, SiTU activation.

The diagnostic profile after the experiment showed an added `_pack_topk_kernel` launch and 27/28/26 launches for B128/B32/B1. The exact replay passed all checks, but the plain timing moved 384.512 -> 383.424 us at B128 (0.28%, below the 0.5% threshold), and regressed B32/B1 to 250.688/132.608 us. The experiment was therefore reverted and is not part of the final source.

### result.json

```
CHECK {"max_abs_err": 0.125, "max_rel_err": 0.008403, "mean_rel_err": 0.001226, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "latency_us": 383.4240138530731, "latency_mode": "graph", "pass": true}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 32, "seq_len": 8192, "latency_us": 250.68798661231995, "latency_mode": "graph", "pass": true}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 8192, "latency_us": 132.60799646377563, "latency_mode": "graph", "pass": true}
```

(diff.patch: 16 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_02
### hypothesis.md

# Hypothesis

The measured KDA recurrent portion is split across causal-convolution, recurrent-update, and gated-norm launches, while the checked workload stores recurrent state as BF16. Extend the existing fused KDA decode kernel to accept BF16 state storage while retaining FP32 recurrence registers. Round the fused convolution output and recurrent output to BF16 at the same boundaries as the original chain, then store four BF16 state values through two BF16x2 stores. This preserves the original state dtype, update addresses, arithmetic precision, and output boundaries while removing launch and intermediate-memory traffic.

For aligned BF16 state pools, use the existing four-stage TMA staging path. The two-stage BF16 `cp.async` trial was rejected because it failed B128 state/output checks and was reverted.

### analysis.md

# Iteration 02

Prediction: `p_38ccb83a6c4e44298f45b2251bd4cf54` (`vibesim:k3_kda`). After each new diagnostic profile I ran `simulate?prediction=.:after`, followed by operator, run-summary, iteration, optimality, and kernel analysis. VibeSim explicitly reports that this environment serves a fixed before-state prediction, so the prediction id and modeled values are unchanged by source edits.

VibeSim still identifies `unified.kda.moe.mxfp4_fused_moe` as the primary modeled candidate: R0=412.401 us, R5=9.454 us, ratio=43.62, headroom=402.947 us, and `has_cached_alternative=false`. The run summary is R0=782.952 us, R5/R6=60.122 us, optimality ratio 0.0768, with 61.14% hardware gap and 29.01% batching gap. KDA recurrent decode is modeled at R0=42.496 us and R5=8.161 us, ratio 5.21; this was the measured source-level target because the final profile confirmed the actual KDA chain was replaced by the fused launch.

The final B128 diagnostic profile is `/workspace/opt_run/iter_02/profile_final_B128_B128_L8192.pt`: 23 launches, 417.470 us kernel sum, 345.664 us diagnostic graph latency, and the BF16 fused KDA kernel at 27.146 us. The earlier valid TMA4 profile measured 33.209 us before the BF16x2 store change; the invalid BF16 `cp.async` variant was reverted after B128 `state_ok:false`.

Plain fixed-flag timing, with the initial baseline from `iter_00/before_plain.json`, improved B128 384.512 -> 378.432 us, B32 249.312 -> 243.168 us, and B1 130.528 -> 126.464 us. Three repeated plain runs were stable at approximately 379.1, 243.1, and 126.4 us. The final replay checks passed at every point with `state_ok:true` and no NaNs.

### result.json

```
CHECK {"max_abs_err": 0.1875, "max_rel_err": 0.012605, "mean_rel_err": 0.000584, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "ok": true, "latency_us": 378.4320056438446, "latency_mode": "graph", "us_step": 1334.7200155258179, "us_step_graph": 378.4320056438446, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192], "correctness": {"max_abs_err": 0.1875, "max_rel_err": 0.012605041169408998, "mean_rel_err": 0.0005840336671099067, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.09375, "max_rel_err": 0.007177, "mean_rel_err": 0.000487, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 32, "seq_len": 8192, "ok": true, "latency_us": 243.16799640655518, "latency_mode": "graph", "us_step": 1293.4720516204834, "us_step_graph": 243.16799640655518, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [32, 8192], "correctness": {"max_abs_err": 0.09375, "max_rel_err": 0.007177032943385038, "mean_rel_err": 0.00048651406541466713, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 8192, "ok": true, "latency_us": 126.46399438381195, "latency_mode": "graph", "us_step": 1280.5759906768799, "us_step_graph": 126.46399438381195, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [1, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
```

(diff.patch: 460 lines, files: python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh, python/sglang/kernels/ops/attention/kda_fused_decode.py)
