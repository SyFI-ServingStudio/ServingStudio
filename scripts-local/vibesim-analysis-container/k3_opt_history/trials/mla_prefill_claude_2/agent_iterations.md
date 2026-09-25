# MLA_PREFILL_CLAUDE trial 2: agent iterations

## agent log.md

- iter_00: baseline. plain: pf49152 11541, pf 8290, 4x4096 7880 us. VibeSim top: merged_front GEMM (closed cuBLAS), attention subtree R0/R5 6.4. Golden captured; control replay bit-exact.
- iter_01: prefix (non-causal) ragged FMHA -> sglang-JIT cute-dsl with full softmax correction (fp8 P prescale 2^8, trtllm-gen accuracy). plain pf49152 11512 -> 11266 us (-2.1%); pf/4x4096 unchanged kernels. CHECK pass all (pf49152 mean 4e-4; others bit-exact). Rejected: cute-dsl on causal (CHECK fail 0.021), prebuilt artifact (2x less accurate).

## iter_00
### analysis.md

# iter_00 — baseline (tree as seeded: previous rounds' accepted levers incl. mla_prefill_claude_1)
Prediction: `.:k3_mla_prefill` (prediction_id p_9318587543334f74b338aa9dcb102cad; fixed before-state; analyze/optimality
resolve only via the default run, the p_ id returns "0 matches"). Sums the 3 cases (47.0 ms kernel time).

run_summary: optimality_ratio 0.085; hardware_gap 81.8%, imbalance 9.2%, batching 0.5%.
optimality(scope=iter), R6/necessary_share = null for every node (not populated for this run) -> ranked by R0-R5 headroom:
| node | R0 ms | R5 ms | R0/R5 |
|---|---|---|---|
| moe.merged_front_prefill (fp32-out GEMM) | 16.8 | 5.0 | 3.36 |
| attention + output gate (subtree) | 7.3 | 1.14 | 6.40 |
| moe.mxfp4_fused_moe_prefill | 8.6 | 2.9 | 2.97 |
| moe.shared_down_prefill | 6.4 | 1.9 | 3.35 |
| moe.latent_up_prefill | 3.5 | 1.1 | 3.15 |
| attention.mla_prefill_attention_prefix | 3.53 | 1.37 | 2.57 |
| attention.mla_prefill_attention_causal | 1.54 | 0.52 | 2.98 |
| attention.o_proj / output_gate_prefill | 1.48 / 1.66 | 0.48 / 0.48 | 3.1 / 3.5 |
| attention.attn_res_prefill | 0.97 | 0.97 | 1.00 |

Measured (per-launch trace, pf49152, 11.5 ms eager): prefix FMHA 3333 us (trtllm-gen fmhaSm100a Dense Q256Kv128),
merged front GEMM (nvjet tss, fp32 out) 2322, shared_down 878, MoE bmm1 881 / bmm2 328, causal FMHA 664, latent_up 493,
qkv_a 328, o_proj 227, g-proj 208, attn_res 2x176, situ 228, add3 132.

Decision: the merged front is the largest node, but it is a closed cuBLAS nvjet kernel at ~1.6 PF/s (~73% of bf16 dense peak);
the prior prefill round microbenchmarked it (bf16-out = numerics FAIL per mla_b512_2, split = slower) -> no code-level lever.
The prefix + causal FMHA leaves (5.07 ms summed R0 vs 1.89 R5, together the largest *replaceable* node: the attention subtree
has the highest R0/R5 = 6.4) have an in-framework alternative kernel: flashinfer's `backend="cute-dsl"` ragged prefill.
Microbench (bench/fmha.py, exact shapes, fp8 QKV): prefix 3471 -> 2861 us, causal 16k 720 -> 545, causal 4x4k 251 -> 208.
Target for iter_01: the ragged prefill kernel (prefix + causal).

## iter_01
### hypothesis.md

# iter_01 hypothesis
Node: `attention.mla_prefill_attention_prefix` (+ causal considered). VibeSim R0/R5 prefix 3.53/1.37 ms (2.57x), causal 1.54/0.52.
Prior trials: mla_prefill_claude_1 (12128 -> 11573 us; fp8 K/V pack + attn_res residual fold) treated the trtllm-gen FMHA as
MUFU-bound and never tried another kernel; mla_b512_2 showed the merged-front GEMM has no numerics-safe lever (bf16 out fails),
so the FMHA is the largest replaceable node.
Hypothesis: flashinfer's cute-dsl Blackwell FMHA is faster on these shapes (microbench prefix 3471 -> 2861 us, causal 720 -> 545).
Trial A (prebuilt artifact, prefix + causal): pf49152 10756 us but CHECK fails on pf / 4x4096 (mean 0.021 / 0.023).
Root cause 1: artifact uses enable_skip_correction -> rescale_threshold 8 -> no fp8 P prescale -> 2x row error vs fp32 (0.021 vs 0.011).
Trial B (own JIT, enable_skip_correction=False -> P prescale 2^8; accuracy vs fp32 = trtllm-gen): still fails causal-only points
(0.0212 / 0.0234): any non-trtllm-gen causal kernel shifts the layer output ~2% (two independent ~1% fp8 roundings, amplified by MoE).
Trial C (adopted): accurate JIT kernel on the non-causal prefix pass only; causal pass stays trtllm-gen (bit-exact on no-prefix points).
Expected: prefix 3333 -> ~3100 us kernel, ~-250 us step on pf49152, 0 change elsewhere.
Rejected: prebuilt artifact on prefix only (10888 us, CHECK bit-exact here only because this driver's prefix V is all-zero so only
LSE is consumed; on real data its O is 2x less accurate than the original -> would trade accuracy away).

### analysis.md

# iter_01 analysis
VibeSim: same static prediction as iter_00 (`.:k3_mla_prefill`, prediction_id p_9318587543334f74b338aa9dcb102cad; simulate /
analyze / optimality / kernels re-fetched into vibesim/, byte-identical to iter_00 - VibeSim models the design, not our profile).
Node `attention.mla_prefill_attention_prefix`: ladder R0 3.53 ms / R5 1.37 ms (R0/R5 2.57), share of iteration R0 ~7.5%
(3.53 / 47.0 ms summed); R6 / necessary_share null for this run.
Measured (trace_pf49152.txt): prefix FMHA 3333 -> 3106 us (trtllm-gen fmhaSm100a Q256Kv128 -> cute-dsl
BlackwellFusedMultiHeadAttentionForward, full correction, P prescale 2^8). Causal FMHA unchanged (678 us, trtllm-gen).
Plain A/B (3 runs each, identical flags): pf49152 11512 +-35 -> 11266 +-30 us (-2.1%, ~7 sigma); pf 8284 -> 8372, 4x4096 7905 -> 7836
(identical kernels there; noise). Replay: pf49152 CHECK pass mean 0.0004 p99 0.0066 rows_over 0; pf / 4x4096 bit-exact.
Bottleneck after: merged-front GEMM (2.3 ms, closed cuBLAS, no lever per mla_b512_2) then prefix FMHA 3.1 ms (now ~2.3x R5).
The remaining FMHA headroom (prebuilt 2.86 ms) is not reachable from the available source: every JIT variant (opt-level 2/3,
sequence barrier, correction double-buffer, thresholds 0-8) is 3.1-3.4 ms.

### result.json

```
{"iter": 1, "node": "attention.mla_prefill_attention_prefix", "files": ["srt/layers/attention/trtllm_mla_backend.py", "kernels/ops/attention/cute_dsl_fmha_fp8.py"],
 "plain_before_us": {"pf49152": [11549.8, 11482.4, 11502.9], "pf": [8226.7, 8293.4, 8332.3], "4x4096": [7841.8, 7885.9, 7988.4]},
 "plain_after_us": {"pf49152": [11288.9, 11277.6, 11230.4], "pf": [8326.2, 8453.2, 8337.5], "4x4096": [7783.2, 7824.4, 7901.3]},
 "replay": {"pf49152": {"latency_us": 11253.0, "pass": true, "mean": 0.000399, "p99": 0.006579, "rows_over": 0},
            "pf": {"latency_us": 8211.6, "pass": true, "mean": 0.0}, "4x4096": {"latency_us": 7715.7, "pass": true, "mean": 0.0}},
 "smoke": "LAYER_SMOKE_OK", "accepted": true}
```

(diff.patch: 194 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/cute_dsl_fmha_fp8.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py)
