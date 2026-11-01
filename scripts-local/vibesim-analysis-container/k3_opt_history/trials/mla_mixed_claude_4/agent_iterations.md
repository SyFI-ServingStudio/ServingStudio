# MLA_MIXED_CLAUDE trial 4: agent iterations

## agent log.md

# log (latency_us: 64,8192,mx16384 | 128,8192,mx4096 | 64,8192,mx16384p49152)
- iter_00: baseline 8560 | 2661 | 13334 (golden capture run)
- iter_01: causal pass restricted to prefill rows of a mixed batch (decode rows go to absorbed decode anyway): 8355 | 2521 | 13163, CHECK pass, max_rel_err 0.0 all points

## iter_00
### analysis.md

# iter_00 — baseline
VibeSim prediction: `.:k3_mla_mixed` (prediction_id p_ae432dbc924742f2bb261e1a0f3f1832; `simulate?prediction=.:before` resolves to it).
optimality (scope=iter), largest leaf headroom (R0 - R5, ms): merged_front_prefill 6.81/3.75 (1.81x), mla_prefill_attention_prefix 3.50/1.37 (2.55x),
mxfp4_fused_moe_prefill 4.24/2.16 (1.96x), shared_down 2.52/1.44, mla_prefill_attention_causal 1.41/0.47 (2.98x), mla_cache_append 0.51/0.008 (64x).
necessary_share (R6) is null for every node in this prediction (no R6 segmentation available).
Measured (eager, --split --profile-kernels) 8489 / 2636 / 13347 us; plain golden run 8560 / 2661 / 13334 us.
Primary point kernel table: front fp32 GEMM 2364, other nvjet GEMMs 2083+257+42, MoE bmm1 947 / bmm2 340, causal trtllm-gen FMHA 849,
attn_res 2x198, situ 242, add3 135; decode rows already go through the absorbed-decode kernel (69 us).
nsys timeline: causal FMHA 976us runs over all 16448 rows (65 sequences) although the 64 decode rows' causal output is discarded
(decode split); host-bound gaps 84us (step start) + 116us (before causal FMHA).
Picked: mla_prefill_attention_causal (2.98x R0/R5, second-largest attention leaf, on the critical path, and it does unneeded work
for the decode rows). The front GEMM is at ~1.5 PFLOP/s (cuBLAS nvjet), MoE is a closed cubin.

## iter_01
### hypothesis.md

# iter_01 hypothesis
Run the causal (in-chunk) pass only over the leading prefill requests when forward_normal_chunked_kv_core splits off the trailing
decode rows (new ForwardBatch.causal_num_reqs; trtllm backend slices cum_seq_lens / seq_lens / seq_lens_cpu / max_q_len).
Same kernel, same per-sequence inputs for the prefill rows -> bit-identical output; decode rows keep their absorbed-decode output.
KV writes happen earlier (set_mla_kv_buffer), untouched -> post-step state identical.
Builds on: mixed decode split already in the tree (seed); prior trials mla_prefill_claude_2 ruled out swapping the causal kernel
(cute-dsl causal failed CHECK at 0.021) — so this keeps trtllm-gen and only removes work.

### analysis.md

# iter_01
Prediction: `.:k3_mla_mixed` (static before-state; the service does not rebuild predictions for edited trees).
Target node: unified.mla.attention.mla_prefill_attention_causal, R0/R5 = 2.98x. The trtllm-gen causal pass received all 65
sequences (16384-token chunk + 64 1-token decode rows); the decode rows are recomputed by absorbed MLA decode and their causal
outputs were dropped by the concat. Removing them from the causal launch is removal of unnecessary work (R6 < R0).

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16448, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 64, "seq_len": 8192, "mixed": false, "tag": "mx16384", "prefill": true, "kind": "mixed", "prefix_len": 0, "chunk": 16384, "draft_tokens": null, "decode_requests": 64, "num_tokens": 16448, "ok": true, "latency_us": 8354.84790802002, "latency_mode": "eager", "us_step": 8354.84790802002, "us_step_graph": null, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": null, "attention_backend": "trtllm_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [64, 8192, "mx16384"], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16448, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 4224, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "mixed": false, "tag": "mx4096", "prefill": true, "kind": "mixed", "prefix_len": 0, "chunk": 4096, "draft_tokens": null, "decode_requests": 128, "num_tokens": 4224, "ok": true, "latency_us": 2521.28005027771, "latency_mode": "eager", "us_step": 2521.28005027771, "us_step_graph": null, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": null, "attention_backend": "trtllm_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192, "mx4096"], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 4224, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16448, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "re
[... truncated]
```

(diff.patch: 126 lines, files: /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py, /sgl-workspace/sglang/python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py, /sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py)

## iter_02
(diff.patch: 134 lines, files: iter_02/kimi_k3.py)
