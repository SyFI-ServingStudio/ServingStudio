# MLA_MIXED_CLAUDE trial 2: agent iterations

## agent log.md

# opt_run log (latency_us per point: mx16384 / mx4096 / mx16384p49152; replay vs golden.pt)
- iter_00: baseline, original tree, golden captured. 15063 / 11514 / 37966 (golden run); VibeSim p_ae432dbc924742f2bb261e1a0f3f1832 top headroom = attention+output gate R0/R5 6.47.
- iter_01: mixed-batch decode rows -> absorbed MLA decode (trtllm-gen), chunked-prefix MHA only for prefill reqs (chunk len unchanged). 8351 / 2562 / 13180, CHECK pass/pass/pass. (Variant with 1 big prefix chunk: 11095 on p49152 but CHECK FAIL, reverted: log2-LSE vs ln merge_state makes output chunk-dependent.)

## iter_00
### hypothesis.md

# iter_00 - baseline (no code change)

Baseline measurement and golden capture only (golden.pt from the ORIGINAL tree, golden.log EXIT 0).

Prior trials read from /workspace/opt_history:
- mla_mixed_claude_1: VOID (no change landed). Nothing to build on, and no lever ruled out.
- k3_mla_prefill / k3_mla_lcprefill trials built the cute-dsl fp8 prefix FMHA, the kv_b_proj
  fp8-pack epilogue GEMM and the causal-on-alt-stream overlap. All are already in this tree,
  and they tune the per-chunk MHA path itself.
- A8 (decode trials): the trtllm-gen MLA decode kernel with is_var_seq=False.

Hypothesis for iter_01: remove the mixed batch's decode rows from the chunked-prefix MHA path and
serve them with absorbed MLA decode (q_absorb -> trtllm-gen decode over the paged fp8 latent
cache -> v_up). That is what the VibeSim leaf list for this node expects. It should remove most
of the 5.17 ms prefix/causal subnode R0.

### analysis.md

# iter_00 - baseline analysis

VibeSim prediction id: `p_ae432dbc924742f2bb261e1a0f3f1832` (run k3_mla_mixed, kimi_k3_sglang, B200).
Verbs used: workspace-info, simulate (prediction=.:before), analyze?level=run_summary|operator,
optimality?scope=iter (raw responses: vibesim_*.json). The `prediction=<id>` query param
returned "0 matches" on analyze/optimality, so the default (this run's before-prediction) was used.

## run_summary
- R0 real 24.2 ms (all three points summed), R5 hardware limit 11.0 ms, necessary ratio 0.45.
- Gap buckets: hardware_gap 5.5 ms, imbalance 4.4 ms, batching 3.3 ms.

## optimality (scope=iter), headroom R0-R5, sorted by headroom
1. `unified.mla.attention.attention + output gate`: R0 6.88 ms, R5 1.06 ms, R0/R5 **6.47**. Largest headroom.
   - Its prefix/causal attention subnode (mla_prefix_gather, kv_b_proj_prefill,
     prefill_attention_prefix/causal, merge_state): R0 5.17 ms, R5 1.99 ms.
   - The leaf list includes `q_absorb`, `mla_decode_attention` and `v_up`. The model
     expects the 1-token decode rows of a mixed batch to be served by absorbed MLA decode,
     not up-projected chunked-prefix MHA.
2. merged_front_prefill GEMM: 6.8 ms, R0/R5 1.81.
3. mxfp4 fused MoE: 4.24 ms, R0/R5 1.96.
4. mla_prefill_attention_prefix: 3.5 ms, R0/R5 2.55.
5. shared_down 2.5 ms. mla_cache_append 0.51 ms (R0/R5 64, but small in absolute terms).

## measured profile (profile.json; --split --profile-kernels is diagnostic only)
| point | eager latency_us | kernel sum us | launches |
|---|---|---|---|
| 64,8192,mx16384 | 15051 | 15252 | 84 |
| 128,8192,mx4096 | 11511 | 11625 | 117 |
| 64,8192,mx16384p49152 | 37902 | 38438 | 249 |

Primary point, top kernels:
- cute-dsl fp8 prefix FMHA: 3409 us over 5 launches
- fixup_zero_kv_rows: 2794 us over 5 launches
- merged-front nvjet: 2289 us
- nvjet 128x256: 1950 us over 5 launches
- MXFP4 MoE: 852 us
- causal fmha: 751 us
- kv_b_proj fp8-pack GEMMs: 582 + 559 us over 5 launches

About 60% of the primary step is the prefix path. It runs because the 64 decode rows (each with
an 8191-token prefix, 524k tokens total) are up-projected by kv_b_proj and attended as chunked MHA.
prefix_chunk_len = 128k // 65 = 2016, which gives 5 chunks. Every chunk also pays the
zero-KV fixup for the prefill row, which has no prefix.

Findings map to source:
- Node: attention + output gate / prefix-causal subnode.
- `srt/models/deepseek_common/attention_forward_methods/forward_mha.py` (forward_normal_chunked_kv_core)
- `srt/model_executor/forward_batch_deepseek_mha_mixin.py` (prepare_chunked_prefix_cache_info)
- `srt/layers/attention/trtllm_mla_backend.py` (forward_extend prefix branch)

### result.json

```
{
 "iter": 0,
 "command": "python3 /tmp/kimi_single_layer_decode.py --point '64,8192,mx16384;128,8192,mx4096;64,8192,mx16384p49152' $(cat ARGS) (golden capture, original tree)",
 "points": {
  "mx16384": {
   "latency_us": 15062.65640258789
  },
  "mx4096": {
   "latency_us": 11514.047622680664
  },
  "mx16384p49152": {
   "latency_us": 37966.33529663086
  }
 },
 "diagnostic_profile_latency_us": {
  "mx16384": 15051.2,
  "mx4096": 11511.1,
  "mx16384p49152": 37901.9
 },
 "check": "golden reference (self), no CHECK"
}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 - hypothesis

Builds on: the iter_00 VibeSim finding (node `attention + output gate`, R0/R5 6.47; the leaf set
includes q_absorb / mla_decode_attention / v_up). It keeps the prior trials' levers:
- lcprefill/prefill trials: cute-dsl fp8 prefix FMHA, the kv_b_proj fp8-pack epilogue, and the
  causal-on-alt-stream overlap. All still serve the prefill request's prefix.
- A8: the trtllm-gen decode kernel. It is reused here through `_run_decode_kernel`
  (is_var_seq=False path).

mla_mixed_claude_1 was VOID, so nothing there is built on or ruled out.

Hypothesis: in a MIXED extend batch, the trailing 1-token requests with a cached prefix are
decode rows. Their attention = softmax(q . [W_UK c ; k_pe]) W_UV c, which absorbed MLA computes
exactly on the latent cache (the same fp8 latent that chunked MHA up-projects).
- Split the batch: chunked-prefix MHA covers only the leading prefill requests.
- The causal pass still covers every row (only the prefill rows are kept).
- The decode rows go through bmm(q_nope, w_kc), trtllm-gen MLA decode over the paged fp8 cache,
  then bmm(., w_vc).

Expected: the prefix-path kernels on mx16384 / mx4096 (≈7.5 ms / 8.9 ms of kernel time) are
replaced by an ~0.1 ms decode.

Ruled out along the way: shrinking the prefill request's prefix chunk count (1 chunk of 49152).
It is faster (11.1 ms), but it changes the output because merge_state reads the kernels' log2 LSE
as natural log. The chunk length stays capacity // batch_size.

### analysis.md

# iter_01 - analysis

VibeSim prediction id: `p_ae432dbc924742f2bb261e1a0f3f1832` (k3_mla_mixed before-prediction). VibeSim takes no after-profile
upload, so the same node tree is re-read (vibesim_*.json) and compared with the measured iter_01
profile (profile.json).

Target node: `unified.mla.attention.attention + output gate` (R0/R5 6.47). Within it:
- prefix/causal subnode R0 5.17 ms -> the prefix path is gone on mx16384 and mx4096.
- The decode rows now map to the leaves `q_absorb` / `mla_decode_attention` / `v_up`.

Measured after the change (diagnostic --split --profile-kernels run):
| point | latency_us | kernel sum us | launches (before -> after) |
|---|---|---|---|
| mx16384 | 8310 | 7924 | 84 -> 37 |
| mx4096 | 2653 | 2710 | 117 -> 36 |
| mx16384p49152 | 13157 | 12865 | 249 -> 236 |

On mx16384, the following kernels are removed:
- prefix FMHA (3409 us)
- fixup_zero_kv_rows (2794 us)
- 5 kv_b_proj pack GEMM pairs (1141 us)
- gathers and merges

The new cost is:
- trtllm-gen MLA decode (fmhaSm100f ... Qk576HV512 PagedKv): 69 us
- two small bmm for q_absorb / v_up
- the fp8 query cast

The primary point is now GEMM/MoE bound:
- merged_front nvjet: 2285 us
- 5x nvjet 128x256: 2038 us
- MXFP4 MoE: 919 + 336 us
- causal fmha: 852 us

These match VibeSim's next nodes: merged_front_prefill R0/R5 1.81, mxfp4 MoE 1.96, shared_down.

## Correctness finding on mx16384p49152
The first version also resized the prefill request's prefix chunks to capacity // num_prefill_reqs
(1 chunk of 49152). It failed the CHECK: 11376 rows over tol, mean 0.035; state_ok true.

Isolation runs (dbg/):
- Kernel-level tests: the cute-dsl fp8 FMHA with and without KV split matches a torch reference
  at kv=49152 (dbg/split_test.py).
- The trtllm-gen and cute-dsl kernels return **log2** LSE (dbg/lse_test.py).
- sglang `merge_state` combines LSE as **natural log**.

So the merged prefix output depends on where the chunk boundaries fall, and the golden has
capacity // batch_size = 2016 built in. With that chunk length restored, the point passes.
The remaining p49152 cost is 25 prefix FMHA launches (3943 us) and 25 merges (769 us).

### result.json

```
{
 "iter": 1,
 "command": "python3 /tmp/kimi_single_layer_decode.py --point '64,8192,mx16384;128,8192,mx4096;64,8192,mx16384p49152' $(cat ARGS) --replay /workspace/opt_run/golden.pt",
 "smoke": "LAYER_SMOKE_OK",
 "points": {
  "mx16384": {
   "latency_us": 8350.848197937012,
   "pass": true
  },
  "mx4096": {
   "latency_us": 2562.1440410614014,
   "pass": true
  },
  "mx16384p49152": {
   "latency_us": 13180.255889892578,
   "pass": true
  }
 }
}
CHECK {"max_abs_err": 0.21875, "max_rel_err": 0.012963, "mean_rel_err": 1.5e-05, "nan": false, "rows": 16448, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "rel_err_max": 0.02, "pass": true}
CHECK {"max_abs_err": 0.171875, "max_rel_err": 0.010036, "mean_rel_err": 0.000127, "nan": false, "rows": 4224, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.004562, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "rel_err_max": 0.02, "pass": true}
CHECK {"max_abs_err": 0.21875, "max_rel_err": 0.01306, "mean_rel_err": 1.5e-05, "nan": false, "rows": 16448, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "rel_err_max": 0.02, "pass": true}
```

(diff.patch: 428 lines, files: /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py, /sgl-workspace/sglang/python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py, /sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py)
