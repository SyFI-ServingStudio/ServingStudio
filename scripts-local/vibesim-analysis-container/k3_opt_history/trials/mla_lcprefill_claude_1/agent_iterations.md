# MLA_LCPREFILL_CLAUDE trial 1: agent iterations

## agent log.md

- iter_00: baseline plain 25099/45709/16819/33161 us. Prefix FMHA = 66% of step on trtllm-gen; the accepted cute-dsl prefix path is disabled by a 16B-misaligned cum_seqlen_k (chunk-1 row of prefix_chunk_cu_seq_lens).
- iter_01: cute-dsl prefix FMHA re-enabled (16B-align cum_seqlen_k of prefix chunk 1): replay 25099->23359 / 45709->42718 / 16819->16030 / 33161->31866 us; CHECK pass all (pf245760 mean 2e-5, others exact).
- iter_02: KV-split prefix FMHA (S by wave model, log2 merge, memcpy Q-rep): replay 23359->21975 / 42718->42955 / 16030->15684 / 31866->32049 us; CHECK pass all (exact). 32k regress vs iter_01.
- iter_03: empirical split policy (S=4 <=6 waves, else S=2): replay 21975->21997 / 42955->42360 / 15684->15740 / 32049->31917 us (vs baseline 25099/45709/16819/33161); CHECK pass all (exact). KEPT.

## iter_00
### hypothesis.md

# iter_00 — baseline only, no change. Golden captured at /workspace/opt_run/golden.pt (seed tree).
Prior levers in the seed: mla_prefill_claude_1 (fp8 K/V pack, attn_res fused add, kv_indices grid), mla_prefill_claude_2
(cute-dsl prefix FMHA, found silently disabled on >128k prefixes here).

### analysis.md

# iter_00 — baseline (seed tree incl. all prior accepted levers)
VibeSim prediction: `.:k3_mla_prefill` (prediction_id p_9318587543334f74b338aa9dcb102cad). Fixed before-state built for the
older 3-case prefill workload (pf49152 / 16k no-prefix / 4x4k); simulate is idempotent. Saved vs_*.json.
run_summary: optimality 0.085, hardware_gap 81.8%. optimality(scope=iter), necessary_share (R6) = null everywhere:
top nodes by R0-R5: merged_front_prefill 16.8/5.0 ms (R0/R5 3.36), attention+output gate subtree 7.3/1.14 (6.40),
mxfp4 MoE 8.6/2.9, shared_down 6.4/1.9, mla_prefill_attention_prefix 3.53/1.37 (2.57).
The prefix-attention leaf scales linearly with prefix length: at 245,760 prefix it is 5x the modeled 49k value, so on this
workload the attention-prefix node is by far the largest (measured below), with R0/R5 ~2.6 -> kernel-efficiency headroom.

Measured (profile_B1_L16384pf245760.json, 25096 us eager):
| kernel | us |
|---|---|
| trtllm-gen fmhaSm100a Dense (prefix, 2 chunks: 131072 + 114688) | 16695 |
| merged front nvjet fp32 GEMM | 2259 |
| 6x nvjet GEMMs | 2124 |
| MoE bmm1/bmm2 | 855 / 322 |
| causal FMHA trtllm-gen | 645 |
| _v1_flat pack x3 | 384 |
| attn_res x2 | 352 |
Prefix FMHA = 66% of the step (pf131072: 8914 of 16820; 32k points: 17255 / 29275).

Finding: the accepted cute-dsl prefix FMHA (mla_prefill_claude_2) is NOT running. Log:
"cute-dsl fp8 ragged prefill unavailable (Misaligned Tensor data on argument #6 ... expected data alignment=16 bytes)".
prefix_chunk_cu_seq_lens is [num_chunks, bs+1] int32 -> row 1 is 8-byte aligned for bs=1; the prefix > 128k capacity
creates 2 chunks, chunk 1's cum_seqlen_k fails the 16B-alignment guard, and `_disable_cute_dsl_fp8_prefill` turns the
path off for the rest of the process -> every point (incl. the single-chunk pf131072 ones measured after) runs trtllm-gen.
Target: node attention.mla_prefill_attention_prefix -> fix the alignment so the faster kernel runs.

### result.json

```
{"iter":0,"latency_us":{"1,16384,pf245760":25099.1,"1,32768,pf229376":45709.5,"1,16384,pf131072":16818.6,"1,32768,pf131072":33161.5},"note":"capture run, plain flags"}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 — re-enable the cute-dsl prefix FMHA on >128k prefixes (16B-align cum_seqlen)
Node: attention.mla_prefill_attention_prefix (66% of the measured step at pf245760).
Cause: `forward_batch.prefix_chunk_cu_seq_lens[chunk_idx]` is a row of a [num_chunks, bs+1] int32 tensor; at bs=1 row 1
sits at +8 B. The cute-dsl kernel is compiled for 16B-aligned cum_seqlen tensors, raises "Misaligned Tensor data on
argument #6", and the backend's exception handler disables the cute-dsl path for the whole process (so even the
single-chunk pf131072 points ran trtllm-gen).
Change: `_align16()` in cute_dsl_fmha_fp8.py clones a misaligned indptr (8 bytes; caching-allocator blocks are 512B aligned).
Why numerics hold: this is exactly the kernel accepted in mla_prefill_claude_2 (full softmax correction, fp8 P prescale 2^8,
trtllm-gen accuracy; rel 0.0156 there); it only lets it run where it was meant to. Builds on mla_prefill_claude_2.

### analysis.md

# iter_01 — analysis
VibeSim: `.:k3_mla_prefill` (p_9318587543334f74b338aa9dcb102cad, fixed before-state; re-fetch identical). Target node
attention.mla_prefill_attention_prefix, ladder R0 3.53 / R5 1.37 ms at the modeled pf49152 (R0/R5 2.57); on this workload
the node is 5x larger (prefix 245,760) and 66% of the measured step.
Post-edit profile (pf245760): cute-dsl BlackwellFusedMultiHeadAttentionForward x2 = 15071 us (was trtllm-gen 16695).
pf131072 16k: 8130 us (was 8914); 32k points 27181 (was 29275) / 15870 (was 17255).
Remaining: prefix FMHA is still 65% of the step. Grid = ceil(16384/256)=64 q-tiles x 12 heads = 768 CTAs, 1 CTA/SM on 148
SMs -> 5.19 waves executed as 6 (last wave 28/148 SMs busy) in EVERY prefix-chunk launch: ~13.5% of the FMHA is tail.
(32k-query points: 1536 CTAs = 10.38 -> 11 waves, 5.6%.) -> iter_02: fill the tail by splitting KV into S batch entries.

### result.json

```
{
 "replay_lines": [
  "CHECK {\"max_abs_err\": 0.171875, \"max_rel_err\": 0.009046, \"mean_rel_err\": 2.1e-05, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "JSON {\"B\": 1, \"seq_len\": 16384, \"mixed\": false, \"tag\": \"pf245760\", \"prefill\": true, \"prefix_len\": 245760, \"num_tokens\": 16384, \"ok\": true, \"latency_us\": 23358.720779418945, \"latency_mode\": \"eager\", \"us_step\": 23358.720779418945, \"us_step_graph\": null, \"us_attn\": null, \"us_moe\": null, \"us_norms\": null, \"finite\": true, \"graph_finite\": null, \"attention_backend\": \"trtllm_mla\", \"attn_heads\": 12, \"seed\": 0, \"error\": null, \"point\": [1, 16384, \"pf245760\"], \"correctness\": {\"max_abs_err\": 0.171875, \"max_rel_err\": 0.009046052155470939, \"mean_rel_err\": 2.1348190784920007e-05, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 32768, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "JSON {\"B\": 1, \"seq_len\": 32768, \"mixed\": false, \"tag\": \"pf229376\", \"prefill\": true, \"prefix_len\": 229376, \"num_tokens\": 32768, \"ok\": true, \"latency_us\": 42718.46389770508, \"latency_mode\": \"eager\", \"us_step\": 42718.46389770508, \"us_step_graph\": null, \"us_attn\": null, \"us_moe\": null, \"us_norms\": null, \"finite\": true, \"graph_finite\": null, \"attention_backend\": \"trtllm_mla\", \"attn_heads\": 12, \"seed\": 0, \"error\": null, \"point\": [1, 32768, \"pf229376\"], \"correctness\": {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 32768, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": fa
[... truncated]
```

(diff.patch: 26 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/cute_dsl_fmha_fp8.py)

## iter_02
### hypothesis.md

# iter_02 — hypothesis
Prior trials: mla_prefill_claude_2/3 (prefix FMHA dominant, backend switch accepted), mla_b512_claude_2 (tail split of a sub-wave-count grid helped).
Hypothesis: the 6th wave of the 768-CTA prefix FMHA runs 28/148 SMs; splitting KV S-ways (flash-decoding style across batch entries) fills it,
predicted -13% FMHA at 16k by the wave model. Numerics: fp32 merge of per-split (O, LSE) — same math as merge_state; must be CHECK-exact.

### analysis.md

# iter_02 — analysis
VibeSim: `.:k3_mla_prefill` (p_9318587543334f74b338aa9dcb102cad, fixed before-state). Node attention.mla_prefill_attention_prefix
(R0 3.53 / R5 1.37 ms at the modeled pf49152, R0/R5 2.57; necessary_share null). The node is still 65% of the measured step after iter_01.
Mapped source: sglang/kernels/ops/attention/cute_dsl_fmha_fp8.py (cute-dsl BlackwellFusedMultiHeadAttentionForward), called per prefix chunk
from trtllm_mla_backend._run_prefill_kernel (forward_mha._chunked_prefix_attn_mha, 2 chunks of <=131072).
Wave-tail: grid ceil(s_q/256) x 12 heads; 16k q -> 768 CTAs = 5.19 waves on 148 SMs (6 executed), 32k -> 10.38 (11).
Edit: split KV into S batch entries over replicated Q (grid x S) + Triton log2-domain merge; S chosen by a wave-cost model (S=5 at 16k).
Profile pf245760 (iter_02b run): FMHA x2 15071 -> 13367 us (+ merge 120, Q memcpy 112). 32k: pf229376 FMHA -740 us, pf131072 no gain -> net regression vs iter_01 there.
(iter_02 first run failed CHECK: merge used natural exp/log, the kernel LSE is log2; iter_02 initial Q copy via expand().reshape was 623-1612 us -> memcpy.)

### result.json

```
{
 "replay_lines": [
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "JSON {\"B\": 1, \"seq_len\": 16384, \"mixed\": false, \"tag\": \"pf245760\", \"prefill\": true, \"prefix_len\": 245760, \"num_tokens\": 16384, \"ok\": true, \"latency_us\": 21975.391387939453, \"latency_mode\": \"eager\", \"us_step\": 21975.391387939453, \"us_step_graph\": null, \"us_attn\": null, \"us_moe\": null, \"us_norms\": null, \"finite\": true, \"graph_finite\": null, \"attention_backend\": \"trtllm_mla\", \"attn_heads\": 12, \"seed\": 0, \"error\": null, \"point\": [1, 16384, \"pf245760\"], \"correctness\": {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 32768, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "JSON {\"B\": 1, \"seq_len\": 32768, \"mixed\": false, \"tag\": \"pf229376\", \"prefill\": true, \"prefix_len\": 229376, \"num_tokens\": 32768, \"ok\": true, \"latency_us\": 42954.91027832031, \"latency_mode\": \"eager\", \"us_step\": 42954.91027832031, \"us_step_graph\": null, \"us_attn\": null, \"us_moe\": null, \"us_norms\": null, \"finite\": true, \"graph_finite\": null, \"attention_backend\": \"trtllm_mla\", \"attn_heads\": 12, \"seed\": 0, \"error\": null, \"point\": [1, 32768, \"pf229376\"], \"correctness\": {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 32768, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows
[... truncated]
```

(diff.patch: 205 lines, files: /tmp/iter02_tree/sglang/kernels/ops/attention/cute_dsl_fmha_fp8.py, /tmp/iter02_tree/sglang/srt/layers/attention/trtllm_mla_backend.py)

## iter_03
### hypothesis.md

# iter_03 — hypothesis
Cites iter_02 (wave-model split: pf245760 -1384 us, 32k points regressed vs iter_01) and mla_b512_claude_2 (tail split).
Hypothesis: picking S from the measured per-shape optimum (4 for <=6-wave 16k grids, 2 for 32k) keeps the 16k gain and turns the 32k points into small wins.

### analysis.md

# iter_03 — analysis
VibeSim node unchanged: `.:k3_mla_prefill` (p_9318587543334f74b338aa9dcb102cad) attention.mla_prefill_attention_prefix, R0/R5 2.57.
iter_02's wave model over-predicts: microbenchmark (tools/mb_split2.py, standalone FMHA+merge, median of 6 interleaved runs):
| tq / kv | S=2 | S=3 | S=4 | S=5 | S=8 |
|---|---|---|---|---|---|
| 16k / 131072 | -5.2% | -7.0% | -8.7% | -7.0% | -3.2% |
| 16k / 114688 | -5.6% | -6.8% | -7.3% | -7.1% | -2.9% |
| 32k / 131072 | -2.2% | -0.1% | -0.2% | +0.3% | +3.1% |
| 32k / 98304  | -2.3% | +0.3% | +0.7% | +1.6% | +5.3% |
Extra Q loads per split cost more than the tail saves beyond S=4 (16k) / S=2 (32k).
Edit: empirical policy S=4 if num_ctas <= 6*num_sms else 2, halving while kv < S*16384.
Profile: pf245760 FMHA x2 14119 us (merge 150); pf229376 FMHA 28680 (iter_01 29444); 32k pf131072 17947 (iter_01 18215); 16k pf131072 8194 (8781).
Accuracy with random V vs fp32 reference: splits equal or better than S=1 (S=1 relO mean 1.23e-2; S=4 8.7e-3).

### result.json

```
{
 "replay_lines": [
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "JSON {\"B\": 1, \"seq_len\": 16384, \"mixed\": false, \"tag\": \"pf245760\", \"prefill\": true, \"prefix_len\": 245760, \"num_tokens\": 16384, \"ok\": true, \"latency_us\": 21996.7041015625, \"latency_mode\": \"eager\", \"us_step\": 21996.7041015625, \"us_step_graph\": null, \"us_attn\": null, \"us_moe\": null, \"us_norms\": null, \"finite\": true, \"graph_finite\": null, \"attention_backend\": \"trtllm_mla\", \"attn_heads\": 12, \"seed\": 0, \"error\": null, \"point\": [1, 16384, \"pf245760\"], \"correctness\": {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 32768, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "JSON {\"B\": 1, \"seq_len\": 32768, \"mixed\": false, \"tag\": \"pf229376\", \"prefill\": true, \"prefix_len\": 229376, \"num_tokens\": 32768, \"ok\": true, \"latency_us\": 42360.191345214844, \"latency_mode\": \"eager\", \"us_step\": 42360.191345214844, \"us_step_graph\": null, \"us_attn\": null, \"us_moe\": null, \"us_norms\": null, \"finite\": true, \"graph_finite\": null, \"attention_backend\": \"trtllm_mla\", \"attn_heads\": 12, \"seed\": 0, \"error\": null, \"point\": [1, 32768, \"pf229376\"], \"correctness\": {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 32768, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_o
[... truncated]
```

(diff.patch: 202 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/cute_dsl_fmha_fp8.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py)
