# MLA_VERIFY_CLAUDE trial 1: agent iterations

## agent log.md

# log (latency_us plain graph: 64x8k vk3 / 128x8k vk3 / 16x64k vk3)
- iter_00: baseline 493.0 / 658.9 / 424.4; golden captured. MoE at HBM roofline; MLA split-kv 128 CTAs; GEMMs ~50% roofline.
- iter_01: satfinite fp8 cvt in set_mla_kv_concat_q + shared/routed alt-stream overlap: 478.7 / 632.3 / 404.9; CHECK pass x3 (rel 0.0). trtllm-gen MLA (A1) ruled out by microbench (2x slower at q_len=4).
- iter_02: MLA output-gate g_proj overlap enabled up to 256 tok (_gate_bs_limit): 474.8 / 634.2 / 404.8; CHECK pass x3. MLA split_kv override + TGV/deep_gemm GEMM swaps ruled out by microbench.
- iter_03: gate fork event at forward_absorb_core, g_proj enqueued after the MLA node (overlaps the MLA, not qkv_a), limit 512: 472.4 / 626.2 / 400.7; CHECK pass x3.

## iter_00
### analysis.md

# iter_00 — baseline (start tree == pristine v0.5.20 layer; no prior levers applied)
Plain run: 64x8k vk3 493.0 us, 128x8k vk3 658.9, 16x64k vk3 424.4 (graph). Golden -> /workspace/opt_run/golden.pt.
VibeSim: workspace serves a fixed before-state prediction p_c7687e585abe4b5fbacdc89caf0c9761 (k3_mla, decode cases;
the prediction id itself is not accepted by analyze/optimality -> queried with the default workspace).
run_summary: optimality 0.127; hardware_gap 37%, batching 33%, imbalance 18%.
optimality (iter): mxfp4_fused_moe r0 379 / r6 228 (necessary_share 0.60) ; attention+gate r0 436, r6 462 (share 1.06, BW-bound);
mla_decode_attention r0 359 r5 228 r6 454 (share 1.27 -> necessary, BW-bound); merged_front r0 159 r5 121 r6 30 (share 0.19);
worst batching kernels: merged_front, shared_down, latent_up, mla_decode, o_proj, output_gate, fused_qkv_a.
Measured (nsys, B=64 vk3, single stream): MoE gemm1 165 + gemm2 86 (1.85 GB mxfp4 weights -> ~7.4 TB/s, at HBM roofline);
MLA cute-dsl split-kv 57 us (302 MB fp8 KV -> 5.3 TB/s, grid 2x64=128 CTAs < 148 SMs); dense bf16 GEMMs at m=256 at ~50% of
their roofline (merged_front 47, shared_down 25, qkv_a 13.7, latent_up 13.5, o_proj 8.5+5); set_mla_kv_concat_q_fp8 17 us for
~2 MB of data; shared experts (situ 10 + shared_down 25) serial before routed MoE.
Decision: MoE cubin is at HBM roofline -> not a code target. Targets: shared-expert overlap (A4), fp8 cvt (mla_b512_claude_1 it3),
then MLA split scheduling, then dense GEMMs.

### result.json

```
JSON {"B": 64, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 256, "ok": true, "latency_us": 493.02399158477783, "latency_mode": "graph", "us_step": 1380.8319568634033, "us_step_graph": 493.02399158477783, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [64, 8192, "vk3"]}
JSON {"B": 128, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 512, "ok": true, "latency_us": 658.847987651825, "latency_mode": "graph", "us_step": 1357.0879697799683, "us_step_graph": 658.847987651825, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192, "vk3"]}
JSON {"B": 16, "seq_len": 65536, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 64, "ok": true, "latency_us": 424.3839979171753, "latency_mode": "graph", "us_step": 1379.8719644546509, "us_step_graph": 424.3839979171753, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [16, 65536, "vk3"]}
```

## iter_01
### hypothesis.md

# iter_01
1. set_mla_kv_concat_q_fp8: replace the software-emulated __NV_NOSAT cvt with hardware satfinite + exact NOSAT fixup for |x|>464/NaN
   (transferred from mla_b512_claude_1 iter_03; proven bit-identical over all bf16 inputs there).
2. KimiK3MoE._forward_fused (non-AR-fusion branch): shared experts (situ_and_mul + shared_down) on alt_stream concurrently with the
   routed MXFP4 MoE; join before the collective (TECHNIQUES A4 / mla_24). Disjoint output slices of buf -> identical numerics.
Ruled out: A1 (trtllm-gen MLA) — micro-benchmark (micro/mla_bench.py) shows trtllm-gen 150 us vs cute-dsl 68 us at q_len=4 x 12 heads.

### analysis.md

# iter_01 analysis (post-change profile, --split --profile-kernels, 64x8k vk3; diagnostic only)
VibeSim prediction id: p_c7687e585abe4b5fbacdc89caf0c9761 (fixed before-state prediction, default workspace; same as iter_00).
Profile: 29 launches, kernel sum 536.9 us vs graph latency 483.5 us -> ~54 us now hidden by the shared/routed overlap.
set_mla_kv_concat_q_fp8: 17.1 -> 7.4 us (satfinite cvt). Remaining top: MoE gemm1 168.7 / gemm2 87.6 (HBM roofline, VibeSim
necessary_share 0.60 -> not a target), MLA 57.7, merged_front 47.5, 112x128 nvjet x3 48.3 (shared_down on side stream, o_proj-like,
latent_up), 64x64 nvjet x2 19.4 (fused_qkv_a + g_proj output gate; g_proj on the critical path because 256 tok > _gate_bs_limit 128).
Smoke (4,8192,vk3): LAYER_SMOKE_OK (iter_01/smoke.log).

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 64, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 256, "ok": true, "latency_us": 478.68800163269043, "latency_mode": "graph", "us_step": 1432.0319890975952, "us_step_graph": 478.68800163269043, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [64, 8192, "vk3"], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 512, "ok": true, "latency_us": 632.319986820221, "latency_mode": "graph", "us_step": 1396.191954612732, "us_step_graph": 632.319986820221, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192, "vk3"], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 64, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_to
[... truncated]
```

(diff.patch: 49 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh, /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)

## iter_02
### hypothesis.md

# iter_02
Built on iter_01 (shared/routed overlap works; kernel sum - latency = 54 us hidden).
Ruled out this iteration (micro-benchmarks, B200, CUDA graph, L2-defeating weight rotation):
- MLA cute-dsl split_kv override (micro/mla_split.py): forcing split 1..16 never beats the built-in heuristic at any point
  (B64: 67 us at split1 = default; B128 default 136 vs >=160; 16x64k default split4 131). Not a lever.
- Dense GEMM backend swap (micro/gemm_bench.py, micro/dg_bench.py): TGV cutedsl loses at m=256/512 on every K3 shape
  (merged_front 74 vs cuBLAS 49); deep_gemm bf16_gemm_nt ties cuBLAS exactly. Not a lever at the primary point.
Hypothesis: the MLA output-gate g_proj (1536x7168, 11.4 us at m=256) runs on the critical path at B=64 because
_gate_bs_limit=128 tokens on Blackwell; the cute-dsl MLA kernel launches only 128 CTAs at B=64 (20 SMs idle), so issuing g_proj
on the gate alt stream (existing _precompute_output_gate path) hides it. Expect -4..-8 us at 64x8k; 128x8k (512 tok) tested with
limit 512 -> neutral (634.2 vs 632.3), so limit set to 256; 16x64k (64 tok) already overlapped -> unchanged. Same GEMM, same inputs
-> bit-identical.

### analysis.md

# iter_02 analysis (64x8k vk3)
VibeSim prediction id: p_c7687e585abe4b5fbacdc89caf0c9761 (default workspace; the worst batching kernels it lists include
output_gate and fused_qkv_a, which is the pair this iteration targets).
profile.json (--split --profile-kernels, diagnostic): 29 launches, kernel sum 553.6 us, latency 480.6 us.
nsys (nsys/iter02_cuda_gpu_trace.csv, plain graph, one step = 471.5 us): g_proj (splitK 10.1 + reduce 13.2) now on the side
stream, but it is issued at attention entry so it overlaps fused_qkv_a (13.7 -> 21.3 us); the MLA starts at 45.2 and ends at
101.7. The overlap goes to another HBM-bound GEMM, not to the MLA's idle SMs, which leads to iter_03.
Also seen: routingIndicesClusterKernel waits about 16 us for SMs held by shared_down (side stream), so gemm1 starts only after
shared_down ends. The overlap hides router+quant+routing (~17 us), but everything here is HBM-bound, so reordering does not reduce bytes.
Replay: 474.8 / 634.2 / 404.8, CHECK pass x3 (rel 0.0). Smoke LAYER_SMOKE_OK.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 64, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 256, "ok": true, "latency_us": 474.8159945011139, "latency_mode": "graph", "us_step": 1451.3920545578003, "us_step_graph": 474.8159945011139, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [64, 8192, "vk3"], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 512, "ok": true, "latency_us": 634.2399716377258, "latency_mode": "graph", "us_step": 1428.8959503173828, "us_step_graph": 634.2399716377258, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192, "vk3"], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 64, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_t
[... truncated]
```

(diff.patch: 64 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh, /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)

## iter_03
### hypothesis.md

# iter_03
Built on iter_02 (g_proj on the gate alt stream at 256 tokens; nsys shows it overlapping fused_qkv_a instead of the MLA).
Hypothesis: fork the gate at attention-core entry (forward_absorb_core override: record an event) and enqueue g_proj on the alt
stream from the o_proj wrap, i.e. after the MLA node, ordered only after the fork event. The graph then runs g_proj next to the
cute-dsl MLA kernel (128 CTAs at B=64 -> 20 idle SMs; 256 CTAs at B=128 -> a 1.73-wave tail), not next to qkv_a or set_mla_kv.
Trials: fork at forward_absorb_core start + immediate issue -> 474.5/631.2/402.9 (g_proj delays set_mla 7->13 us);
event fork + deferred issue, limit 256 -> 472.4/632.1/400.6 (MLA starts at 36.2 instead of 42.7-45.2, ends 98.4);
same with limit 512 -> 128x8k 627.2 (kept). Same GEMM, same input -> bit-identical.
Ruled out: merging g_proj into fused_qkv_a (micro/merge_bench.py: n=3648 16.8 us at m=256 vs 12.2 qkv_a alone, and it loses at m=64).

### analysis.md

# iter_03 analysis (64x8k vk3)
VibeSim prediction id: p_c7687e585abe4b5fbacdc89caf0c9761 (default workspace). mla_decode_attention is BW-bound at share 1.27
(the kernel runs at 5.3 TB/s with 128 CTAs), so there is HBM headroom to hide the 22 MB g_proj read under it.
nsys (nsys/iter03b_cuda_gpu_trace.csv, limit-256 trial): qkv_a back to 14.1 us, set_mla 6.5 us, MLA 36.2 -> 98.4 (62.2 us,
+5 us from sharing HBM with g_proj), g_proj splitK 12.9 + reduce 24.5 fully under the MLA. The next critical path is unchanged
from iter_02: merged_front 46.5, shared/routing 31, MoE 250, tail 24.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 64, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 256, "ok": true, "latency_us": 472.4479913711548, "latency_mode": "graph", "us_step": 1424.8000383377075, "us_step_graph": 472.4479913711548, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [64, 8192, "vk3"], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 512, "ok": true, "latency_us": 626.1759996414185, "latency_mode": "graph", "us_step": 1394.11199092865, "us_step_graph": 626.1759996414185, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192, "vk3"], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 64, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tok
[... truncated]
```

(diff.patch: 163 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh, /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)
