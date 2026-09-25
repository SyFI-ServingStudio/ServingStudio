# MLA_B512_CLAUDE trial 1: agent iterations

## agent log.md

# Kimi-K3 MLA layer, B200 -- latency_us (plain run.sh flags), B512@8k / B256@8k / B16@64k mix
- iter_00: baseline seeded tree: 976.1 / 638.3 / 273.7; golden captured; VibeSim top node mxfp4_fused_moe (r0/r5 16.9).
- iter_01: MoE fallback tactic (16,377) at >=4 rows/expert (flashinfer_trtllm.py): 914.1 / 627.1 / 273.7; CHECK pass x3, rel 0.0.
- iter_02: split merged front (routed cols first, gate_up on alt stream), 3 variants: 931.1..933.3 / 625.5..634.2 (B16 272.0); CHECK pass; slower -> reverted.
- iter_03: set_mla_kv_concat_q fp8 kernel: hw satfinite cvt + exact NOSAT fixup (set_mla_kv_concat_q.cuh), kernel 17.8->8.0 us: 904.6 / 617.9 / 269.7; CHECK pass x3, rel 0.0.

## iter_00
### hypothesis.md

# iter_00 hypothesis (sets up iter_01)

The MoE GEMMs run the `t128x8x512` cubin: tile_N = 8 tokens. At B=512, top-2, 112 local experts
there are ~9 rows/expert, so every expert's FP4 weight tile is streamed from HBM twice (two token
tiles). FlashInfer with no autotune returns tactic -1; C++ `selectDefaultTileN` takes the SMALLEST
candidate around nextPow2(rows/expert). A tile_N=16 tactic should read the weights once.

Prior trials:
- Builds on TECHNIQUES B "TRT-LLM MoE tactic buckets / tactic re-selection: no effect", with its note
  "new m values (512, 1024, 16k+) are untuned -- worth one look". At B<=128 (<=2.3 rows/expert) tile 8
  is correct, which is why that trial found nothing; this workload is out of that regime.
- mla_6 / mla_27 (flashinfer_trtllm.py tweaks, 0.0%) were at decode m; not the same lever.
- Must not disturb A9 / mla_20 (route_quant_fused specialisation) that feeds this MoE call.

### analysis.md

# iter_00 -- baseline (seeded tree, no edit)

Plain run (run.sh): B512@8k 976.1 us, B256@8k 638.3 us, B16@64k mix 273.7 us.
Golden captured from this tree -> /tmp/golden.pt (+ per-point golden_*.pt).

## VibeSim
- workspace-info: prediction `k3_mla_b512` = p_6b53ccf740d742fd8779d087584820d7 (kimi_k3_sglang, B200);
  handle used for all queries: `prediction=.:k3_mla_b512` (vs_*.json in this folder).
- run_summary: optimality_ratio 0.24; batching gap 48%, imbalance 19%.
- optimality (scope=iter, sorted by r0-r5 headroom; totals summed over 3 cases):
  | node | r0 us | r5 us | r6 us | r0/r5 |
  |---|---|---|---|---|
  | mxfp4_fused_moe | 776.9 | 46.0 | 245.7 | 16.9 |
  | mla_decode_attention | 686.4 | 502.7 | 1000 (>r0) | 1.37 |
  | merged_front | 188.1 | 116.4 | 83.8 | 1.62 |
  | shared_down | 73.8 | 43.0 | - | 1.72 |
  | latent_up | 47.0 | 25.3 | - | 1.86 |
  | fused_qkv_a_proj | 40.4 | 15.1 | - | 2.7 |
  | output_gate | 31.9 | 11.1 | - | 2.9 |
  | o_proj | 24.1 | 11.1 | - | 2.2 |
- Reading: MoE is "r0/r5 large, r6 nonzero" = kernel-efficiency bottleneck -> top candidate.
  MLA decode is bandwidth-bound (r6 > r0); only scheduling knobs remain.

## Measured profile, B=512 (profile.json, --split --profile-kernels, diagnostic only)
MLA fmha 384.7 | MoE GEMM1 t128x8x512 258.5 | MoE GEMM2 t128x8x512 119.3 | nvjet 256x128 x3 74.9 |
nvjet 224x128 (merged_front) 73.5 | nvjet 64x128 x2 21.2 | set_mla_kv_concat_q 17.8 | attn_res 15.8 |
nvjet 96x64 13.9 | situ_and_mul 12.9 | add/add3/finalize ~25. Kernel sum 1062.9, graph 982.4.

## Node picked
mxfp4_fused_moe (378 us of kernels at B512). Source: srt/layers/moe/moe_runner/flashinfer_trtllm.py
-> flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe. The cubin name shows tile_N=8.

### result.json

```
JSON {"B": 512, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 976.0640263557434, "latency_mode": "graph", "us_step": 1716.5440320968628, "us_step_graph": 976.0640263557434, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [512, 8192]}
JSON {"B": 256, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 638.3360028266907, "latency_mode": "graph", "us_step": 1737.056016921997, "us_step_graph": 638.3360028266907, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [256, 8192]}
JSON {"B": 16, "seq_len": 65536, "mixed": true, "tag": "mix", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 273.72801303863525, "latency_mode": "graph", "us_step": 1946.4319944381714, "us_step_graph": 273.72801303863525, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [16, 65536, "mix"]}
```

(diff.patch: 1 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis

Built on iter_00 (tile_N=8 fallback re-streams weights) and TECHNIQUES B "tactic re-selection" (decode
m, no effect -- re-evaluated here at untuned m). Forcing the per-tile anchors (16,57)/(32,50) regressed
(config index matters), so the lever is the autotuner's measured pick (16,377), applied only when
FlashInfer would fall back (tactic -1, no profiling/file cache for this op, not in tuning mode), only for
K3's latent MoE shape (hidden 3584, intermediate 3072), only at >= 4 rows/expert, and only if the tactic
is in the runner's valid list. Retiling tokens does not change per-row accumulation order -> expected
bit-exact output (confirmed: max_rel_err 0.0).

Result: 976.1->914.1 (-6.3%), 638.3->627.1 (-1.8%), 273.7->273.7 (guard off at B16 mix: 32 rows).

### analysis.md

# iter_01 -- MoE tile_N fallback tactic (16, 377)

## Experiment (tactic sweep in graph replay, exp_tactic.py, out of tree)
| tactic | B512 | B256 | B16mix |
|---|---|---|---|
| default -1 -> (8,221) | 976 | 638 | 271.7 |
| (16,57) anchor | 998 | 680 | 291 |
| (32,50) anchor | 1043 | 677 | 304.5 |
| (16,377) autotuner pick | 913.8 | 626.1 | 273.8 |
| (16,769)/(16,391)/(16,783) | ~914 | ~628 | - |
Autotuner profiling (tune_log.jsonl): the default wins at <=128 tokens, tile 16 wins at 256 and 512 tokens
-> guard: rows (tokens*top_k) >= 4 * num_experts.

## Profile after (profile.json, B=512, diagnostic)
MoE GEMM1 t128x16x256 216.8 (was 258.5), GEMM2 99.7 (was 119.3): MoE 316.5 vs 377.8 us.
MLA fmha 385.7 (unchanged). Kernel sum 1003.4, graph 920.9.

## VibeSim (prediction p_6b53ccf740d742fd8779d087584820d7, `.:k3_mla_b512`)
VibeSim holds a fixed prediction (no profile upload); the new profile is compared to its ladder:
MoE is now ~1.97 GB weights+scales in 316 us (~6.2 TB/s, ~80% HBM). r5 (46 us/case-sum) assumes
ideal fusion; the remaining gap is inside the closed cubin. MLA decode 2.42 GB KV / 386 us = 6.26 TB/s,
also ~80% of HBM, and r6 > r0 there -> bandwidth-bound. The bottleneck is now shared by two
near-BW-bound closed kernels (~700 of ~915 us); next is the cuBLAS front GEMMs (merged_front 73.6 us
vs ~53 us compute roofline at m=512).

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 512, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 914.143979549408, "latency_mode": "graph", "us_step": 1442.2719478607178, "us_step_graph": 914.143979549408, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [512, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 256, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 627.0719766616821, "latency_mode": "graph", "us_step": 1413.5359525680542, "us_step_graph": 627.0719766616821, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [256, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "mixed": true, "tag": "mix", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 273.72801303863525, "latency_mode": "graph", "us_step": 1601.9840240478516, "us_step_graph": 273.72801303863525, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attenti
[... truncated]
```

(diff.patch: 241 lines, files: /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_02
### hypothesis.md

# iter_02 hypothesis

Built on TECHNIQUES A4 (shared/routed alt-stream overlap, mla_24) and its note "at large m both
branches are compute-bound so the gain shrinks"; at B=512 the MoE is still ~80% HBM-bound (iter_01
analysis), so moving the 12288-col shared gate_up GEMM (~56 us) off the critical path and into the
MoE window could save ~45-50 us. The m<=16 TGV path (A2, mla_21) is kept by gating at >=17 tokens.
Split along weight rows (a contiguous view of the merged weight); expected bit-exact (it was).

Outcome: ruled out (+17 us at B512). SM contention with cluster-launched routing and GEMM2 outweighs
the overlap; carveout did not help. Not retried in later iterations.

### analysis.md

# iter_02 -- split merged MoE front (ruled out)

## Profile (profile.json: nsys graph-node timelines, B=512)
Critical path on the iter_01 tree (single stream except the MoE branch), 910 us span:
front 52 | MLA fmha 386 | post-attn (v_up, o_proj, gate, attn_res) 41 |
merged_front nvjet 224x128 72.7 (hidden 7168 -> [gate_up 12288 | router 112 | latent 3584], fp32 out) |
router/quant/routing 17 | MoE GEMM1 213 + GEMM2 97.5 + finalize 8 | norm + latent_up 21.7 + add3 8.
Only shared situ (12) + shared_down (41) overlap the MoE today.

## VibeSim (`.:k3_mla_b512`, p_6b53ccf740d742fd8779d087584820d7)
merged_front: r0 188.1 / r5 116.4 / r6 83.8 (case sum), r0/r5 1.62. The "r6 < r0" gap reads as
batching/scheduling; the idea tested here is a scheduling change (the MoE only needs 23% of the
columns, so the other 77% could overlap the bandwidth-bound MoE).

## Result (plain flags, replay vs golden)
| variant | B512 | B256 | CHECK |
|---|---|---|---|
| iter_01 tree | 914.1 | 627.1 | pass, 0.0 |
| split, gate_up issued on alt before routed | 931.2 | 631.1 | pass, 0.0 |
| split, gate_up issued after top-k | 933.3 | 634.2 | pass, 0.0 |
| + cuBLAS SM carveout 32/64/96 for gate_up | 931.1 | 625.5..628.1 | pass, 0.0 |

Why it loses (nsys_split*.sqlite): (1) the routed-only GEMM (3696 cols) is one 132-CTA wave of
long-K tiles, 25.4 us instead of ~17; (2) the 256-CTA cluster-4 gate_up GEMM blocks the cluster-8
routingIndicesClusterKernel, so GEMM1 still starts at ~579 us (was 569); (3) shared_down then lands on
GEMM2 instead of GEMM1 and slows it 97 -> 124 us. Net +17 us. Reverted.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 512, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 931.1680197715759, "latency_mode": "graph", "us_step": 1502.5919675827026, "us_step_graph": 931.1680197715759, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [512, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 256, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 631.1360001564026, "latency_mode": "graph", "us_step": 1485.2479696273804, "us_step_graph": 631.1360001564026, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [256, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "mixed": true, "tag": "mix", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 272.0319926738739, "latency_mode": "graph", "us_step": 1633.631944656372, "us_step_graph": 272.0319926738739, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attentio
[... truncated]
```

(diff.patch: 137 lines, files: iter_02/kimi_k3_try2.py)

## iter_03
### hypothesis.md

# iter_03 hypothesis

The fused fp8 quantize+scatter+concat kernel is instruction-bound, not memory-bound, because the
NOSAT fp8 conversion is emulated. Replacing it with the hardware satfinite cvt plus an explicit
overflow/NaN fixup keeps the kernel's aten-matching semantics bit-for-bit and should cut it to its
latency floor (~5 us), saving ~10 us on the serial pre-attention path at every B.

Prior trials: builds on TECHNIQUES A8 / mla_25 (which introduced the 16-warp CTA for this kernel at
B=1) and rules in a different axis than the dead end "KV-concat CTA size changes beyond A8: +-0" --
CTA shape could not help because the time was ALU emulation, not occupancy or launch shape.
iter_02 (split front) is ruled out and not built on.

### analysis.md

# iter_03 -- fp8 KV/Q quantize kernel: hardware cvt instead of emulated NOSAT

## Node
VibeSim leaf `unified.mla.attention.mla_cache_append` (+ q concat) under the attention subtree
(prediction p_6b53ccf740d742fd8779d087584820d7, `.:k3_mla_b512`, vs_optimality.json). It sits on the
serial critical path right before the MLA fmha (iter_02 nsys timeline: 35.6..53.1 us).
Source: kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh, `set_mla_kv_concat_q_fp8_kernel`.

## ncu (B=512, before)
Grid 832x256, 25.7 us under ncu; DRAM 8.2% of peak, SM busy 51% (ALU pipe), 7.36M warp instructions =
~1100 per warp for 18 bf16->fp8 conversions per lane; 120k FP64 instructions. Data moved is ~7.7 MB
(~1.3 us at HBM rate). Cause: `__nv_cvt_float2_to_fp8x2(f, __NV_NOSAT, __NV_E4M3)` -- cuda_fp8.hpp
only maps __NV_SATFINITE to `cvt.rn.satfinite.e4m3x2.f32`; NOSAT runs the scalar software routine twice.

## Exhaustive equivalence (cvt_test*.cu)
satfinite and NOSAT differ exactly for |x| > 464 or non-finite: NOSAT gives sign|0x7F for overflow and
0x7F (sign dropped) for NaN. hw satfinite + that fixup vs the library NOSAT: 0 mismatches over all
65536 bf16 values x 64 partners in both lanes of the pair.

## Result
| kernel us (--profile-kernels) | B512 | B256 | B16mix |
|---|---|---|---|
| before | 17.8 | 17.0 | 5.7 |
| after | 8.0 | 7.2 | 3.1 |
Plain latency_us A/B (2 runs each): B512 914.8 -> 905.6, B256 627.5 -> 617.8. nsys span 910.5 -> 901.9.
Replay: 904.6 / 617.9 / 269.7, CHECK pass, max_rel_err 0.0, state_ok on all points.
Remaining ~5 us of this node is PDL wait + fixed latency (B256 and B512 take the same time).

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 512, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 904.6080112457275, "latency_mode": "graph", "us_step": 1455.5200338363647, "us_step_graph": 904.6080112457275, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [512, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 256, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 617.8879737854004, "latency_mode": "graph", "us_step": 1482.1759462356567, "us_step_graph": 617.8879737854004, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [256, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "mixed": true, "tag": "mix", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 269.6639895439148, "latency_mode": "graph", "us_step": 1618.399977684021, "us_step_graph": 269.6639895439148, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attentio
[... truncated]
```

(diff.patch: 24 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh)
