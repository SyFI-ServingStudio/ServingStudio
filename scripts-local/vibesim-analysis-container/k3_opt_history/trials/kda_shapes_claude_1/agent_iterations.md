# KDA_SHAPES_CLAUDE trial 1: agent iterations

## agent log.md

- iter_00: baseline 1x1M 128.4 | 16x64k 183.7 | 128x8k 376.3 | 32x8k 243.2 | 1x8k 128.4 us (plain). Bottleneck: 41us merged front serial ahead of latency-bound routed MoE chain.
- iter_01: split merged MoE front at m<=32 (routed rows first, gate_up+shared on alt stream, latent tail before join), TGV fp32 halves: 1x1M 128.4->112.1 | 16x64k 183.7->171.5 | 128x8k 376.3->376.4 | 32x8k 243.2->235.0 | 1x8k 128.4->112.1; CHECK pass all, rel 0.0

## iter_00
### analysis.md

# iter_00 analysis (baseline = tree as handed over)
VibeSim prediction: `.:k3_kda` (prediction_id p_38ccb83a6c4e44298f45b2251bd4cf54, fixed before-state; `simulate?prediction=.:before` returns it).
- analyze run_summary: optimality 0.077; gap buckets: hardware_gap 61%, batching 29% (worst batching: merged_front 93us-batching, shared_down, latent_up).
- analyze operator: mxfp4_fused_moe 52.7%, merged_front 15.3%, qkvbfg 6.6%, shared_down 6.4%, kda_recurrent 5.4%, latent_up 4.2%.
- optimality iter: mxfp4_fused_moe r0/r5=43.6 (launch/latency bound at small B, closed cubin), merged_front r0/r5=1.32 (near roofline),
  necessary_share null for all nodes (R6 not populated in this prediction).

Measured (plain flags, graph): 1x1M 128.4 | 16x64k 183.7 | 128x8k 376.3 | 32x8k 243.2 | 1x8k 128.4 us.

nsys graph-node timeline, B=1 (iter_00/nsys_b1.sqlite), one step ~125us:
 attn_res 4.4 -> qkvg TGV 19.4 (|| bfa tiny GEMVs) -> KDA fused decode 8.1 -> o_proj TGV 6.4 -> add+attn_res 5.5
 -> merged_front nvjet (15984x7168 fp32) 41.2us  -> [routed chain on alt s143: router 4.8, quant 3.8, routing 5.6,
 GEMM1 12.7, GEMM2 8.3, finalize 5.1 = 30.5us] || [situ + shared_down TGV 15.4 on main] -> norm 2.5 -> latent_up 10.9 -> add3 3.1

Reading: merged_front is ~at the HBM roofline (r0/r5 1.32; microbench: TGV only 1us faster than cuBLAS, bench/gemm_bench.py), so a kernel
swap does not pay. The MoE (r0/r5 43) at B=1 is a latency-bound chain of 6 small launches that sits *serially* behind the whole 229MB front,
although it only needs the router+latent rows (3696 of 15984). The "batching" bucket on merged_front is exactly this scheduling gap.
Target: the dataflow between merged_front and mxfp4_fused_moe (critical-path scheduling), not a kernel.

## iter_01
### hypothesis.md

# iter_01 hypothesis
Split the merged MoE front at m<=32 (single-rank schedule): [router | latent down] rows first on the main stream (TGV fp32),
shared gate_up rows (TGV fp32) + SiTU + shared_down on the alt stream; routed MXFP4 chain + latent norm + latent_up on main,
join only before add3. The latency-bound routed chain (~30us of tiny launches, 33MB weights) then overlaps the 264MB of
bandwidth-bound shared-expert GEMMs instead of queueing behind them. Expected ~-15..25us at B=1.
Numerics: row views of the same merged weight; TGV does a full-K fp32 reduction per output and was measured bit-identical to
the merged cuBLAS output (0.0 abs err, m=1..64). Everything downstream is unchanged -> output and state bit-exact.
Prior trials: re-tests the dead end kda_b512_claude_1 iter_03 / mla_b512_claude_1 iter_02 (front split slower at B>=128, MoE GEMMs
SM-bound) in the latency-bound regime; gated to m<=32 so B=128 keeps the merged path. Builds on A4 (shared/routed alt-stream
overlap) and A2/mla_21 (fp32-output TGV for the front, bit-exact).

### analysis.md

# iter_01 analysis
Prediction `.:k3_kda` (p_38ccb83a6c4e44298f45b2251bd4cf54) + iter_00 nsys timeline. Node: unified.kda.moe.merged_front
(r0/r5 1.32, batching gap 93us of 120us case-sum -- the biggest batching entry) feeding unified.kda.moe.mxfp4_fused_moe
(r0/r5 43.6, latency bound at B<=32). Kernel swap on merged_front is worth ~1us (microbench bench/gemm_bench.py: TGV 35.0 vs
cuBLAS 35.9us, the GEMM is at ~6.5TB/s). The lever is the schedule: the routed chain waits for all 15984 front rows but uses 3696.
bench/split_bench.py: TGV [router|latent] 3696x7168 fp32 = 9.6us, gate_up 12288x7168 fp32 = 27.4us, both max_abs_err 0.0 vs the
merged cuBLAS fp32 front (cuBLAS halves are NOT exact: 4e-5, split-K).

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 1048576, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 112.09599673748016, "latency_mode": "graph", "us_step": 1683.4880113601685, "us_step_graph": 112.09599673748016, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [1, 1048576], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 171.48800194263458, "latency_mode": "graph", "us_step": 1410.688042640686, "us_step_graph": 171.48800194263458, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [16, 65536], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 128, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "mixed": false, "tag": "", "prefill": false, "prefix_len": null, "num_tokens": null, "ok": true, "latency_us": 376.3520121574402, "latency_mode": "graph", "us_step": 1304.927945137024, "us_step_graph": 376.3520121574402, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 1
[... truncated]
```

(diff.patch: 102 lines, files: /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)

## iter_02
(diff.patch: 4089 lines, files: b/kernels/jit/csrc/elementwise/l2_prefetch.cuh, b/kernels/ops/elementwise/l2_prefetch.py, b/srt/models/kimi_k3.py)
