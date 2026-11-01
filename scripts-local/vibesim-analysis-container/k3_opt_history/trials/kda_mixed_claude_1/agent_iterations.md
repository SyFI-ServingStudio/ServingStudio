# KDA_MIXED_CLAUDE trial 1: agent iterations

## agent log.md

# Kimi-K3 KDA mixed-batch optimization log (latency_us: mx16384 / mx4096 / mx16384p49152)
- iter_00: baseline, no edit. plain 8762.8 / 2815.9 / 8961.2 (golden run 8725.4 / 2760.0 / 8717.7). Top non-GEMM node: Triton h-scan 659 us.
- iter_01: h-scan gate -> CUDA kda_chunk_h when long seqs fit on SMs and chunk-steps saved outweigh 1-chunk seqs (fla/index.py, fla/chunk_delta_h.py). replay 8598.7 / 2764.9 / 8500.3, CHECK pass (exact, err 0). Split long/short launch variant: exact but not faster, dropped.
- iter_02: kda_conv_l2norm flattened (seq,strip) grid (jit/csrc/attention/kda_conv_l2norm.cuh, ops/attention/kda_conv_l2norm.py). replay 8381.6 / 2754.7 / 8471.6 (plain 8455.3 / 2747.4 / 8705.0), CHECK pass exact.

## iter_00
### hypothesis.md

# iter_00
Baseline only, no edit. The tree already carries the accepted kda_lcprefill_claude_4 levers (kda_chunk_h CUDA h-scan,
kda_conv_l2norm, situ_and_mul, pinned inter_solve) and the decode levers (TECHNIQUES A2-A9). A2/A3 (TGV small-m GEMMs)
do not engage at m = 16448 and the fused decode kernel (A5) does not run in EXTEND.

### analysis.md

# iter_00 baseline analysis
VibeSim prediction: `simulate?prediction=.:before` -> `p_7935251272b445188fbd5338ed932b36` (run k3_kda_mixed, 3 cases).
Raw: optimality.json, analyze_operator.json, run_summary.json.

run_summary: optimality 0.478; hardware_gap 27%, batching 19.6% (worst: mxfp4 MoE 1.54 ms, **kda_chunk_prefill 1.20 ms**, merged_front 0.77 ms), imbalance 5.5%.

optimality ladder (3 cases summed, r0 -> r5 ms):
- merged_front_prefill (fp32 front GEMM) 6.81 -> 3.75 (1.81x) -- nvjet at ~1.6 PFLOPS; prior trials (kda_prefill_claude_2 iter_00) found no faster library kernel.
- **kda_chunk_prefill 2.49 -> 0.057 (43.7x, largest ratio, 2.43 ms headroom, biggest batching gap)**
- mxfp4_fused_moe 4.24 -> 2.16 (1.96x) -- closed TRT-LLM cubin, tactics autotuned.
- qkvbfg_a_proj 2.62 -> 1.44; shared_down 2.52 -> 1.44; latent_up 1.48 -> 0.84 (all nvjet near ceiling).
necessary_share is null for every node (r6 not modeled), so the r0/r5 ratio is the discriminator.

Measured per-kernel (profile_B64_L8192mx16384.json, 8351 us kernels / 31 launches):
nvjet fp32 front 2352; nvjet tst x2 (badd) 1404; nvjet 944; MoE bmm1 895, bmm2 326;
**chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (Triton h-scan) 659 us**; attn_res x2 400; nvjet 228;
kda_conv_l2norm 190; situ 169; intra 124; inter_solve 107; chunk_o 106; recompute_w_u 98 ...

Decision: target kda_chunk_prefill -> the h-scan. The mixed batch has N = 65 sequences (1 chunk + 64 1-token decode rows),
so the CUDA kda_chunk_h gate `cdiv(V,16)*N*H <= #SMs` (6240 > 148) sends it to Triton. Source:
kernels/ops/attention/fla/chunk_delta_h.py::chunk_gated_delta_rule_fwd_h.
Plain baseline: 8762.8 / 2815.9 / 8961.2 us (golden-capture run: 8725.4 / 2760.0 / 8717.7).

### result.json

```
{"iter":0,"plain_latency_us":{"64,8192,mx16384":8762.8,"128,8192,mx4096":2815.9,"64,8192,mx16384p49152":8961.2},"golden_run_us":{"64,8192,mx16384":8725.4,"128,8192,mx4096":2760.0,"64,8192,mx16384p49152":8717.7}}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis — route the mixed-batch KDA h-scan to the CUDA warp-specialized kernel

Node: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (Triton h-scan), 659 us at the primary point (iter_00 profile), 2nd-largest non-GEMM node.

Built on: opt_history `kda_prefill_claude_2` (introduced CUDA `kda_chunk_h`, bit-exact vs Triton, ~2.5x per chunk step)
and `lcprefill_4` (its gate `cdiv(V,16)*N*H <= num_SMs`). In the mixed batch N = 65 so the gate always rejects CUDA even though
only one sequence is long. Ruled out: GEMM work (iter_00 / prior trials show nvjet at cuBLAS ceiling), bf16 front GEMM (numerics).

Hypothesis: gate on the number of *long* (>1 chunk) sequences and on the saved chunk-steps vs the ~1.2 us penalty per 1-chunk CTA group
(bench: 1x16384+64x1 triton 677 us vs cuda 243 us; 1x4096+128x1 triton 226 vs cuda 306 -> keep Triton). Expected ~-400 us at the primary
point, ~0 at mx4096 (Triton kept), ~-400 at p49152. Output bit-exact.

Side experiment (not adopted): split long-seq (4 stage) / short-seq (2-3 stage, alt stream) launches — exact but not faster
(255 vs 243 us), see hscan_split_bench.log.

### analysis.md

# iter_01 analysis
Kernel bench (bench/hscan.py, B200): CUDA scan exact (maxdiff 0) in all cases; faster when long sequences dominate
(1x16384+64x1: 677->243 us), slower with many 1-chunk sequences and short prefill (1x4096+128x1: 226->306 us) -> gate rule
`n_long*ctas_per_seq <= SMs and 11*max_chunks > 12*n_short`.
Replay (same command as golden): 8598.7 / 2764.9 / 8500.3 us vs baseline plain 8762.8 / 2815.9 / 8961.2 us.
CHECK: max_abs_err 0 on all three points, state_ok true. mx4096 unchanged path (difference is noise, ~±50 us).

### result.json

```
{
 "measure": "replay (identical to golden command, --replay)",
 "points": [
  {
   "point": [
    64,
    8192,
    "mx16384"
   ],
   "latency_us": 8598.655700683594,
   "check": {
    "max_abs_err": 0.0,
    "max_rel_err": 0.0,
    "mean_rel_err": 0.0,
    "nan": false,
    "rows": 16448,
    "rows_over_tol": 0,
    "frac_rows_over_tol": 0.0,
    "p99_row_rel_err": 0.0,
    "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)",
    "state_ok": true,
    "rel_err_max": 0.02,
    "pass": true
   },
   "baseline_plain_us": 8762.8
  },
  {
   "point": [
    128,
    8192,
    "mx4096"
   ],
   "latency_us": 2764.928102493286,
   "check": {
    "max_abs_err": 0.0,
    "max_rel_err": 0.0,
    "mean_rel_err": 0.0,
    "nan": false,
    "rows": 4224,
    "rows_over_tol": 0,
    "frac_rows_over_tol": 0.0,
    "p99_row_rel_err": 0.0,
    "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)",
    "state_ok": true,
    "rel_err_max": 0.02,
    "pass": true
   },
   "baseline_plain_us": 2815.9
  },
  {
   "point": [
    64,
    8192,
    "mx16384p49152"
   ],
   "latency_us": 8500.255584716797,
   "check": {
    "max_abs_err": 0.0,
    "max_rel_err": 0.0,
    "mean_rel_err": 0.0,
    "nan": false,
    "rows": 16448,
    "rows_over_tol": 0,
    "frac_rows_over_tol": 0.0,
    "p99_row_rel_err": 0.0,
    "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)",
    "state_ok": true,
    "rel_err_max": 0.02,
    "pass": true
   },
   "baseline_plain_us": 8961.2
  }
 ]
}
```

(diff.patch: 71 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/chunk_delta_h.py, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/index.py)

## iter_02
### hypothesis.md

# iter_02 hypothesis — kda_conv_l2norm: flatten the (seq, strip) grid
Node: `sglang::kda_conv_l2norm::kda_conv_l2norm_kernel`, 190 us at the primary point (iter_00 profile; 96 us at mx4096).
Grid was (dim/256=18, cdiv(max_seqlen,128), N): in the mixed batch (max_seqlen 16384, N=65) that is 18*128*65 = 150k CTAs,
~98% of which exit immediately (1-token decode rows own 1 strip each), so the kernel is launch/scheduling-bound, not HBM-bound.
Hypothesis: grid y = total strips = sum_i cdiv(len_i,128) (host list `extend_seq_lens_cpu`, no sync), each block finds its
(seq, strip) with a warp-wide inclusive scan over per-seq strip counts. Same per-token arithmetic -> bit-exact.
Expected: ~-90 us primary, ~-50 us mx4096.
Built on: iter_01 (h-scan gate, kept); opt_history kda_prefill_claude_2 (introduced the fused conv+l2norm kernel, whose
grid assumed uniform batches). Ruled out: attn_res (86 KB/token at 200 us = ~7 TB/s, HBM-bound), GEMMs (cuBLAS ceiling, prior trials).

### analysis.md

# iter_02 analysis
Replay (identical command): 8381.6 / 2754.7 / 8471.6 us (iter_01: 8598.7 / 2764.9 / 8500.3). Plain: 8455.3 / 2747.4 / 8705.0
(baseline plain 8762.8 / 2815.9 / 8961.2). CHECK: all 3 pass, max_abs_err 0, state_ok.
Next node: remaining KDA chunk pipeline (intra 124, inter_solve 107, chunk_o 106, recompute_w_u 98, cumsum 53 us) pays a full
64-token chunk per 1-token decode row (256 real + 64 padded chunks); VibeSim still flags kda_chunk_prefill batching gap.

### result.json

```
{
 "replay": [
  {
   "point": [
    64,
    8192,
    "mx16384"
   ],
   "latency_us": 8381.631851196289,
   "check": {
    "max_abs_err": 0.0,
    "max_rel_err": 0.0,
    "mean_rel_err": 0.0,
    "nan": false,
    "rows": 16448,
    "rows_over_tol": 0,
    "frac_rows_over_tol": 0.0,
    "p99_row_rel_err": 0.0,
    "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)",
    "state_ok": true,
    "rel_err_max": 0.02,
    "pass": true
   }
  },
  {
   "point": [
    128,
    8192,
    "mx4096"
   ],
   "latency_us": 2754.6560764312744,
   "check": {
    "max_abs_err": 0.0,
    "max_rel_err": 0.0,
    "mean_rel_err": 0.0,
    "nan": false,
    "rows": 4224,
    "rows_over_tol": 0,
    "frac_rows_over_tol": 0.0,
    "p99_row_rel_err": 0.0,
    "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)",
    "state_ok": true,
    "rel_err_max": 0.02,
    "pass": true
   }
  },
  {
   "point": [
    64,
    8192,
    "mx16384p49152"
   ],
   "latency_us": 8471.58432006836,
   "check": {
    "max_abs_err": 0.0,
    "max_rel_err": 0.0,
    "mean_rel_err": 0.0,
    "nan": false,
    "rows": 16448,
    "rows_over_tol": 0,
    "frac_rows_over_tol": 0.0,
    "p99_row_rel_err": 0.0,
    "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)",
    "state_ok": true,
    "rel_err_max": 0.02,
    "pass": true
   }
  }
 ],
 "plain": [
  {
   "point": [
    64,
    8192,
    "mx16384"
   ],
   "latency_us": 8455.29556274414,
   "check": null
  },
  {
   "point": [
    128,
    8192,
    "mx4096"
   ],
   "latency_us": 2747.391939163208,
   "check": null
  },
  {
   "point": [
    64,
    8192,
    "mx16384p49152"
   ],
   "latency_us": 8705.023765563965,
   "check": null
  }
 ],
 "prev_iter_01_replay_us": [
  8598.7,
  2764.9,
  8500.3
 ],
 "baseline_plain_us": [
  8762.8,
  2815.9,
  8961.2
 ]
}
```

(diff.patch: 117 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/attention/kda_conv_l2norm.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/kda_conv_l2norm.py)
