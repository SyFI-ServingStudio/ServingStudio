# MLA_PREFILL_CLAUDE trial 1: agent iterations

## agent log.md

# opt_run log
- iter_00: baseline. pf49152 ~12.0-12.2 ms, pf 8.35-8.59 ms, 4x4096 7.9-7.97 ms (eager; ~2% run-to-run noise). VibeSim: front GEMM / attention subtree / MoE top headroom; attention K/V packing = ~800us of avoidable elementwise.
- iter_01: TRT-LLM prefill fp8 K/V pack (chunk+prefix, Triton mla_kv_pack_quantize_fp8) + Q cast once. A/B 3 rounds: pf49152 12096->11609 (+4.0%), pf 8541->8329 (+2.5%), 4x4096 8049->7915 (+1.7%); CHECK bit-exact (err 0) all points.
- iter_02: fold pending residual add (prefix_a+prefix_b) into attn_res TMA kernel (producer loads addend row, consumer fp32-add+RNE, writes prefix), nvb=1 fused-add config (2,1,200). kernel time 418->361us; A/B vs iter_01: pf49152 11597->11575 (+0.19%), pf 8399->8297 (+1.2%), 4x4096 7882->7915 (-0.4%, noise; prelim +2.5%); CHECK bit-exact all points.
- null A/B (identical trees, 3 rounds): pf49152 -0.15%, pf +1.99%, 4x4096 -0.93% -> harness noise/position bias; secondary-point deltas < ~2% are not significant.

## iter_00
### analysis.md

# iter_00 — baseline analysis
Prediction: `.:k3_mla_prefill` (prediction_id p_9318587543334f74b338aa9dcb102cad; the server's fixed before-state; the
`p_...` id itself does not resolve for analyze/optimality, the `.:<run_name>` handle does). Sums the 3 cases (47.0 ms kernel time).

run_summary: optimality 0.085, hardware_gap 82%, imbalance 9%, batching 0.5%.
optimality (iter scope, sorted by R0-R5 headroom), r6 = null everywhere (no necessary-share given):
| node | R0 ms | R5 ms | R0/R5 |
|---|---|---|---|
| moe.merged_front_prefill (fp32-out GEMM 16384x7168x15984) | 16.8 | 5.0 | 3.4 |
| attention + output gate (subtree) | 7.3 | 1.14 | 6.4 |
| mxfp4_fused_moe_prefill | 8.6 | 2.9 | 3.0 |
| shared_down_prefill | 6.4 | 1.9 | 3.3 |
| latent_up_prefill | 3.5 | 1.1 | 3.2 |
| mla_prefill_attention_prefix | 3.5 | 1.37 | 2.6 |
| mla_cache_append | 0.68 | 0.01 | 63.7 |
| mla_prefix_gather | 0.077 | 0.011 | 7.2 |

Measured (torch profiler, pf49152 point, 12185 us eager): prefix FMHA 3502, front GEMM 2289, 6 nvjet GEMMs 2067,
MoE bmm 880+328, causal FMHA 681, and ~800 us of aten copy / fp8-convert elementwise kernels in the attention
data path (strided k_nope/v copies of the 49152-token prefix: 172+68+61+167 us; chunk: 58+24+24+22+58+26 us).

GEMMs run at ~73% of the bf16 dense peak (compute-bound, closed nvjet kernels) and the FMHA is bounded by MUFU exp
(16384*49152*12 exps ~ 2.1 ms floor) -> little code-level headroom. The attention subtree has the largest R0/R5 (6.4)
among big nodes, and the part of it that is NOT a kernel-efficiency problem is the elementwise K/V packing:
12 heads is not a power of two, so `_concat_and_cast_mha_k` falls to aten slice-copies, and the TRT-LLM backend
has no `pack_prefix_chunk_kv`, so each prefix K/V is built bf16 then cast to fp8 by strided non-vectorized kernels.
Target for iter_01: that packing (maps to R0 >> R5 on the attention subtree / mla_cache_append-like memory ops).

## iter_01
### hypothesis.md

# iter_01 hypothesis — fp8 K/V packing for the TRT-LLM ragged prefill (chunk + prefix)
Node: attention subtree (VibeSim R0/R5 = 6.4, the largest among big nodes); in the profile ~800 us/step of
aten slice-copies and strided bf16->fp8 casts building K=[k_nope|k_pe] and V for the FMHA
(12 heads is not a power of two, so `_concat_and_cast_mha_k` falls back to two aten copies; the TRT-LLM backend
has no `pack_prefix_chunk_kv`, so the 49,152-token prefix K/V goes bf16-concat -> strided `.to(fp8)`; Q is cast
to fp8 once per ragged launch, i.e. twice).
Change: TRTLLMMLABackend gets `pack_extend_kv` (chunk) and `pack_prefix_chunk_kv` (prefix) that call the
existing Triton `mla_kv_pack_quantize_fp8` (one launch, unit scale), and `prepare_chunked_prefill_qkv` that casts Q once.
`forward_normal_prepare` uses the hook when the backend offers it.
Why identical: bf16 -> fp32 -> e4m3 RNE at unit scale is exactly what `.to(torch.float8_e4m3fn)` does for in-range
values (microbench: bit-identical on 49152/16384/4096-token shapes); the FMHA sees the same bytes. Scaled KV
checkpoints (k_scale != 1) keep the old path for the chunk; a pre-packed fp8 K/V is consumed at unit scale.
Prior trials: A8 (mla_25) touched only the decode KV-concat kernel; no prior trial touched the prefill K/V path
(grep of agent_iterations.md for pack_prefix_chunk_kv / concat_and_cast: none).

### analysis.md

# iter_01 — analysis (post-edit re-profile)
Prediction: `.:k3_mla_prefill` (VibeSim prediction is fixed to the before-state; simulate is idempotent, so the
post-edit analysis is measured: profile.json = torch-profiler kernel table at pf49152, 11208 us diagnostic run).

Findings (pf49152, per step, us): prefix FMHA 3334, front GEMM (fp32 out, 16384x7168x15984) 2295, six nvjet bf16
GEMMs 2056 (shared_down 873, latent_up 488, fused_qkv_a 324, g_proj 229, o_proj 208, q_b 73), MoE bmm1 874 /
bmm2 327, causal FMHA 652, situ 228, attn_res x2 212, **bf16 `prefix_a+prefix_b` adds x2 206**, add3 134,
fp8 K/V pack x2 101 (was ~800 us of aten copy/convert), finalize 68, create_chunked_prefix_cache_kv_indices 50,
quant 47, get_mla_kv_buffer 33, q fp8 convert 30, merge 27. Kernel sum ~= step time (GPU never idle).

GEMM microbench (/tmp/mb_gemm.py): front fp32-out 2332 vs bf16-out 2254 us (bf16 = numerics FAIL, mla_b512_2);
splitting gate_up (bf16) from router+latent (fp32) 1952+623 = worse; standalone nvjet GEMMs run 1.25-1.44 PF ->
no tactic headroom. FMHA prefix is MUFU-exp bound (~2.1 ms floor).
Next: the remaining avoidable memory passes — the 2x [16384,7168] bf16 pending-residual adds before attn_res.

### result.json

```
{
 "ab_3rounds_latency_us": {
  "orig": {
   "1,16384,pf49152": [
    12119.392395019531,
    12114.239692687988,
    12052.767753601074
   ],
   "1,16384,pf": [
    8560.640335083008,
    8556.639671325684,
    8504.38404083252
   ],
   "4,4096,pf": [
    8077.792167663574,
    8079.296112060547,
    7988.287925720215
   ]
  },
  "new": {
   "1,16384,pf49152": [
    11652.480125427246,
    11639.967918395996,
    11535.584449768066
   ],
   "1,16384,pf": [
    8317.119598388672,
    8398.847579956055,
    8270.879745483398
   ],
   "4,4096,pf": [
    7908.383846282959,
    7904.287815093994,
    7932.831764221191
   ]
  }
 },
 "replay": [
  {
   "max_abs_err": 0.0,
   "max_rel_err": 0.0,
   "mean_rel_err": 0.0,
   "nan": false,
   "rows": 16384,
   "rows_over_tol": 0,
   "frac_rows_over_tol": 0.0,
   "p99_row_rel_err": 0.0,
   "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)",
   "state_ok": true,
   "rel_err_max": 0.02,
   "pass": true
  },
  {
   "B": 1,
   "seq_len": 16384,
   "mixed": false,
   "tag": "pf49152",
   "prefill": true,
   "prefix_len": 49152,
   "num_tokens": 16384,
   "ok": true,
   "latency_us": 11561.280250549316,
   "latency_mode": "eager",
   "us_step": 11561.280250549316,
   "us_step_graph": null,
   "us_attn": null,
   "us_moe": null,
   "us_norms": null,
   "finite": true,
   "graph_finite": null,
   "attention_backend": "trtllm_mla",
   "attn_heads": 12,
   "seed": 0,
   "error": null,
   "point": [
    1,
    16384,
    "pf49152"
   ],
   "correctness": {
    "max_abs_err": 0.0,
    "max_rel_err": 0.0,
    "mean_rel_err": 0.0,
    "nan": false,
    "rows": 16384,
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
   "max_abs_err": 0.0,
   "max_rel_err": 0.0,
   "mean_rel_err": 0.0,
   "nan": false,
   "rows": 16384,
   "rows_over_tol": 0,
   "frac_rows_over_tol": 0.0,
   "p99_row_rel_err": 0.0,
   "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)",
   "state_ok": true,
   "rel_err_max": 0.02,
   "pass": true
  },
  {
   "B": 1,
   "seq_len": 16384,
   "mixed": false,
   "tag": "pf",
   "prefill": true,
   "prefix_len": 0,
   "num_tokens": 16384,
   "ok": true,
   "latency_us": 8283.32805633545,
   "latency_mode": "eager",
   "us_step": 8283.32805633545,
   "us_step_graph": null,
   "us_attn": null,
 
[... truncated]
```

(diff.patch: 122 lines, files: /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py, /sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py)

## iter_02
### hypothesis.md

# iter_02 hypothesis — fold the pending residual add into the attn_res TMA kernel
Node: unified.mla.attention.attn_res_prefill (both aggregation points: layer entry and post-o_proj).
Measured (iter_01/profile.json, trace): each aggregation runs `aten::add [16384,7168]+[16384,7168]` (~103 us,
3 x 235 MB at ~6.9 TB/s) and then attn_res_fused_tma re-reads that sum from HBM. VibeSim scores attn_res itself
at R0/R5 = 1.00 (bandwidth-bound, no in-kernel headroom) and has no node for the standalone add, so the only lever
is removing the extra memory pass across the kernel boundary: expected ~2 x 470 MB less traffic (~2 x 65 us ideal).
Change: the TMA producer also bulk-loads the addend row into a per-slot smem row with the prefix row (same mbarrier);
consumers compute bf16(float(a)+float(b)) (exactly torch's CUDAFunctor_add: fp32 add, one RNE rounding), use it for
score/mix, and store it as the materialized prefix (like the existing bank-row snapshot). `_aggregate_fused_add`
takes this path whenever the TMA kernel applies (H=7168, SM100).
Why identical: same fp32 add + rounding as the aten kernel; the rest of the kernel sees the same bf16 row.
Prior trials: the HIP path already folds this add (attn_res_hip); mla_27 tried a fused finalize+shared JIT (regressed,
different node); no trial touched fused_tma.cuh or the pending add (grep attn_res / prefix_b in trials: kda_9,
mla_b512_claude_1 mention attn_res only as a profile line).
Standalone check (/tmp/t_attnres.py): bit-identical out/prefix/bank for T in {37,300,1000,4096,16384}, nvb 1-8,
write_prefix on/off; add+kernel 259 us -> fused 228 us at T=16384, nvb=2.

### analysis.md

# iter_02 — analysis
Prediction: `.:k3_mla_prefill` (fixed before-state). analyze(level=operator): unified.mla.attention.attn_res_prefill
0.972 ms / 2.07% of 47.0 ms summed kernel time; optimality(scope=iter): attn_res_prefill r0/r5 = 1.00 (at the HBM
limit as a kernel), add3_prefill r0/r5 = 1.11. The standalone pending-residual `aten::add [16384,7168]` has no
VibeSim node -> taken from the measured profile (iter_01/profile.json: 2 x ~103 us).

Findings:
- The driver's MLA layer aggregates with nvb = 1 (kernel instance KimiK3AttnResTrait<7168,1,2,0>, occupancy 2).
- Fused kernel standalone at T=16384 (/tmp/t_attnres_cfg.py), nvb=1: add+kernel 215 us; fused with the default
  (2,2,0) config 211 us; fused with (2,1,200) 184 us -> the extra addend smem row makes 2 CTA/SM lose;
  added a fused-add tuning entry for nvb=1.
- In-layer (profile.json, pf49152): attn_res x2 = 361 us vs 212 + 206 (adds) = 418 us before -> -57 us of kernel time.
- Step-level A/B vs iter_01 (3 rounds, plain flags): pf49152 +0.19% (prelim run with the untuned config: +0.28%),
  i.e. smaller than the kernel saving; pf / 4x4096 within noise (+1.2% / -0.4%; prelim +1.7% / +2.5%).
- CHECK: bit-exact (max_rel_err 0.0, state_ok) on all 3 points.

### result.json

```
{
 "ab_3rounds_latency_us": {
  "orig": {
   "1,16384,pf49152": [
    11581.727981567383,
    11602.144241333008,
    11607.263565063477
   ],
   "1,16384,pf": [
    8398.91242980957,
    8398.143768310547,
    8400.927543640137
   ],
   "4,4096,pf": [
    7921.696186065674,
    7911.3922119140625,
    7812.064170837402
   ]
  },
  "new": {
   "1,16384,pf49152": [
    11580.639839172363,
    11580.672264099121,
    11562.335968017578
   ],
   "1,16384,pf": [
    8533.023834228516,
    8251.168251037598,
    8106.975555419922
   ],
   "4,4096,pf": [
    7935.0080490112305,
    7871.712207794189,
    7939.072132110596
   ]
  }
 },
 "summary": {
  "1,16384,pf49152": {
   "before_us": 11597.045262654623,
   "after_us": 11574.549357096354,
   "delta_pct": 0.19397963057634918
  },
  "1,16384,pf": {
   "before_us": 8399.327913920084,
   "after_us": 8297.055880228678,
   "delta_pct": 1.2176216328203153
  },
  "4,4096,pf": {
   "before_us": 7881.717522939046,
   "after_us": 7915.264129638672,
   "delta_pct": -0.4256255898792554
  }
 },
 "all_pass": true,
 "check_lines": [
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}"
 ]
}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16384, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 16384, "mixed": false, "tag": "pf49152", "prefill": true, "prefix_len": 49152, "num_tokens": 16384, "ok": true, "laten
[... truncated]
```

(diff.patch: 367 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py, /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py, /sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.)

## iter_03
### hypothesis.md

# iter_03 hypothesis — split create_chunked_prefix_cache_kv_indices across CTAs
Node: unified.mla.attention prefix gather metadata (VibeSim mla_prefix_gather r0 0.077 ms vs r5 0.011, r0/r5 = 7.2
in iter_00/opt.json — the largest ratio after mla_cache_append, which iter_01 addressed).
Measured (iter_02/profile.json): `create_chunked_prefix_cache_kv_indices` 52 us for a 49,152-index int32 copy
(~400 KB): grid = (batch_size,) = 1 CTA looping 96 x 512 serially -> latency-bound, not bandwidth-bound.
Change: optional grid axis 1 strides the per-request loop (program_id(1) / num_programs(1); a 1-D launch, e.g. the
prefill CUDA-graph runner, is unchanged); forward_batch_deepseek_mha_mixin launches (bs, min(cdiv(max_len,512),64))
from the host-side prefix_chunk_max_seq_lens (no sync).
Why identical: pure int32 copy of the same elements.
Prior trials: none touched kv_indices.py (grep create_chunked_prefix in trials: no hits).

### analysis.md

# iter_03 — analysis
Prediction `.:k3_mla_prefill` (fixed before-state): optimality(scope=iter) mla_prefix_gather r0/r5 = 7.2 (0.077 ms vs
0.011 ms, summed over the 3 cases; only pf49152 has a prefix).
Measured: kernel 52 us (1 CTA, 96 serial iterations) -> 3.2 us with 64 CTAs (profile.json, pf49152).
A/B vs iter_02 (3 rounds, plain flags): pf49152 11567 -> 11525 (+0.36%), pf 8336 -> 8306 (+0.37%, not on this
path -> noise), 4x4096 7846 -> 7915 (-0.88%, not on this path -> A/B ordering bias suspected; null A/B run in
/workspace/opt_run/null_ab). CHECK bit-exact all 3 points.
Next: get_mla_kv_buffer_kernel (34 us, one program per 576-byte row; multi-row variant 31 -> 14 us in microbench).

### result.json

```
{
 "ab_3rounds_latency_us": {
  "orig": {
   "1,16384,pf49152": [
    11543.71166229248,
    11593.91975402832,
    11562.335968017578
   ],
   "1,16384,pf": [
    8345.727920532227,
    8330.304145812988,
    8333.279609680176
   ],
   "4,4096,pf": [
    7852.928161621094,
    7847.040176391602,
    7837.152004241943
   ]
  },
  "new": {
   "1,16384,pf49152": [
    11533.408164978027,
    11514.20783996582,
    11527.423858642578
   ],
   "1,16384,pf": [
    8348.76823425293,
    8223.872184753418,
    8344.639778137207
   ],
   "4,4096,pf": [
    7876.768112182617,
    7930.816173553467,
    7936.09619140625
   ]
  }
 },
 "summary": {
  "1,16384,pf49152": {
   "before_us": 11566.655794779459,
   "after_us": 11525.013287862143,
   "delta_pct": 0.360022012033172
  },
  "1,16384,pf": {
   "before_us": 8336.437225341797,
   "after_us": 8305.760065714518,
   "delta_pct": 0.36798885180859114
  },
  "4,4096,pf": {
   "before_us": 7845.706780751546,
   "after_us": 7914.560159047444,
   "delta_pct": -0.8775930610206022
  }
 },
 "all_pass": true,
 "check_lines": [
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}",
  "CHECK {\"max_abs_err\": 0.0, \"max_rel_err\": 0.0, \"mean_rel_err\": 0.0, \"nan\": false, \"rows\": 16384, \"rows_over_tol\": 0, \"frac_rows_over_tol\": 0.0, \"p99_row_rel_err\": 0.0, \"rule\": \"prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)\", \"state_ok\": true, \"rel_err_max\": 0.02, \"pass\": true}"
 ]
}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "rows": 16384, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.0, "rule": "prefill_rowwise(frac_rows_over_tol<=0.005,p99<=tol,mean<=0.01)", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 16384, "mixed": false, "tag": "pf49152", "prefill": true, "prefix_len": 49152, "num_tokens": 16384, "ok": true, "latency_us":
[... truncated]
```

(diff.patch: 397 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py, /sgl-workspace/sglang/python/sglang/kernels/ops/kvcache/kv_indices.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py, /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py, /sgl-workspace/sglang/python/s)
