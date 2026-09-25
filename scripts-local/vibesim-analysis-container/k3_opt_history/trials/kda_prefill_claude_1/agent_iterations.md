# KDA_PREFILL_CLAUDE trial 1: agent iterations

## agent log.md

# opt log (latency_us: 1x16384pf / 1x16384pf49152 / 4x4096pf)
- iter_00: baseline 9128 / 9139 / 8786
- iter_01: chunk_delta_h h-kernel BV16/nw4/ns3 when grid<=2*SMs (kda_chunk_prefill) -> 8925 / 9000 / 8891; CHECK pass (bit-exact)
- iter_02: kda_backend drop int(query_start_loc[-1]) host sync -> 8860 / 9023 / 8809; CHECK pass (bit-exact)
- iter_03: attn_res pending residual add fused into TMA kernel (port of mla_b512_claude_3) -> 8896 / 8926 / 8601; CHECK pass (bit-exact); primary within noise, kernel time -60us
- iter_04: KDA prefill copy removal: strided l2norm q/k, strided o_norm gate, strided v in recompute_w_u -> 8584 / 8693 / 8594; CHECK pass (bit-exact)

## iter_00
### hypothesis.md

# iter_00
Baseline only; no edit. Prior trials: the seed tree already contains the accepted work from earlier rounds (seed_vs_head.diff):
- kda_fused_decode packed decode;
- TGV shared_down for m<=128;
- the alt-stream shared overlap.

None of it touches the chunked prefill h-kernel launch config or the prefill host sync.

### analysis.md

# iter_00 baseline analysis
VibeSim prediction handle `prediction=.:k3_kda_prefill` (log_dir /opt/vibesim/repo/logs/k3_kda_prefill); endpoints used: analyze(level=run_summary|operator|iteration), optimality(scope=iter).

## VibeSim findings
- run_summary: optimality_ratio 0.070; hardware_gap is 85% of the gap; batching is 2.4%, all of it in `unified.kda.attention.kda_chunk_prefill`, which has 1.16 ms batching of 2.89 ms predicted.
- optimality(scope=iter), ranked by headroom r0-r5:

  | Node | r0/r5 |
  |---|---|
  | merged_front_prefill | 3.36 |
  | mxfp4_fused_moe_prefill | 2.97 |
  | qkvbfg | 6.06 |
  | shared_down | 3.35 |
  | qkvbfg_a_proj | 3.17 |
  | **kda_chunk_prefill** | **37.9** (highest ratio) |
  | latent_up | 3.15 |
  | o_proj | 3.08 |
- The GEMM nodes sit at about 3x r5, a uniform cuBLAS ceiling that the tree already tunes (see prior trials). kda_chunk_prefill is the outlier: its batching bucket says the chunk pipeline does not fill the machine.

## Measured (profile_B1_L16384pf.json, trace_1.json)
Kernel total is about 8.7 ms against 9128 us latency.

| Kernel | Time (us) |
|---|---|
| merged MoE fp32 GEMM | 2325 |
| qkvbfg GEMM | 895 |
| shared_down (alt stream) | 896 |
| MoE bmm | 860 + 322 |
| **chunk h kernel** | **630** (grid 4x12 = 48 CTAs on 148 SMs) |
| latent_up | 489 |
| situ | 228 |
| o_proj | 217 |
| aten::add [16384,7168] | 2 x 104 |
| q/k/v .contiguous copies | 3 x 63 |
| gate reshape copy | 59 |

- GPU idle is about 390 us. Of that, 150-260 us comes after the host sync `int(query_start_loc[-1])` in KDABackend.forward_extend.

### result.json

```
{
 "iter": 0,
 "latency_us": {
  "1,16384,pf": 9127.9,
  "1,16384,pf49152": 9139.0,
  "4,4096,pf": 8786.0
 },
 "check": "n/a (golden captured from original tree)"
}
```

## iter_01
### hypothesis.md

# iter_01 hypothesis
The h kernel's V columns recur independently. Halving BV (32 to 16) therefore doubles the CTA count with no change to the per-column math. num_warps stays at 4, which keeps the dot-product shapes and reductions bit-identical.

I gate the change on `cdiv(V,32)*N*H <= 2*SMs`, because it regresses at N=16. It is also skipped when the SGLANG_GDN_CHUNK_H_* env knobs pin the tile.

Prior trials:
- It builds on the kda_11 / A7 Triton launch-tuning line from the decode rounds, which tuned the recurrent/decode kernels only.
- The chunked prefill h-kernel config had not been tried.
- Changing num_warps was ruled out because it changes the accumulation order.

Expected: about 150 us at the primary point and a smaller gain at 4x4096.

### analysis.md

# iter_01 analysis
VibeSim prediction handle `prediction=.:k3_kda_prefill` (log_dir /opt/vibesim/repo/logs/k3_kda_prefill); endpoints used: analyze(level=run_summary|operator|iteration), optimality(scope=iter).

- Target node: `unified.kda.attention.kda_chunk_prefill`. It has the highest r0/r5 (37.9), and its predicted batching loss is 1.16 ms.
- Mapped to source: `kernels/ops/attention/fla/chunk_delta_h.py::chunk_gated_delta_rule_fwd_h`. The grid is (cdiv(V,BV), N*H) = (4, 12) at BV=32, i.e. 48 CTAs, and the kernel runs a sequential loop over 256 chunks, so it is latency-bound.
- Microbenchmark (micro/bench_h.py), time in us by BV/num_warps/num_stages:

  | Case | 32/4/2 | 16/4/3 |
  |---|---|---|
  | N=1 | 609 | 395 |
  | N=4 | 173 | 134 |
  | N=16 | 104 | 139 |

  So the smaller tile helps only when the grid is small.
- Post-edit profile (profile.json, split-mode diagnostic): the h kernel is about 480 us at the primary point, down from 630-664 us.

### result.json

```
{
 "iter": 1,
 "edit": "chunk_delta_h BV=16/nw4/ns3 for small grids",
 "before_us": {
  "1,16384,pf": 9127.9,
  "1,16384,pf49152": 9139.0,
  "4,4096,pf": 8786.0
 },
 "after_us": {
  "1,16384,pf": 8925.0,
  "1,16384,pf49152": 9000.0,
  "4,4096,pf": 8891.0
 },
 "after_primary_single_point_runs": [
  8918,
  8920,
  8915
 ],
 "check": {
  "all_pass": true,
  "max_rel_err": 0.0
 },
 "note": "4x4096 within run-to-run noise (+-100us, A/B alternating runs inconclusive); microbench shows N=4 h kernel 173->134us"
}
```

(diff.patch: 57 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/chunk_delta_h.py)

## iter_02
### hypothesis.md

# iter_02 hypothesis
query_start_loc for EXTEND/MIXED is built in hybrid_linear_attn_backend.py as [extend_start_loc, extend_start_loc[-1] + extend_seq_lens[-1]], so its last element equals sum(extend_seq_lens_cpu). Reading that value from the host copy removes a device-to-host sync and keeps the launch queue ahead of the GPU. The numerics are unchanged.

Only EXTEND/MIXED with a host copy take the new path. TARGET_VERIFY returns earlier. SPLIT_PREFILL, DRAFT_EXTEND_V2 and a missing host copy fall back to the sync.

Builds on iter_01. No earlier trial touched host syncs on the prefill path; the decode rounds ran under CUDA graphs, where this sync does not exist.

Expected: 50-150 us.

### analysis.md

# iter_02 analysis
VibeSim prediction handle `prediction=.:k3_kda_prefill` (log_dir /opt/vibesim/repo/logs/k3_kda_prefill); endpoints used: analyze(level=run_summary|operator|iteration), optimality(scope=iter).

- The VibeSim run_summary batching bucket (2.4%) attributes the predicted non-hardware gap to kda_chunk_prefill. What remains in the measured timeline is launch-bound idle rather than kernel time.
- Gap analysis of iter_01/trace_1.json (gaps.py): 387 us total GPU idle.
  - 153 us around the `.item()`/`int()` sync. The CPU side shows cudaStreamSynchronize at 701 us.
  - 40 us and 80 us launch-bound gaps before cumsum and intra, because the CPU queue restarts from empty after the sync.
- Mapped to source: `srt/layers/attention/linear/kda_backend.py::forward_extend`, the line `logical_num_tokens = int(query_start_loc[-1])`.
- Post-edit profile: 40 launches instead of 41 (the sync's d2h copy is gone).

### result.json

```
{
 "iter": 2,
 "edit": "kda_backend forward_extend: host-side logical_num_tokens",
 "before_us": {
  "1,16384,pf": 8925.0,
  "1,16384,pf49152": 9000.0,
  "4,4096,pf": 8891.0
 },
 "after_us": {
  "1,16384,pf": 8859.7,
  "1,16384,pf49152": 9022.5,
  "4,4096,pf": 8809.4
 },
 "check": {
  "all_pass": true,
  "max_rel_err": 0.0
 }
}
```

(diff.patch: 80 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/chunk_delta_h.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/linear/kda_backend.py)

## iter_03
### hypothesis.md

# iter_03 hypothesis
Fuse the pending residual add into the attn-res TMA kernel. The producer bulk-copies the addend row into one extra ring slot row, and the consumers add in fp32 with one RNE rounding. That is bit-exact with torch's bf16 add. The sum is stored to prefix_out, which saves a full read of the prefix and one write per aggregation.

Builds on prior trial mla_b512_claude_3, which validated this exact change on the MLA layer (incremental.diff, "addend in the TMA ring"). It was not yet in this tree: grep for "addend" found nothing. The patch applied cleanly.

Expected: about 60-100 us.

### analysis.md

# iter_03 analysis
VibeSim prediction handle `prediction=.:k3_kda_prefill` (analyze level=run_summary/operator, optimality scope=iter; see iter_00/vs_*.json).

- VibeSim's run_summary gives the `fusion` gap bucket 0. The simulator models the attention-residual aggregation as already fused, so the measured trace carries work the prediction does not: two standalone bf16 adds, `aten::add [16384,7168]`, at 2 x 104 us.
- Each add reads 2 x 235 MB and writes 235 MB, and sits at the HBM limit.
- Mapped to source:
  - `srt/layers/attn_residual.py::_aggregate_fused_add` does `prefix = prefix_a + prefix_b`, then `_aggregate`, which launches the TMA kernel `attn_res_fused_tma_kernel` (241 us for 2 calls).
  - It is called from `KimiK3DecoderLayer` for both aggregation points of layer 5. Layer 5 is not a block-write layer, so the prefix is needed downstream.
- Post-edit profile (split-mode diagnostic):
  - The 2 aten::add kernels are gone.
  - attn_res with the kFuseAdd trait takes 392 us for 2 calls, so the net kernel time saved is about 60 us.
  - 38 launches instead of 40.

### result.json

```
{
 "iter": 3,
 "edit": "attn_res pending-add fused into TMA kernel (port of mla_b512_claude_3)",
 "before_us": {
  "1,16384,pf": 8859.7,
  "1,16384,pf49152": 9022.5,
  "4,4096,pf": 8809.4
 },
 "after_us": {
  "1,16384,pf": 8895.5,
  "1,16384,pf49152": 8925.5,
  "4,4096,pf": 8600.5
 },
 "after_primary_single_point_runs": [
  8850.3,
  8930.4,
  8862.8
 ],
 "check": {
  "all_pass": true,
  "max_rel_err": 0.0
 },
 "note": "primary within noise (+-50us); kernel-time delta -60us per profile"
}
```

(diff.patch: 401 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/chunk_delta_h.py, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/linear/kda_backend.py, /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py)

## iter_04
### hypothesis.md

# iter_04 hypothesis
Read the strided views in place and keep every reduction tile unchanged, so the results are bit-identical.
- **(a) l2norm:** a strided-heads variant of l2norm_fwd_kernel. It uses the same BT=16, BD, num_warps=8, num_stages=3 and the same [BT, BD] tile, so the row-sum reduction is identical. It addresses rows (t, h) at t*stride + h*D. Microbenchmark: bit-exact at T in {1, 7, 100, 2048, 4096, 16384}, 78 -> 24 us per tensor.
- **(b) gated norm:** `layer_norm_gated_fwd_kernel` gets a constexpr G_HEADS path that loads the gate from the [T/H, H, D] view. Only the elementwise gate load changes. Microbenchmark: bit-exact for sigmoid and silu, 89 -> 35 us.
- **(c) recompute_w_u:** `_recompute_w_u_fwd_kernel` takes a runtime `stride_v`, and u is allocated packed. o gets its own buffer when v is a view. The small-grid fused path keeps `.contiguous()`.

Builds on iter_01, which established that tile and num_warps must stay fixed for bit-exactness. Prior MLA trials removed copies on the MLA q path; the KDA prefill copies were not tried. The GDN helper `gdn_prefill_qkv_prepare` materializes the copies instead, so it was ruled out.

Expected: about 200 us.

### analysis.md

# iter_04 analysis
VibeSim prediction handle `prediction=.:k3_kda_prefill` (analyze level=run_summary/operator, optimality scope=iter; see iter_00/vs_*.json).

- VibeSim node `unified.kda.attention.kda_chunk_prefill` has r0/r5 37.9, the worst ratio in the layer. Its r5 counts only the chunk-kernel traffic, but the measured node also contains four layout copies that the model semantics do not require (the "unnecessary work" reading).
- Measured (iter_03 profile):

  | Copy | Time |
  |---|---|
  | `direct_copy_kernel` for q/k/v `.contiguous()` | 3 x 63 us |
  | o_norm gate `g.reshape` | 59 us |
  | total | 272 us |

- These tensors are head-packed, token-strided views: q/k/v are splits of the packed conv output, and the gate is a split of the qkvbfg GEMM output.
- Mapped to source:
  - `kernels/ops/attention/fla/kda.py::chunk_kda`: `l2norm_fwd(q.contiguous())`, `l2norm_fwd(k.contiguous())`, `v.contiguous()`.
  - `kernels/ops/attention/fla/fused_norm_gate.py::LayerNormGatedFunction.forward`: `g.reshape(-1, D)`.
- Post-edit profile:
  - All 4 copies are gone; 34 launches.
  - Two `l2norm_fwd_strided_heads_kernel` calls take 42 us in total.
  - The gated norm is unchanged at 33 us.
  - `_recompute_w_u_fwd_kernel` goes from 94 to 91 us.

### result.json

```
{
 "iter": 4,
 "edit": "strided q/k l2norm, strided o_norm gate, strided v in recompute_w_u",
 "before_us": {
  "1,16384,pf": 8895.5,
  "1,16384,pf49152": 8925.5,
  "4,4096,pf": 8600.5
 },
 "after_us": {
  "1,16384,pf": 8584.3,
  "1,16384,pf49152": 8692.9,
  "4,4096,pf": 8594.2
 },
 "check": {
  "all_pass": true,
  "max_rel_err": 0.0
 }
}
```

(diff.patch: 672 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/chunk_delta_h.py, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/fused_norm_gate.py, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/kda.py, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/l2norm.py, /sgl-works)
