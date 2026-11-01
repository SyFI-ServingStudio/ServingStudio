# KDA_SHAPES_CLAUDE trial 3: agent iterations

## agent log.md

# Kimi-K3 KDA decode layer — optimisation log (B200, CUDA graph)
Scored point 1,1048576. Identical fixed-flag command for all before/after numbers (run.sh / rep.sh);
--split/--profile-kernels only for diagnostics. Golden captured from the start tree (golden_B*_L*.pt).

| iter | change | 1x1M med (samples) | 16x64k | 128x8k | 32x8k | 1x8k | CHECK |
|---|---|---|---|---|---|---|---|
| 00 | baseline | 111.8 / 107.8 (bimodal 103.8..114.0) | 167.3 | 376.1 | 230.7 | 108-114 | golden |
| 01 | attn_res direct kernel, T<=128 | 103.8 | 165.2 | 373.1 | 228.7 | 105.7 | pass; 128 rel 0.0189 (risky) -> dropped |
| 02 | same, T<=32 | 103.9 | 165.2 | 374.3 | 228.7 | 104.0 | pass; 128 bit-exact |

Probes (iter_02, knob.json, not kept): L2 prefetch off -> worse (105.9); K-prefix 1/2, 1/4 -> noise.
Final tree edits: kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, kernels/ops/kimi_k3/attn_res.py.
Final replay (all points): 1x1M 101.7 | 16x64k 163.2 | 128x8k 374.2 | 32x8k 226.3 | 1x8k 105.8 — all CHECK pass (rel 0 / 0.0152 / 0 / 0.0120 / 0)

## iter_00
### hypothesis.md

# iter_00 hypothesis
Baseline only; no edit. Prior trials consulted for the next steps:
- kda_b512_claude_3: an attn-res fused-add rewrite failed CHECK because the residual+addend add was rounded
  differently (fused into the fp32 accumulation). Lesson: keep the exact bf16 add.rn + round of the prefix.
- mla_b512_claude_3 / mla_prefill_claude_1: attn-res TMA tuning (chunk rows / occupancy) gave gains at large T
  only; at decode T the TMA pipeline's add/fence/bulk-reload/TMEM phases are fixed latency.
=> next: a decode-sized (one CTA per token, register-direct) attn_res kernel.

### analysis.md

# iter_00 — baseline (start tree, no edits)

## Measurement (fixed flags, run.sh plain / rep.sh)
- 1x1M: repeat A/B of 6 samples per process, two processes:
  [103.8 107.8 111.8 105.8 114.0 114.0] and [103.8 107.8 106.1 103.8 114.0 114.0] -> median 107.8-111.8, bimodal 103.8 / 114.0
- 16x64k 167.3 | 128x8k 376.1 | 32x8k 230.7-230.8 | 1x8k 108-114

## VibeSim (run=k3_kda, fixed pristine prediction, 783 us un-overlapped sum)
analyze?level=operator: mxfp4_fused_moe 412 (52.7%), merged_front 120, qkvbfg_a_proj 52, shared_down 50,
kda_recurrent 42.5, latent_up 33, o_proj 15, conv 12, norms ~9.6 each.
optimality?scope=iter: largest headroom r0-r5 is mxfp4_fused_moe (r0/r5 43x, but that is the pristine
unfused model; the start tree already runs the flashinfer trtllm MXFP4 bmm path), then qkvbfg (3.0x).

## Map to the source / critical path (nsys --cuda-graph-trace=node, B=1, tl.py)
The step is ~104 us on one main stream (st141) plus two alt streams (l2_prefetch on st142, qkvg TGV on st143).
Serial main-stream nodes per step:
  elementwise 0.9 -> attn_res_fused_add (5.3) -> tiny_n/tiny_k -> [wait qkvg TGV 16 us] -> kda_decode 8.5
  -> o_proj TGV 4.6 -> attn_res_fused_add (6.1) -> front TGV 10 -> router 5.5 -> quant 2.7 -> routing 2.7
  -> MXFP4 bmm1 23.7 -> bmm2 9.6 -> finalize 7.0 -> flashinfer 4.0 -> latent TGV 10.8 -> add3 3.1
The two attention-residual aggregations (srt/layers/attn_residual.py -> kernels/ops/kimi_k3/attn_res.py
-> attn_res_fused_add_kernel<Trait<7168,1,4,200>>) are fully serial (nothing overlaps their front),
~11.8 us combined, for moving 2x(2 rows x 14 KB) per token: pure latency, not bandwidth. They are the one
node on the critical path that is local and whose ideal cost is ~1-2 us.

### result.json

```
{"iter": 0, "change": "baseline", "latency_us": {"1,1048576": {"median": 111.8, "samples": [103.8,107.8,111.8,105.8,114.0,114.0], "samples_run2": [103.8,107.8,106.1,103.8,114.0,114.0]},
 "16,65536": 167.3, "128,8192": 376.1, "32,8192": 230.7, "1,8192": 108.0},
 "check": "golden captured from this tree (golden_B*_L*.pt)", "kept": true}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis
At decode sizes the TMA attn_res kernel is latency bound (serial add -> fence -> bulk reload -> TMEM phases),
per iter_00 analysis. A one-CTA-per-token kernel with kDim/8 = 896 threads holding one 16-B vector per row in
registers needs one global read and one write per row and two block reductions: expected ~1.3 us/call saved,
~2.6 us/step.
Prior trials: kda_b512_claude_3 (fused-add rounding broke CHECK) -> keep an explicit add.rn.f32 then
__float22bfloat162_rn for prefix_out, bit-identical to the TMA path; mla_b512_claude_3 / mla_prefill_claude_1
(TMA tuning helps only at large T) -> leave the TMA path for T above the threshold.
Threshold tried: T <= 128.

### analysis.md

# iter_01 — analysis
Target node: attention-residual aggregation (attn_res_fused_add_kernel, 2x per layer, serial on the main stream).
nsys (iter_01/nsys_b1): attn_res_direct_kernel 4.0 us at the layer start (was 5.3) and 6.1 us before the MoE
(the second one overlaps the start of the l2_prefetch of the MoE weights; was ~6.1-6.7); step ~100.9 us vs ~104.
Isolated microbench (work/mb_attnres.py, CUDA graph, 20 calls/graph): TMA 4.10 -> direct 2.76 us at T=1,
4.2 -> 2.8 at T=16/32, 4.42 -> 3.07 at T=128; direct vs TMA max output diff <= 1 bf16 ulp, prefix bit-exact.
Replay: every point passes, but 128x8k max_rel_err = 0.0189 (limit 0.02): the 1-ulp shifts in the attn_res
output propagate through the MoE at B=128.

### result.json

```
{"iter": 1, "change": "attn_res_direct_kernel for T<=128",
 "latency_us": {"1,1048576": {"median": 103.8, "samples": [103.7,103.8,104.0,103.8,103.8,105.8]}, "16,65536": 165.2, "128,8192": 373.1, "32,8192": 228.7, "1,8192": 105.7},
 "replay_check": {"1,1048576": [true, 0.0], "16,65536": [true, 0.0152], "128,8192": [true, 0.0189], "32,8192": [true, 0.0120], "1,8192": [true, 0.0]},
 "kept": false, "reason": "128x8k rel_err 0.0189 too close to 0.02 for a 0.8% gain at that point; superseded by iter_02"}
```

(diff.patch: 290 lines, files: work/t128/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, work/t128/kernels/ops/kimi_k3/attn_res.py)

## iter_02
### hypothesis.md

# iter_02 hypothesis
iter_01 showed the direct kernel is ~1.3 us/call faster and within 1 bf16 ulp of the TMA kernel, but at
B=128 the shift reaches rel_err 0.0189 (limit 0.02) for a gain of only ~0.8% at that point. Capping the
direct path at T<=32 keeps the scored point (B=1, bit-exact) and 16/32 gains, and makes 128x8k bit-identical
to the original. Prior trials: kda_b512_claude_3 (rounding of the fused add), mla_b512_claude_3 /
mla_prefill_claude_1 (TMA path only pays at large T), iter_01 (this run).

### analysis.md

# iter_02 — analysis
profile.json (split/profile-kernels, diagnostic): attn_res per call 6.74 -> 5.58 us at 1x1M, 5.94 -> 4.87 us at
16x64k (direct kernel); 128x8k keeps attn_res_fused_add_kernel (6.71 vs 6.68, unchanged path).
Numerics: with the direct path capped at T<=32, 128x8k is bit-exact (rel_err 0.0), B=1 points bit-exact,
16x64k 0.0152 / 32x8k 0.0120 (from 1-ulp bf16 output shifts through the MoE), all well under 0.02.
Other probes (knob.json, diagnostic only; knob removed afterwards, defaults unchanged):
- layer-start L2 prefetch off: 1x1M median 105.9 (vs 103.9 on) -> the prefetch is useful, kept.
- K-prefix 1/2 or 1/4 instead of 1/3: 103.8 / 104.0 -> noise, kept 1/3.
Remaining critical path at B=1 (nsys): router -> quant -> routing -> MXFP4 bmm1 (23.7, contended by the
concurrent shared gate_up TGV on the alt stream) -> bmm2 -> finalize -> flashinfer -> latent TGV -> add3.
Not attempted in this run: re-prioritising the routed vs shared MoE chains.

### result.json

```
{"iter": 2, "change": "attn_res_direct_kernel for T<=32 (TMA path for larger T)",
 "latency_us": {"1,1048576": {"median": 103.9, "samples_run1": [103.8,105.7,103.9,103.8,103.8,105.9], "samples_run2": [103.8,103.9,103.9,103.8,103.8,105.9]},
   "16,65536": 165.2, "128,8192": 374.3, "32,8192": 228.7, "1,8192": 104.0},
 "baseline_same_session": {"1,1048576": {"median": 107.8, "samples": [103.8,107.8,106.1,103.8,114.0,114.0]}, "16,65536": 167.3, "128,8192": 376.1, "32,8192": 230.8, "1,8192": 114.0},
 "replay_check": {"1,1048576": [true, 0.0], "16,65536": [true, 0.0152], "128,8192": [true, 0.0], "32,8192": [true, 0.0120], "1,8192": [true, 0.0]},
 "kept": true}
```

(diff.patch: 291 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py)
