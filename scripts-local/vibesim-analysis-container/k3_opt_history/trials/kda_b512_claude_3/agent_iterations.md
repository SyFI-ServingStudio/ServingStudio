# KDA_B512_CLAUDE trial 3: agent iterations

## agent log.md

- iter_00 baseline: B512 601.4/600.4, B256 472.4/472.5, B128 376.3/377.2 us (pred p_1066e0f8b4d148de9d40979e8ca7f1e5)
- iter_01 KDA half-warp-row loop: REJECTED (bit-exact; spills @5CTA/SM 613.8us, 602.4us @4CTA/SM)
- iter_02 fuse prefix_a+prefix_b into attn_res TMA (producer TMA-stages addend): KEPT, B512 ~597.0 (-3.9), B256 468.2 (-4.3), B128 374.1 (-2.6), exact
- iter_03 capture routed before shared (graph keeps router on front stream): KEPT, B512 595.3 (-1.7), B256 466.7, B128 374.1, exact; side-stream prequant sub-trial rejected
- iter_04 port kda_b512_claude_2 KDA prologue hoist + quad-split reduction: KEPT, B512 592.1 (-3.2), B256 464.2, B128 371.1, exact

## iter_00
### analysis.md

# iter_00 — baseline (tree = accepted round kda_b512_claude_1)
VibeSim prediction: p_1066e0f8b4d148de9d40979e8ca7f1e5 (simulate?prediction=.:before, run k3_kda_b512)
Baseline plain flags: B512 601.4/600.4, B256 472.4/472.5, B128 376.3/377.2 us.
Operator shares (analyze): mxfp4_fused_moe 57%, kda_recurrent_decode 11.9%, merged_front 11.6%,
shared_down 4.6%, qkvbfg 4.5%, latent_up 2.9%. run_summary gap: batching (MoE weight amortization).
Findings:
- MoE GEMM1 ~215us @6.1TB/s, GEMM2 ~99us @6.6TB/s: near HBM roofline, tactic already tuned.
- KDA decode 80us: ncu DRAM 59.5%, latency-bound (long/short scoreboard), 5 CTA/SM, 8.3 waves.
- Two standalone torch adds (prefix_a+prefix_b, ~4.5us each) sit on the critical path right
  before each attn_res TMA kernel; the TMA kernel already streams the prefix row -> fusable, bit-exact.

## iter_01
### hypothesis.md

# iter_01 — KDA decode half-warp-row inner loop (node: kda_recurrent_decode)
Hypothesis: the xor-16 butterfly step becomes a local add if each half-warp owns a state row,
cutting shuffles. Prior: kda_7/kda_22 (bf16 port), kda_b512_claude_1 (vectorized bf16 ld/st, 5 CTA/SM).
Result: REJECTED. Bit-exact, but at 5 CTA/SM it spilled (STACK:72; kernel 94us; 613.8/478.6/380.3),
at 4 CTA/SM kernel 82.1 vs 80.2us (step 602.4). The kernel is latency-bound, not shuffle-bound.
Variant saved at /workspace/opt_run/kda_halfwarp_variant.cuh; tree restored.

### result.json

```
{"kept": false, "latency_us": {"B512": 602.4, "B256": null, "B128": null}, "note": "4CTA/SM variant; 5CTA/SM: 613.8/478.6/380.3", "check_pass": true, "max_rel_err": 0.0}
```

(diff.patch: 284 lines, files: /workspace/opt_run/kda_halfwarp_variant.cuh)

## iter_02
### hypothesis.md

# iter_02 — fold prefix_a + prefix_b into attn_res_fused_tma
Hypothesis: the standalone bf16 add (2 x ~4.5us @B512) can be fused: the TMA kernel already streams
the prefix row; load prefix_b alongside, add in fp32 with a single bf16 round (same as torch),
store the prefix from registers. Bit-exact. Expected ~-5..8us.
Prior trials: kda_b512_claude_1 (merged-front split slower -> don't touch GEMMs upstream of router);
TECHNIQUES B (fused finalize+shared regressed -> keep fusions off the MoE path); the HIP path
(_aggregate_hip) already folds this add, confirming semantics.

### analysis.md

# iter_02 analysis
Prediction p_1066e0f8b4d148de9d40979e8ca7f1e5; node: residual add feeding attn_res (x2 per layer:
attention side and MLP side). Profile (iter_00): at::native vectorized_elementwise add 9.5us total,
serial between o_proj / add3 and attn_res_fused_tma. attn_res at B512 runs nvb=1 config
(chunk_rows=2, occupancy=2, 296 CTAs, 1 chunk/token).
Stage 1 (consumer gmem load of addend): add removed, but attn_res 16.4 -> 26.4us (2 calls); the
per-token addend load serialized with the chunk wait (1 chunk/token) -> B512 601.3 (no gain).
Stage 2 (producer TMA-loads the addend into a double-buffered smem slot on the prefix chunk's
full barrier): B512 597.2/596.4/597.3.

### result.json

```
{"kept": true, "node": "residual add -> attn_res_fused_tma (x2)",
 "files": ["kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh", "kernels/ops/kimi_k3/attn_res.py", "srt/layers/attn_residual.py"],
 "before_us": {"B512": [601.4, 600.4], "B256": [472.4, 472.5], "B128": [376.3, 377.2]},
 "after_us": {"B512": [597.2, 596.4, 597.3], "B256": [468.3, 468.3, 468.1], "B128": [374.1, 374.1, 374.1]},
 "check": {"B512": {"pass": true, "max_rel_err": 0.0}, "B256": {"pass": true, "max_rel_err": 0.0}, "B128": {"pass": true, "max_rel_err": 0.0}},
 "smoke": "LAYER_SMOKE_OK"}
```

(diff.patch: 215 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py, /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py)

## iter_03
### hypothesis.md

# iter_03 — capture the routed branch before the shared branch
Hypothesis: capturing routed first keeps it on the front GEMM's graph stream -> router launches
immediately; shared (situ_and_mul -> shared_down) forks to alt. Bit-exact (scheduling only).
Prior: kda_b512_claude_1 added the side-stream shared overlap (keep overlap; only reorder); kda_b512_claude_2 (judged FAIL on noise) carried the same reorder.
Result: GEMM1 start 247.1 -> 240.6us; GEMM1 +4us from more shared_down overlap; net B512 -1.7us.
Sub-trial REJECTED: mxfp8 activation prequant on a side stream (published via route_quant_handoff
with a ready event). (a) on alt_stream: serialized ahead of situ_and_mul, shared_down launched
after GEMM1 and co-ran with GEMM2 (99 -> 126us): B512 599.4, B256 462.2. First attempt also NaN'd
at B512 (alt-stream-allocated x_q freed & reused; needed record_stream). (b) own 3rd stream:
router/quant/situ contend (10.2/10.9/13.3us): B512 595.3, B256 466.4 -> no gain, reverted.

### analysis.md

# iter_03 analysis
Prediction p_1066e0f8b4d148de9d40979e8ca7f1e5; node: MoE front -> routed chain (router/quant/routing
between merged_front GEMM and MoE GEMM1). nsys (iter_02 tree, B512): the graph mapped the shared
branch (captured first) onto the front GEMM's stream; the router landed on another stream and
started 3.5us after the front GEMM ended; front end -> GEMM1 start = 18.3us.
Ruled out first: add3 -> latent_up addmm(beta=1) epilogue. Microbench m=512: linear 19.9 + add3 2.6
vs addmm 22.9us (and 0.006 rel err); the 8.3us add3 in the timeline is mostly PDL early-launch wait.

### result.json

```
{"kept": true, "node": "moe front fork order (routed vs shared stream)",
 "files": ["srt/models/kimi_k3.py"],
 "before_us": {"B512": [597.2, 596.4, 597.3], "B256": [468.3, 468.3, 468.1], "B128": [374.1, 374.1, 374.1]},
 "after_us": {"B512": [595.3, 595.3, 595.3], "B256": [466.4, 467.3, 466.3], "B128": [374.1, 374.0, 374.1]},
 "check": {"B512": {"pass": true, "max_rel_err": 0.0}, "B256": {"pass": true, "max_rel_err": 0.0}, "B128": {"pass": true, "max_rel_err": 0.0}},
 "smoke": "LAYER_SMOKE_OK"}
```

(diff.patch: 242 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py, /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py, /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)

## iter_04
### hypothesis.md

# iter_04 — port kda_b512_claude_2's KDA kernel edits (prologue load hoist + quad-row split reduction)
Prior: kda_b512_claude_2 iter_01 (hoist all prologue global loads above the conv-state stores; the
runtime cs_q/cs_k pointer select blocked the compiler) and iter_02 (warp_reduce_sum_quad_split: same
butterfly partners, 16 SHFL/chunk instead of 22, 48 regs, no spill). Both bit-exact there; that trial
was judged FAIL only on noise (judge before-sigma 7.3us), not correctness. Applied its
trial_1->trial_2 incremental cuh diff (applies cleanly: our start tree = its trial-1 tree).
Contrast with iter_01 here: the quad split keeps 48 regs (my half-warp variant spilled).

### analysis.md

# iter_04 analysis
Prediction p_1066e0f8b4d148de9d40979e8ca7f1e5; node: kda_recurrent_decode (VibeSim #2, R0/R5 4.4)
= kda_decode_fusion_many_heads_kernel, 80.2us @B512 in this tree. ncu (iter_00): latency-bound
(DRAM 59.5%, long/short scoreboard stalls). iter_01 (half-warp rows) showed shuffle count alone
is not the lever when it costs registers.

### result.json

```
{"kept": true, "node": "kda_recurrent_decode", "files": ["kernels/jit/csrc/attention/kda_fused_decode.cuh"],
 "before_us": {"B512": [595.3, 595.3, 595.3], "B256": [466.4, 467.3, 466.3], "B128": [374.1, 374.0, 374.1]},
 "after_us": {"B512": [592.2, 592.2, 592.0], "B256": [464.2, 464.3, 464.2], "B128": [371.0, 370.1, 372.0]},
 "check": {"B512": {"pass": true, "max_rel_err": 0.0}, "B256": {"pass": true, "max_rel_err": 0.0}, "B128": {"pass": true, "max_rel_err": 0.0}},
 "smoke": "LAYER_SMOKE_OK"}
```

(diff.patch: 658 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh, /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/kimi_k3/attn_res.py, /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py, /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)
