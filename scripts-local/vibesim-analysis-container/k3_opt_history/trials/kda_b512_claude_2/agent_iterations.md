# KDA_B512_CLAUDE trial 2: agent iterations

## agent log.md

# opt_run log (Kimi-K3 KDA layer, B200; points 512/256/128 x 8192; plain /tmp/run.sh)
- iter_00: baseline 601.2 / 472.5 / 376.7 us; VibeSim top: mxfp4_fused_moe (18.3x), kda_recurrent_decode (4.4x). MoE live-ids tactic probe <1% -> rejected, reverted.
- iter_01: KDA fused decode prologue load hoist (kda_fused_decode.cuh): 601.2->598.4 / 472.5->471.3 / 376.7->376.2; CHECK pass (bit-exact vs start tree), kernel 80.4->78.4 us.
- iter_02: KDA quad-row split warp reduction (kda_fused_decode.cuh): 598.4->597.3 / 471.3->469.3 / 376.2->374.2; CHECK pass, kernel 78.4->76.6 us.

## iter_00
### hypothesis.md

# iter_00 hypothesis (exploration, no net edit)
Node probed: unified.kda.moe.mxfp4_fused_moe (VibeSim #1).
Built on kda_b512_claude_1 iter_01 (one-shot tactic autotune, already in tree). Tried a live-ids tuning hook
(profile tactics with the real local routing instead of FlashInfer's synthetic ids over 896 global experts).
Real-ids sweep: best tactic ~300 us vs the chosen ~303 us for gemm1+gemm2 -> <1% of the node, below the judge bar.
Rejected and reverted (flashinfer_trtllm.py back to start tree). Also ruled out (history): kda_b512_claude_1 iter_03
(side-stream shared gate_up under the MoE slows the MoE). GEMM kernels are at 1.24-1.37 PF/s at m=512 vs 1.61 peak
achieved at 8192^3 -> small headroom with numerics risk (margin 0.0166 vs 0.02). Next node: kda_recurrent_decode.

### analysis.md

# iter_00 — baseline (tree as handed over; includes prior accepted levers)
Plain fixed command (/tmp/run.sh), graph latency_us: B512 601.2, B256 472.5, B128 376.7.
CHECK vs /tmp/golden_orig.pt: pass at all points (max_rel_err 0.0166 / 0.0150 / 0.0126, state_ok).

VibeSim: workspace-info -> simulate(prediction=.:k3_kda_b512) -> analyze(operator) -> optimality(scope=iter) -> kernels.
optimality (R0/R5): mxfp4_fused_moe 18.3x (0.91 ms headroom), kda_recurrent_decode 4.4x (0.155 ms), merged_front 1.67x,
qkvbfg 3.3x, shared_down 1.78x. (The prediction is a fixed model of the pre-history tree; used for ranking only.)

Measured B512 graph timeline (nsys, ~597 us): MXFP4 MoE gemm1 216 + gemm2 99 us (1.85 GB weights at ~6 TB/s,
near the HBM floor), KDA fused decode 80 us (402 MB state traffic, ~5 TB/s vs 6.3 TB/s copy floor -> ~63 us),
merged_front fp32 GEMM 77 us, qkvg GEMM 35 us, shared path overlapped on alt stream.

### result.json

```
{"iter": 0, "kind": "baseline", "latency_us": {"512,8192": 601.2, "256,8192": 472.5, "128,8192": 376.7},
 "check": {"512,8192": {"max_rel_err": 0.016598, "pass": true}, "256,8192": {"max_rel_err": 0.015019, "pass": true}, "128,8192": {"max_rel_err": 0.012605, "pass": true}},
 "rejected_probe": "MoE live-ids tactic hook: <1% node gain, reverted"}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis — KDA fused decode prologue: one memory round trip instead of three
Node: unified.kda.attention.kda_recurrent_decode (VibeSim #2, R0/R5 4.4) = kda_decode_fusion_many_heads_kernel
(kernels/jit/csrc/attention/kda_fused_decode.cuh), 80.4 us in the B512 graph.

Evidence (ncu, isolated microbench /workspace/opt_run/kda_bench.py, same shapes, CUDA graph):
issue slots 65%, DRAM 60% -> latency-bound. Warp-stall sampling: 43% of samples in the prologue
(before the state loop); SASS shows three serial global-load waits: q/k conv loads, then v conv loads,
then gate/beta/onorm loads. The compiler cannot hoist the later loads above the conv-state stores because
`cs_qk = is_q ? cs_q : cs_k` is a runtime pointer select (restrict lost).
Change: in the K3 configuration (kUpdateConvState && kApplyOnorm && kPreloadOnormParams) load all prologue
inputs into registers first, then compute + store. Arithmetic (same FMA order, same conv_silu/bf16 rounding)
is unchanged -> bit-exact output, state and conv state (microbench torch.equal on all 3).

Built on / ruled out (this run, microbench B512 baseline 80.9 us):
- issue all 4 TMA chunks up front: 83.0 us (worse); 1 up front: 87.1; 3 up front: 81.2 (neutral) -> keep 2.
- TMA bulk store of the updated state from smem (per-warp 1 KB: 84.6 us; per-block 32 KB at end: 83.6) -> worse.
- also hoisting slot-independent loads above the ssm_state_indices load: 79.0 vs 78.0 (worse, reverted).
- history: kda_23/kda_25 + kda_b512_claude_1 iter_02 (3-stage TMA, launch bounds 6) ruled out.
Microbench: 80.9 -> 78.0 us (B512), 43.5 -> 40.1 (B256), 20.5 -> 20.4 (B128).

### analysis.md

# iter_01 analysis — prologue load hoist in KDA fused decode
Node: unified.kda.attention.kda_recurrent_decode (VibeSim #2, R0/R5 4.4) = kda_decode_fusion_many_heads_kernel
(kernels/jit/csrc/attention/kda_fused_decode.cuh), 80.4 us in the B512 graph.

Evidence (ncu, isolated microbench /workspace/opt_run/kda_bench.py, same shapes, CUDA graph):
issue slots 65%, DRAM 60% -> latency-bound. Warp-stall sampling: 43% of samples in the prologue
(before the state loop); SASS shows three serial global-load waits: q/k conv loads, then v conv loads,
then gate/beta/onorm loads. The compiler cannot hoist the later loads above the conv-state stores because
`cs_qk = is_q ? cs_q : cs_k` is a runtime pointer select (restrict lost).
Change: in the K3 configuration (kUpdateConvState && kApplyOnorm && kPreloadOnormParams) load all prologue
inputs into registers first, then compute + store. Arithmetic (same FMA order, same conv_silu/bf16 rounding)
is unchanged -> bit-exact output, state and conv state (microbench torch.equal on all 3).

Built on / ruled out (this run, microbench B512 baseline 80.9 us):
- issue all 4 TMA chunks up front: 83.0 us (worse); 1 up front: 87.1; 3 up front: 81.2 (neutral) -> keep 2.
- TMA bulk store of the updated state from smem (per-warp 1 KB: 84.6 us; per-block 32 KB at end: 83.6) -> worse.
- also hoisting slot-independent loads above the ssm_state_indices load: 79.0 vs 78.0 (worse, reverted).
- history: kda_23/kda_25 + kda_b512_claude_1 iter_02 (3-stage TMA, launch bounds 6) ruled out.
Microbench: 80.9 -> 78.0 us (B512), 43.5 -> 40.1 (B256), 20.5 -> 20.4 (B128).

Post-change profile (profile.json, diagnostic --profile-kernels): KDA kernel B512 80.4 -> 78.4 us, B256 43.0 -> 41.0,
B128 24.5 -> 23.8. ncu after: prologue share 43% -> 37%, one remaining load wait (plus ssm_state_indices).
Remaining hot spot: state loop is shuffle-latency bound (short_scoreboard 18% of samples on SHFL.BFLY chains).

### result.json

```
{"iter": 1, "node": "unified.kda.attention.kda_recurrent_decode", "files": ["python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh"],
 "before_us": {"512,8192": 601.2, "256,8192": 472.5, "128,8192": 376.7},
 "after_us": {"512,8192": 598.4, "256,8192": 471.3, "128,8192": 376.2},
 "check": {"512,8192": {"max_rel_err": 0.016598, "state_ok": true, "pass": true}, "256,8192": {"max_rel_err": 0.015019, "state_ok": true, "pass": true}, "128,8192": {"max_rel_err": 0.012605, "state_ok": true, "pass": true}},
 "smoke": "LAYER_SMOKE_OK", "kernel_us_b512": {"before": 80.4, "after": 78.4}, "kept": true}
```

(diff.patch: 331 lines, files: kda_hoist1.cuh)

## iter_02
### hypothesis.md

# iter_02 hypothesis — 4-row split warp reduction in the KDA state loop
Node: unified.kda.attention.kda_recurrent_decode (still VibeSim #2; 78.4 us after iter_01).
Evidence: iter_01 ncu — the chunk loop holds 55% of stall samples, dominated by short_scoreboard on
SHFL.BFLY chains: per chunk each warp reduces 4 rows as 2 pairs (dot_hk: 5+1 shuffles, dot_hq: 5), 22 SHFL.
Change: warp_reduce_sum_quad_split(a,b,c,d) — the same xor 16/8/4/2/1 butterfly tree, but each level halves
the values a lane carries (16: 2 shfl, 8: 1, 4/2/1: 1 each), totals land in lanes 0/8/16/24; dot_hk totals
broadcast with 4 shfl.idx. 16 SHFL per chunk instead of 22 and one chain of 4 rows instead of two pairs.
Every total is the identical sequence of fp32 adds (fp add is commutative; same partner at every level),
so output/state stay bit-exact (verified torch.equal in the microbench).
Built on iter_01. Ruled out in this iteration: launch bounds 6 blocks/SM for bf16 (40 regs, spills: 88.5 us).

### analysis.md

# iter_02 analysis — quad-row reduction
Node: unified.kda.attention.kda_recurrent_decode (still VibeSim #2; 78.4 us after iter_01).
Evidence: iter_01 ncu — the chunk loop holds 55% of stall samples, dominated by short_scoreboard on
SHFL.BFLY chains: per chunk each warp reduces 4 rows as 2 pairs (dot_hk: 5+1 shuffles, dot_hq: 5), 22 SHFL.
Change: warp_reduce_sum_quad_split(a,b,c,d) — the same xor 16/8/4/2/1 butterfly tree, but each level halves
the values a lane carries (16: 2 shfl, 8: 1, 4/2/1: 1 each), totals land in lanes 0/8/16/24; dot_hk totals
broadcast with 4 shfl.idx. 16 SHFL per chunk instead of 22 and one chain of 4 rows instead of two pairs.
Every total is the identical sequence of fp32 adds (fp add is commutative; same partner at every level),
so output/state stay bit-exact (verified torch.equal in the microbench).
Built on iter_01. Ruled out in this iteration: launch bounds 6 blocks/SM for bf16 (40 regs, spills: 88.5 us).

Microbench: B512 78.0 -> 77.8, B256 40.1 -> 39.4, B128 20.4 -> 18.7 us. 48 regs, 0 local.
Graph profile: KDA kernel B512 78.4 -> 76.6, B256 41.0 -> 39.3, B128 23.8 -> 22.0 us.

### result.json

```
{"iter": 2, "node": "unified.kda.attention.kda_recurrent_decode", "files": ["python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh"],
 "before_us": {"512,8192": 598.4, "256,8192": 471.3, "128,8192": 376.2},
 "after_us": {"512,8192": 597.3, "256,8192": 469.3, "128,8192": 374.2},
 "after_runs": {"512,8192": [597.3, 597.3], "256,8192": [469.4, 469.2], "128,8192": [374.2, 374.1]},
 "check": {"512,8192": {"max_rel_err": 0.016598, "state_ok": true, "pass": true}, "256,8192": {"max_rel_err": 0.015019, "state_ok": true, "pass": true}, "128,8192": {"max_rel_err": 0.012605, "state_ok": true, "pass": true}},
 "smoke": "LAYER_SMOKE_OK", "kernel_us_b512": {"before": 78.4, "after": 76.6}, "kept": true}
```

(diff.patch: 86 lines, files: kda_quad.cuh)
