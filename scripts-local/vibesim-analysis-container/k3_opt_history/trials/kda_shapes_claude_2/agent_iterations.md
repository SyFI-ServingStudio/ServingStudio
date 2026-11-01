# KDA_SHAPES_CLAUDE trial 2: agent iterations

## agent log.md

# Optimization log — Kimi-K3 KDA decode layer (B200)
| iter | change | 1@1M | 16@64k | 128@8k | 32@8k | 1@8k | CHECK vs orig | verdict |
|---|---|---|---|---|---|---|---|---|
| 00 | seed tree (baseline; same-job re-baseline) | 109.92 | 171.33 | 374.14 | 231.42 | 109.92 | pass (0/0/0.0126/0.0072/0) | baseline |
| 01 | L2 prefetch o_proj+routed-front on alt stream, 32 CTAs | 105.86 | 169.31 | 374.18 | 228.77 | 105.89 | pass (identical errs; bit-exact vs seed) | accepted |

## iter_00
### hypothesis.md

# iter_00 → iter_01 hypothesis
The MoE phase is HBM-bound and the late attention phase leaves HBM idle for about 14 µs. If we move bytes from the first
into the second (L2 prefetch of the next weights on a side stream), the MoE phase gets shorter by roughly bytes/BW.
Prefetch is side-effect-free, so it stays bit-exact vs the seed.

Prior trials: no L2 *weight* prefetch appears in /workspace/opt_history. The mla_b512_claude_3 iteration's
"bank prefetch" is PDL operand prefetch, not the same thing. Dead ends I avoided:
- kda_b512_claude_{1,2,3}: splitting the merged front with gate_up on the alt stream (the seed already has that for ≤32 tokens),
  deferring shared_event (no gain), the KDA half-warp variant (spills).
- kda_25: TGV tactic buckets (no effect).

### analysis.md

# iter_00 — baseline (seed tree) analysis

VibeSim: prediction `p_38ccb83a6c4e44298f45b2251bd4cf54` (run k3_kda, fixed before-state, 3 cases).
- `optimality?scope=iter`: mxfp4_fused_moe R0/R5 43.6 (52.7% of the modeled time), merged_front 1.32,
  qkvbfg_a_proj 1.54, qkvbfg_a_proj_bfa 16.7, shared_down 1.49, kda_recurrent_decode 5.2, latent_up 1.66.
- `analyze?level=run_summary` gap buckets: hardware_gap 61%, batching 29%.
- R6 (necessary) is null, so the "necessary" split can't be read. Reading: the big GEMMs are within about 1.3–1.7x of R5
  (bandwidth-bound). The latency-bound nodes (MoE routed chain, KDA recurrence, bfa GEMVs) carry the headroom.

nsys (graph-node trace, B=1 L=1M, ~106 µs/replay; see timeline_b1.txt / profile.json):
- 0–38 µs, attention phase. qkvg TGV (88 MB) streams from 6.4 to 24 µs. After that come the KDA recurrence
  (22.5–30.7), o_proj (30.3–36.2) and the 1-CTA attn_res (32.4–43.4), all latency-bound, while HBM is almost idle.
- 38–107 µs, MoE phase. It streams ~400 MB (routed front 53, gate_up 176, shared_down 88, latent_up 51,
  MXFP4 ~34) at ~5.9 TB/s, close to the B200 copy roofline (~6.0 TB/s).
  - The routed front (58 CTAs) takes 25.5 µs because it shares HBM with gate_up.
- TGV tactic sweep: all tactics are within 1% (dead end). cuBLAS is slower.

### result.json

```
{"iter":0,"desc":"seed tree baseline, plain flags","latency_us":{"1,1048576":112.06,"16,65536":171.46,"128,8192":374.27,"32,8192":231.81,"1,8192":112.03},
 "same_job_rebaseline_in_iter01_ab":{"1,1048576":109.92,"16,65536":171.33,"128,8192":374.14,"32,8192":231.42,"1,8192":109.92},
 "pristine_head_latency_us":{"1,1048576":136.67,"16,65536":200.1,"128,8192":402.94,"32,8192":262.62,"1,8192":136.67},
 "check_vs_orig":{"1,1048576":[true,0.0],"16,65536":[true,0.0],"128,8192":[true,0.0126],"32,8192":[true,0.0072],"1,8192":[true,0.0]}}
```

## iter_01
### hypothesis.md

# iter_01 hypothesis
See iter_00/hypothesis.md: fill the HBM-idle late-attention window with L2 prefetch of the weights read first in the MoE phase.
Expected: at most ~75 MB / 6 TB/s ≈ 12 µs of MoE-phase streaming could move. What I actually measured was ~4 µs, because L2
capacity and the routed front's 58-CTA ceiling limit the benefit. It must use few CTAs so it doesn't perturb the 12-CTA
KDA kernel. There is no prior trial of this in opt_history.

### analysis.md

# iter_01 — side-stream L2 prefetch of o_proj + routed-front weights

Change: a new JIT kernel `l2_prefetch` (per-128B `prefetch.global.L2::evict_normal`, grid-stride). The KDA layer calls it on
the MoE alt_stream right after the qkvg GEMM is issued, in graphed decode, for ≤32 tokens on the split-front schedule. It
prefetches o_proj.weight (22 MB) and `_front_w[n_gu:]` (the router + latent-down rows, 53 MB). The MoE split-front path
already forks and joins alt_stream, so no extra join is needed.

Profiles (nsys, B=1 L=1M):
- nb=148 (`nsys_nb148/`): the prefetch slows the 12-CTA KDA recurrence from 8.2 to 11 µs, which is on the critical path. Net ~0 (span 105.8).
- nb=32 (`nsys_nb32/`): the KDA recurrence stays at 8.2 µs, o_proj drops from 5.9 to 4.6 µs, and the routed-front weight is L2-hot. Span 102.4.
- Rejected sub-variant (`rejected_fork_after_front/`): forking gate_up after the routed front. It is hot (7.5 µs), but gate_up then
  contends with MXFP4 gemm1 and shared_down lands on the critical path. Span 107.

Knob sweep (1M / 8k), measured twice. The temporary knob was removed afterwards.
| config | 1M | 8k |
|---|---|---|
| off | 109.9 | 109.9 |
| nb=16 | 110.1 | 112.0 |
| nb=32 | 105.9 | 107.8 |
| nb=48/64/148 | 110.1 | 110.1 |
| nb=32 + 20 MB of gate_up | 105.9 | 113.9 |
| nb=32 + 40 MB of gate_up | 113.9 | 116.1 |

Adding gate_up rows over-subscribes L2, so only o_proj and the routed front are prefetched.

VibeSim: the container serves only the fixed before-state prediction (`simulate` is idempotent). I re-read `optimality` for the
nodes touched: o_proj and merged_front (routed part) are bandwidth-bound nodes with R0/R5 of 1.3–1.7, and the prefetch
moves their bytes into the window where the KDA recurrence (R0/R5 5.2) is latency-bound. The largest remaining headroom is
mxfp4_fused_moe (R0/R5 43.6). At B=1 that is the router → quant → routing → gemm1 → gemm2 → finalize latency chain
(56.7→87.4 µs in the nb=32 timeline), running parallel to gate_up + shared_down (HBM-bound, 264 MB).

### result.json

```
{"iter":1,"desc":"side-stream L2 prefetch (32 CTAs) of o_proj + routed-front weights after qkvg GEMM, <=32 tokens",
 "files":["python/sglang/srt/models/kimi_k3.py","python/sglang/kernels/jit/csrc/elementwise/l2_prefetch.cuh","python/sglang/kernels/ops/elementwise/l2_prefetch.py"],
 "before_us_same_job":{"1,1048576":[109.92,109.95],"16,65536":[171.33,171.2],"128,8192":[374.14,374.18],"32,8192":[231.42,230.85],"1,8192":[109.92,109.95]},
 "after_us_same_job":{"1,1048576":[105.86,105.82],"16,65536":[169.31,169.25],"128,8192":[374.18,374.14],"32,8192":[228.77,228.7],"1,8192":[105.89,107.81]},
 "check_vs_orig":{"1,1048576":[true,0.0],"16,65536":[true,0.0],"128,8192":[true,0.0126],"32,8192":[true,0.0072],"1,8192":[true,0.0]},
 "check_vs_seed":"all pass, max_rel_err 0.0 (bit-exact)", "smoke":"LAYER_SMOKE_OK", "accepted":true}
```

(diff.patch: 176 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/l2_prefetch.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/elementwise/l2_prefetch.py, /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)
