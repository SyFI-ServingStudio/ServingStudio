# KDA_LCPREFILL_CLAUDE trial 2: agent iterations

## agent log.md

- iter_00: baseline (lcprefill_claude_1 tree): capture-run 8303.6 | 16718.9 | 8369.2 | 16575.6 us; VibeSim p_c37b6f92: GEMMs ~3x r5 at practical ceiling, kda_chunk 38x (batching)
- iter_01: g/bfa in-proj GEMMs on side stream overlapping KDA chain: bit-exact but A/B <=0.3% (persistent GEMMs serialize; step is gap-free) -> REJECTED, reverted

## iter_00
### hypothesis.md

Baseline only, no edit. Built on kda_lcprefill_claude_1 (h-scan CUDA kernel + conv tuning, in tree) and
kda_prefill_claude_1 (copy removal, in tree). GEMM library swaps ruled out by those trials (nvjet at ceiling).

### analysis.md

# iter_00 — baseline (start tree = accepted kda_lcprefill_claude_1 state)
VibeSim prediction: `.:k3_kda_prefill` (id p_c37b6f92e6cb4d3ebb83e7ca5a107f7f; `.:before` does not resolve in this container).
run_summary: optimality 0.070, hardware_gap 84.7%, imbalance 5.9%, batching 2.4% (all in kda_chunk_prefill), fusion/excess 0.
optimality(scope=iter), by headroom r0-r5 (3 cases summed):
- merged_front_prefill (gemm_fp32_output) r0 16.8ms / r5 5.0 (3.4x)
- mxfp4_fused_moe_prefill 8.6 / 2.9 (3.0x)
- qkvbfg prefill 6.1 / 1.0 (6.1x; qkvbfg_a_proj 3.2x, bfa 2.9x, f_b 4.3x)
- shared_down 6.4/1.9, latent_up 3.5/1.1, o_proj 1.5/0.48 (all ~3.1-3.4x)
- kda_chunk_prefill 2.89 / 0.076 (37.9x, only node with batching gap: small-grid serial scan)
- attn_res 1.00x, add3 1.11x, routed_norm 1.19x -> at roofline, no headroom.
necessary_share is null for every node (no R6 model) -> all headroom is kernel efficiency/scheduling.

Measured (16k/pf245760, --profile-kernels): nvjet GEMMs 4685us (~1.65 PFLOPS = practical bf16 ceiling per
kda_prefill_claude_1/lcprefill_claude_1 microbenchmarks), MXFP4 MoE 1205, KDA chain ~930 (h 287, intra 120, solve 105,
o 96, recompute 87, conv 115, l2norm 41, cumsum 46, gated norm 31), attn_res 402, situ 239, add3 133.
Kernel swaps on the GEMMs are a dead end (prior trials). The actionable reading: the KDA chain is latency/small-grid
bound (38x over r5, batching gap) while the GEMMs are compute-bound and independent GEMM work exists:
the output-gate g projection (1536 of the 6144 fused qkvg columns, ~212us) is only consumed by o_norm after the
whole KDA chain, and the [f_a|b] (N=140, 48us) + f_b (16us) GEMMs run badly under-filled on the main stream.
=> target: qkvbfg node + kda_chunk_prefill node jointly (overlap the independent g/bfa GEMM work with the KDA chain).
Latencies (profile run, diagnostic): 8363 / 16780 / 8549 / 16682 us. Golden capture run: 8304 / 16719 / 8369 / 16576.

### result.json

```
{"note":"baseline, see profile.json"}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis (REJECTED)
Split the fused [q,k,v,g] in-proj GEMM (row slices of the merged weight: every output column is the same dot product,
bit-exact) and run the output-gate g GEMM (1536 cols, ~212us) plus the under-filled [f_a|b] (N=140) and f_b GEMMs
on the layer's bfa alt stream, concurrently with the latency-bound KDA chain (conv -> intra -> solve -> w/u -> small-grid
h-scan -> o); join before o_norm. Modes A/B'd: 1 = g after qkv on alt (overlaps KDA chain), 2 = g concurrent with qkv,
3 = only bfa/f_b on alt. Built on: kda_lcprefill_claude_1 (h-scan kernel; its segmented side-stream pipeline also found
no gain from overlap), TECHNIQUES A4 (side-stream overlap, "at large m both branches are compute-bound so the gain shrinks").

### analysis.md

# iter_01 analysis
VibeSim prediction `.:k3_kda_prefill` (p_c37b6f92e6cb4d3ebb83e7ca5a107f7f) targeted nodes: qkvbfg prefill (6.1x r5,
f_b 4.3x, bfa 2.9x) + kda_chunk_prefill (37.9x, batching gap).
Result: bit-exact (CHECK max_rel_err 0.0, state_ok on all 4 points), but same-job interleaved A/B at 16k/pf245760
(2 reps): mode0 8434.8/8440.5, mode1 8426.6/8397.8, mode2 8471.6/8426.3, mode3 8438.9/8379.3 -> <=0.3%, below noise.
nsys timeline (timeline_mode0.txt): the step is 7834us of gap-free GPU work on one stream; the persistent nvjet GEMM
kernels hold every SM, so a side-stream GEMM just serializes with the KDA chain (the existing shared-expert "overlap"
also serializes: routed routing kernels start only when shared_down ends). The driver's ~8400us vs nsys 7834us suggests
sustained-load clock drop, i.e. the step is energy/throughput-bound: remaining wins must reduce work or raise kernel
efficiency, not overlap. Reverted.

### result.json

```
{
 "points": {
  "1,16384,pf245760": {
   "latency_us": 8431.839942932129,
   "check": true,
   "max_rel_err": 0.0
  },
  "1,32768,pf229376": {
   "latency_us": 16729.055404663086,
   "check": true,
   "max_rel_err": 0.0
  },
  "1,16384,pf131072": {
   "latency_us": 8395.872116088867,
   "check": true,
   "max_rel_err": 0.0
  },
  "1,32768,pf131072": {
   "latency_us": 16773.02360534668,
   "check": true,
   "max_rel_err": 0.0
  }
 },
 "ab_16k_pf245760": {
  "MODE0 REP1": 8434.847831726074,
  "MODE1 REP1": 8426.624298095703,
  "MODE2 REP1": 8471.6157913208,
  "MODE3 REP1": 8438.943862915039,
  "MODE0 REP2": 8440.511703491211,
  "MODE1 REP2": 8397.791862487793,
  "MODE2 REP2": 8426.336288452148,
  "MODE3 REP2": 8379.327774047852
 },
 "verdict": "rejected, reverted"
}
```

(diff.patch: 1 lines, files: )
