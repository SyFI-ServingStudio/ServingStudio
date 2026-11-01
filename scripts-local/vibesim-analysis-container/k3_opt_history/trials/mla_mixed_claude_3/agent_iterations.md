# MLA_MIXED_CLAUDE trial 3: agent iterations

## agent log.md

# log
- iter_00: baseline control 8287.3 / 2480.3 / 13147.5 us (mx16384 / mx4096 / p49152), CHECK PASS x3 (err 0). exp1 CuTe causal: 8598/2704/13512, CHECK FAIL -> dead end.
- iter_01: MoE tail add3 -> in-place addmm_ accumulation into prefix_sum (kimi_k3.py _forward_fused/_forward_shared). eager 8326->8224 / 2580->2522 / 13166->13050; nsys steady step 7690->7615. CHECK PASS (mean 0.0016). Dead ends: exp2 MoE tune-at-exact-m (no gain); addmm out= (hidden 72us memcpy); side-stream pre-seed copy (not overlapped).

## iter_00
### hypothesis.md

# iter_00 — baseline
No change. Starting tree (includes prior accepted levers) is copied to /workspace/opt_run/original/sglang.
Prior trials: mla_prefill_claude_2 (CuTe causal rejected for numerics; I re-checked it in exp1 and it fails CHECK and is slower),
mla_b512_claude_1 (MoE tactic lever), A4 (shared/routed overlap gains little at large m), mla_prefill_claude_3 (causal/prefix overlap),
mla_mixed_claude_1 (VOID on this workload).

### analysis.md

# iter_00 analysis
VibeSim prediction: .:k3_mla_mixed (id p_ae432dbc924742f2bb261e1a0f3f1832)
run_summary: optimality 0.45, batching gap 13.6%, imbalance 18%.
Top headroom nodes (r0/r5): attention+output gate; merged_front_prefill 1.81; prefix attention 2.55;
mxfp4_fused_moe_prefill 1.96; shared_down 1.75; causal 2.98; latent_up 1.76; mla_cache_append 64x (tiny absolute).
nsys (primary, 7490 us/step): front GEMM 2281, shared_down 860 || MoE bmm1 852 + bmm2 317, causal FMHA 746, latent_up 483,
qkv_a 324, o_proj 226, situ_and_mul 223, g_proj 209, attn_res 173+181, add3 134, decode FMHA 66.
Ladder/necessary_share: GEMM-bound; cuBLAS vs DeepGEMM within ±5% (mb/gemm_bench.py), so the GEMM backend is not a lever.
Candidate nodes: MoE bmm1 tactic (1.7 PF vs 2.28 bmm2; 16448 tokens tuned at bucket 32768), add3 fold, decode-attn overlap.

### result.json

```
{
 "iter": 0,
 "desc": "baseline (starting tree, same-node control replay)",
 "latency_us": {
  "64,8192,mx16384": 8287.3,
  "128,8192,mx4096": 2480.3,
  "64,8192,mx16384p49152": 13147.5
 },
 "check": {
  "64,8192,mx16384": "PASS",
  "128,8192,mx4096": "PASS",
  "64,8192,mx16384p49152": "PASS"
 },
 "max_err": 0.0
}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis
Fold the tail 3-way add into GEMM epilogues: shared_down = addmm(prefix_sum, act, Wd^T) on the side stream, then
shared_output.addmm_(latent_normed, Wup^T) in place. Expect -~120 us/step on mx16384 (add3 removed, epilogue C reads hidden).
Gated to tp==1, no AR fusion, alt stream present, m>16 (decode buckets keep the TGV/add3 path) — a folded prefix_sum must not be all-reduced.
Built on: iter_00 nsys (add3 134 us). Ruled out first: exp2 (MoE tune at exact m instead of pow2 bucket 32768) — no measurable change
(8416 vs 8330/8416 control; tactic bucket not a lever); mla_b512_claude_1's tactic lever therefore does not transfer at m=16448.
Not an attempt at A4 (shared/routed overlap), which prior trials found gives little at large m.
Numerics: single fp32 epilogue rounding instead of 3 bf16 roundings -> mean_rel 0.0016 vs original (tol 0.01).

## Revision (final iter_01 form)
v1 `torch.addmm(prefix_sum, ..., out=shared_output)` hides a 236 MB D2D memcpy (72-95 us, found in CUPTI MEMCPY, not the kernel table). It stalled
the main stream (exp3 -> 64 us idle gap). v2 pre-seeding shared_output on the side stream under the front GEMM did not overlap: the copy ran at the GEMM tail (exp4).
v3 (kept): accumulate in place into prefix_sum (`prefix_sum.addmm_` on the side stream for shared_down, then `prefix_sum.addmm_(latent, Wup^T)`).
prefix_sum on the MLP side is dead after the MoE (fresh from `_aggregate_fused_add`, or the layer-local attn output), so there is no copy.

### analysis.md

# iter_01 analysis
VibeSim prediction .:k3_mla_mixed (p_ae432dbc924742f2bb261e1a0f3f1832); opt.json re-pulled (static prediction, unchanged).
Node: MoE tail `_add3(out, shared_output, prefix_sum)` = 134 us/step (943 MB at ~7 TB/s, already BW-roofline -> necessary_share of the add pass ~0:
the sum can live in the GEMM epilogues which are compute-bound). Ladder: shared_down/latent_up are compute-bound nvjet tst (1.2-1.6 PF),
so an extra C read in their epilogue is hidden.
Measured (nsys ksum, ksum.py exp3/ctl.sqlite exp3/fold.sqlite): add3 -1203/9 us; nvjet tst GEMMs 17700 -> 17718 (+2 us/step); total GPU -118 us/step.

### result.json

```
{
 "iter": 1,
 "node": "moe tail add3 -> in-place beta=1 accumulation of shared_down and latent_up into prefix_sum",
 "latency_us": {
  "64,8192,mx16384": {
   "before": 8325.8,
   "after": 8224.0
  },
  "128,8192,mx4096": {
   "before": 2579.7,
   "after": 2522.2
  },
  "64,8192,mx16384p49152": {
   "before": 13165.9,
   "after": 13050.3
  }
 },
 "latency_note": "eager replay; before = mean of 2 same-node controls (exp3), after = exp5 fold1; noise +-1%",
 "nsys_steady_step_us_mx16384": {
  "before": [
   7697,
   7686
  ],
  "after": 7615
 },
 "check": "PASS x3: max 0.0075, mean 0.00158, rows over tol 0, state True"
}
```

(diff.patch: 83 lines, files: /sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py)
