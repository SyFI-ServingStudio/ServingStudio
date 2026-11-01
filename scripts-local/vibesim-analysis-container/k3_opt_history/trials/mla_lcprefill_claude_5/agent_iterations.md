# MLA_LCPREFILL_CLAUDE trial 5: agent iterations

## agent log.md

iter_00 | baseline | 16k/pf245760 20686.2, 32k/pf229376 39738.6, 16k/pf131072 15161.5, 32k/pf131072 30316.7 | golden captured
iter_01 | single FMHA launch over prefix chunks (bit-exact) | A/B +0.15/-0.06/-0.28/+0.29% | CHECK pass | REVERTED (neutral)

## iter_00
### hypothesis.md

# iter_00 — baseline (no change)
Baseline measurement + golden capture only. Builds on mla_lcprefill_claude_1/2 (accepted:
cute-dsl prefix FMHA engaged, KV split, 40 emulated exp2) — current tree already contains them.
Rules out repeating mla_lcprefill_claude_3 as-is (single launch over the *whole* prefix with
recomputed split boundaries -> non bit-exact, FAIL at 32k/pf229376).

### analysis.md

# iter_00 — baseline analysis (pristine tree)

Kernel breakdown @ 1,16384,pf245760 (step 20686 us, profile.json):
| kernel | us | note |
|---|---|---|
| cute-dsl Fp8PrefillFmha x2 | 12232 | prefix chunks 131072 + 114688, S=4 KV splits each, 3072 CTAs/launch (20.76 waves) |
| nvjet tss 128x256 (merged front, fp32 out) | 2255 | ~74% of bf16 peak |
| nvjet tst 128x256 x6 | 2128 | shared_down / latent_up / o_proj |
| MoE bmm1/bmm2 | 855/322 | |
| trtllm-gen causal FMHA | 646 | |
| _v1_flat (fp8 pack) x3 | 383 | |
| attn_res x2 | 354 | |
| merge_kv_splits x2 + merge_attn_states x2 | 94 + 55 | |
| DtoD memcpy x9 | 90 | Q replication for KV splits (4 per launch) |

VibeSim (summed 4 cases, prediction=.:k3_mla_lcprefill): optimality 0.244; top node
unified.mla.attention.mla_prefill_attention_prefix R0 54.7 ms vs R5 22.0 ms (2.49x);
merged_front_prefill 33.8 vs 10.0; buckets batching 30% / imbalance 29% / hardware_gap 16%.
The "imbalance/batching" share on the prefix node is the per-chunk launch structure
(two partial last waves, per-chunk Q quant + replication + two merges).

FMHA achieved ~14 scores/clk/SM vs ~23 MUFU(+emulation) / ~25 fp8-tensor bound.

Numerics: trial mla_lcprefill_claude_3 shows any change to the prefix accumulation
(different KV split boundaries) costs mean_rel 0.008-0.013 on 2-chunk points and failed
32k/pf229376 -> only bit-exact changes to the attention path are admissible.
merge_attn_states.cu merges log2-domain LSEs with natural expf/logf (fast-math build);
that must be preserved as-is.

### result.json

```
{
 "iter": 0,
 "change": "baseline (no edit)",
 "latency_us": {
  "1,16384,pf245760": 20686.176300048828,
  "1,32768,pf229376": 39738.590240478516,
  "1,16384,pf131072": 15161.503791809082,
  "1,32768,pf131072": 30316.70379638672
 },
 "check": "golden captured from this tree (reference)"
}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis
Builds on mla_lcprefill_claude_1 (cute-dsl prefix FMHA engaged, S=4/S=2 split policy) and
mla_lcprefill_claude_2 (exp2 emulation). mla_lcprefill_claude_3 got +1.4% by changing the
prefix split structure but FAILED numerics at 32k/pf229376 -> rules out any change to split
boundaries. Hypothesis: the 2-chunk overhead (partial tail waves x2, extra Q replication,
launch gaps) can be removed while keeping per-chunk split boundaries identical, by one
ragged launch over all chunk-splits; expected ~0.5-1% @16k/pf245760 and pf229376.
Result: bit-exact but neutral (+0.15%); the FMHA is power-bound, not wave-quantization bound.

### analysis.md

# iter_01 analysis
Baseline profile (iter_00) @16k/pf245760: prefix FMHA 2 launches (131072 + 114688 KV, S=4 splits,
20.76 waves each) = 12.2 ms, i.e. 59% of step. VibeSim top node mla_prefill_attention_prefix
(2.49x from R5), buckets batching 30% / imbalance 29% -> per-chunk launch structure
(two partial tail waves, per-chunk Q quant + 4x Q replication + two merges).

Tried: pack both chunks into one fp8 K/V buffer and run a single ragged FMHA launch over
C*S = 8 segments with the SAME per-chunk split boundaries (bit-exact by construction).
Replay CHECK: all 4 points pass, max_abs_err 0 vs golden.
Same-job A/B (ab.sh, median of 3, plain flags):
| point | A us | B us | gain |
| 16k/pf245760 | 20913.6 | 20882.7 | +0.15% |
| 32k/pf229376 | 39981.4 | 40006.0 | -0.06% |
| 16k/pf131072 | 15383.8 | 15426.8 | -0.28% (1 chunk, path not taken: noise) |
| 32k/pf131072 | 31076.5 | 30987.4 | +0.29% (1 chunk: noise) |
Profile B: fused launch time ~= sum of the two launches; the tail-wave saving (~1 wave of 41)
is eaten by the sustained-power clock drop (FMHA is power-throttled: 7.4 -> 8.5 ms standalone).
Side experiment: re-scheduling the exp2 emulation interleave in softmax_step: bit-exact, 0% gain.
Verdict: neutral, below the 0.14% 3-sigma bar & extra 2x prefix fp8 buffer -> REVERTED.

### result.json

```
{"iter":1,"change":"single cute-dsl FMHA launch over all prefix chunks (bit-exact)","status":"reverted (neutral)",
 "ab_median_us":{"1,16384,pf245760":{"A":20913.6,"B":20882.7,"gain_pct":0.15},
  "1,32768,pf229376":{"A":39981.4,"B":40006.0,"gain_pct":-0.06},
  "1,16384,pf131072":{"A":15383.8,"B":15426.8,"gain_pct":-0.28},
  "1,32768,pf131072":{"A":31076.5,"B":30987.4,"gain_pct":0.29}},
 "check":"PASS all 4 points, max_abs_err 0"}
```

(diff.patch: 274 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/cute_dsl_fmha_fp8.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py, /sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py)
