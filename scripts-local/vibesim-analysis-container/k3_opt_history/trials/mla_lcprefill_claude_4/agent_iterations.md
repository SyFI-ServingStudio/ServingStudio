# MLA_LCPREFILL_CLAUDE trial 4: agent iterations

## agent log.md

- iter_00: baseline 20991.6 / 39915.0 / 15286.9 / 30586.0 us (pf245760 / 32k pf229376 / 16k pf131072 / 32k pf131072); golden captured.

## iter_00
### hypothesis.md

# iter_00 hypothesis
Baseline measurement only. Prior trials: mla_lcprefill_claude_1/2 (KV-split prefix FMHA, exact, kept - present in tree);
mla_lcprefill_claude_3 (single-launch chunk merge, non-exact, FAILED 32k/pf229376 CHECK) -> rule out any change to
merge/rounding order of prefix attention; only bit-exact restructurings are safe.
mla_prefill_claude_3 iter_01: overlapping the causal pass with the prefix pass on a side stream was bit-exact and paid off.

### analysis.md

# iter_00 analysis (baseline, no change)
VibeSim prediction: `.:k3_mla_lcprefill` = p_e31fb571641b42389d9bf7d84eace52c (vs_*.json here).
run_summary: R0 141.9 ms / 4 cases, R5 34.6 ms, optimality 0.244; buckets batching 30%, imbalance 29%, hardware_gap 16%.
necessary_share (R6): null for every node.

| node | R0 ms | R5 ms | R0/R5 |
|---|---|---|---|
| mla_prefill_attention_prefix | 54.72 | 21.99 | 2.49 |
| merged_front_prefill | 33.82 | 10.01 | 3.38 |
| mxfp4_fused_moe_prefill | 16.15 | 5.77 | 2.8 |
| shared_down_prefill | 12.7 | 3.85 | 3.3 |
| latent_up_prefill | 7.04 | 2.24 | 3.14 |
| mla_prefill_attention_causal | 2.64 | 0.92 | 2.89 |
| mla_cache_append | 1.34 | - | 63 |
| mla_prefix_gather | 0.83 | - | 7.29 |

Measured profile, 1,16384,pf245760 (kernel sum 20799 us ~ latency 20992, fully serial):
prefix FMHA x2 12185, front fp32 GEMM 2322, nvjet 128x256 x6 2188, MoE bmm1 877, causal trtllm-gen 657,
pack x3 385, attn_res x2 356, MoE bmm2 329, merge_kv_splits x2 94, D2D memcpy x9 91, get_mla_kv_buffer x2 61, merge_attn_states x2 54.
Diagnostics: MUFU ex2 f32/f16x2/bf16x2 all 4.64e12/s on B200 (packed ex2 no lever); prefix FMHA ~59% MUFU, ~56% fp8 MMA.
Fused gather+kv_b_proj+pack Triton: byte-exact but 508 us vs 504 us ref at T=131072 (no gain).

### result.json

```
{"iter":0,"change":"none (baseline)","latency_us":{"1,16384,pf245760":20991.6,"1,32768,pf229376":39915.0,"1,16384,pf131072":15286.9,"1,32768,pf131072":30586.0},"check_pass":"golden captured"}
```

(diff.patch: 0 lines, files: )
