# MLA trial 6: agent iterations

## agent log.md

iter_00: baseline plain graph latency 680.416/345.536/357.856 us; VibeSim served p_38ccb83a6c4e44298f45b2251bd4cf54 (KDA), selected measured TRT-LLM MXFP4 MoE as the next target.
iter_01: B=128 TRT-LLM ceiling 128->256; replay 680.416/345.536/357.856 -> 680.448/345.536/357.856 us; CHECK pass on all points, no speedup.

## iter_00
### hypothesis.md

## Iteration 00 hypothesis for the next edit

The measured bottleneck is the TRT-LLM MXFP4 routed MoE. PDL is already enabled by
default on B200 and an isolated `SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS=0` diagnostic raised
the B=128 replay from the plain 680.416 us baseline to 687.520 us, so PDL should remain
enabled.

The first source experiment will change only the autotuning ceiling passed to the
TRT-LLM MXFP4 call for the B=128 decode shape. It will preserve expert ids, routing
weights, packed weights/scales, activation type, output buffer, and all state writes;
therefore it changes only tactic selection and must preserve the layer output and
post-step state. B=1 and B=16 will retain their existing shape-specific ceilings until
the all-point replay proves otherwise.

### analysis.md

## Iteration 00: baseline

The required `workspace-info` call was made first. The initial profile was measured with
the fixed MLA/CuteDSL/FlashInfer-MXFP4 CUDA-graph command. The driver wrote per-point
profiles and `run.json`; `profile.json` is a copy of that complete per-point report.

Measured plain graph replay latency (the comparison command):

| point | latency_us | launches | kernel sum us |
| --- | ---: | ---: | ---: |
| B=128, L=8192 | 680.416 | 29 | 759.221 |
| B=1, L=1048576 | 345.536 | 30 | 393.957 |
| B=16, L=65536 | 357.856 | 30 | 428.014 |

The split/profile diagnostic showed the real B=128 launch table is dominated by the two
FlashInfer TRT-LLM MXFP4 expert GEMMs: 284.842 us and 143.290 us. The CuteDSL MLA decode
kernel was 108.141 us. The same MoE node remains dominant at the other points, although
the long-context MLA kernel is the largest individual launch for B=1 and B=16.

VibeSim simulation was requested with `prediction=.:before`. This service returned
prediction id `p_38ccb83a6c4e44298f45b2251bd4cf54`, run `k3_kda`, even though
`workspace-info` also listed the MLA id `p_c7687e585abe4b5fbacdc89caf0c9761`. The
analysis endpoints served the fixed KDA before-state for every selector, so that service
limitation is recorded rather than silently calling it MLA. Its node shape and backend
still match the measured target's TRT-LLM MXFP4 expert stage and were used only as a
stage-level candidate signal.

VibeSim results for the served prediction:

- operator: `unified.kda.moe.mxfp4_fused_moe` was 0.542877 ms and 53.96% of modeled
  kernel time; the next largest operator was 0.144538 ms.
- run summary: 1.006110 ms modeled kernel time; 84.24% of the gap was batching and
  9.78% was hardware gap.
- optimality node `unified.kda.moe.mxfp4_fused_moe`, path `iter/0/0/1/3`: R0 was
  0.542878 ms, R5 was 0.009836 ms, R6/R7 were 0.231980 ms, `R0/R5=55.195`, and
  `necessary_share=R6/R0=0.427316`. `has_cached_alternative` was false.
- the attention candidate was much smaller in the measured table, so it was not targeted.

The active source mapping is `Mxfp4MoEMethod.apply` in
`python/sglang/srt/layers/quantization/mxfp4.py`, which dispatches the SM100 path to
`_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen` in
`python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`. That call passes the
dynamic `next_power_of_2(num_tokens)` autotuning ceiling and PDL to the exact TRT-LLM
MXFP4 entry point seen in the profile.

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

## Iteration 01 hypothesis

Raise only the B=128 TRT-LLM MXFP4 `tune_max_num_tokens` ceiling from 128 to 256 so
the native runner can choose a different workspace/tactic bucket. This was intended to
alter only kernel scheduling. The all-point replay proved numerical output and state
were exactly unchanged, but the latency did not improve, so this hypothesis is rejected
for the final optimization.

### analysis.md

## Iteration 01: B=128 TRT-LLM tuning ceiling

VibeSim was simulated again after the new profile. The service returned the same fixed
before-state prediction `p_38ccb83a6c4e44298f45b2251bd4cf54` (`k3_kda`), with the same
stage-level results: `mxfp4_fused_moe` 0.542877 ms / 53.96%, `R0/R5=55.195`,
`necessary_share=0.427316`, R6/R7 0.231980 ms, and no cached alternative. The same
kernel-set drill-down reported backend `sglang_trtllm_mxfp4` and
`has_cached_alternative=false`.

The re-profile diagnostic measured 703.968 / 346.560 / 381.408 us graph latency and
the plain golden replay measured 680.448 / 345.536 / 357.856 us. The B=128 delta from
the identical plain baseline was +0.032 us, well inside noise and far short of the 5%
target. Launch names and counts stayed unchanged. The measured result therefore rejects
this tuning-ceiling hypothesis; the bottleneck remains the same TRT-LLM MXFP4 expert
GEMMs.

### result.json

```
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "ok": true, "latency_us": 680.4479956626892, "latency_mode": "graph", "us_step": 1397.1199989318848, "us_step_graph": 680.4479956626892, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 1, "seq_len": 1048576, "ok": true, "latency_us": 345.5359935760498, "latency_mode": "graph", "us_step": 1385.759949684143, "us_step_graph": 345.5359935760498, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [1, 1048576], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "ok": true, "latency_us": 357.85600543022156, "latency_mode": "graph", "us_step": 1379.4560432434082, "us_step_graph": 357.85600543022156, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": "cutedsl_mla", "attn_heads": 12, "seed": 0, "error": null, "point": [16, 65536], "correctness": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "rel_err_max": 0.02, "pass": true}}
```

(diff.patch: 37 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)
