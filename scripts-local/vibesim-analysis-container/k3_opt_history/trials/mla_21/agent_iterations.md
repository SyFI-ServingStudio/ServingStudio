# MLA trial 21: agent iterations

## iter_00
### hypothesis.md

Iteration 00 hypothesis for the next edit

The scored B=1 graph is dominated by the Blackwell MLA decode launch. The
current `CuteDslMLABackend` intentionally constructs its parent with
`backend="trtllm-gen"`, so the ordinary non-DCP decode uses the cached TRT-LLM
path while DCP keeps the explicit CuteDSL path. I will benchmark the
alternative decode backend on the same source path. The change is limited to
kernel selection/configuration: query, paged FP8 KV, sequence lengths, cache
write, output shape, and all arithmetic inputs remain identical, so the
layer output and post-step KV/state should be unchanged within the required
2% tolerance. Any candidate that fails the smoke, replay checks, or either
secondary workload will be reverted.

### analysis.md

Iteration 00 baseline analysis

VibeSim workspace-info was queried first. `simulate?prediction=.:before` built/fetched
prediction `p_c7687e585abe4b5fbacdc89caf0c9761` (`k3_mla`, KimiK3SglangModel,
NVIDIA B200). The analysis service exposes the fixed prediction through the
analysis verbs; `prediction_id` was accepted but did not change the fixed log.

Operator analysis (1.158900 ms predicted kernel time):

- `unified.mla.moe.mxfp4_fused_moe`: 0.379161 ms, 32.72%.
- `unified.mla.attention.mla_decode_attention`: 0.358836 ms, 30.96%.
- `unified.mla.moe.merged_front`: 0.158805 ms, 13.70%.
- `shared_down`: 0.065361 ms, 5.64%; `latent_up`: 0.043098 ms, 3.72%.

Run-summary analysis: R0=1.158899 ms, R5=0.147162 ms, hardware-necessary
R6=0.147162 ms, optimality/necessary ratio=0.126985. The main gap buckets were
batching 0.379954 ms (32.79%), hardware gap 0.427345 ms (36.88%), and imbalance
0.204437 ms (17.64%).

Roofline `optimality?scope=iter` ranked the fused MXFP4 MoE first by
`headroom_r0_minus_r5`: R0=0.000379160 s, R5=0.000008573 s, R6/R7=
0.000227756 s, R0/R5=44.23, necessary_share=0.6007. MLA decode was the
next actionable leaf: R0=0.000358836 s, R5=0.000227559 s, R6/R7=
0.000454164 s, R0/R5=1.577, necessary_share=1.266. Its VibeSim kernel
record reports alternatives `sglang_cutedsl_mla` and `sglang_trtllm_mla`,
with `has_cached_alternative=true`.

The actual CUDA-graph profile confirms the long-context MLA leaf is the
dominant single launch at the scored point: `fmhaSm100fKernel...` 130.679 us
at B=1/L=1048576. The profile also shows the current source's fused KV/query
launch (`set_mla_kv_concat_q_fp8_kernel`) at 5.494 us and the fused local
route/quant launch at 4.042 us. At B=128/L=8192 the MLA kernel is 101.260 us
and the largest MXFP4 expert GEMM is 144.860 us; at B=16/L=65536 MLA is
101.648 us and the largest MXFP4 expert GEMM is 49.741 us.

The next experiment therefore targets only the MLA decode launch selection/
configuration, with the existing MoE fusion left unchanged. The service's
`kernels` endpoint required `kernel_set`; querying the two leaf names returned
the shape/backend/alternative records above.

## iter_01
### hypothesis.md

Iteration 01 hypothesis

Use `backend="cute-dsl"` in `CuteDslMLABackend` so the ordinary decode leaf
uses the VibeSim-reported alternative. This changes only the implementation of
the same MLA operation; the query, FP8 paged KV cache, cache write, sequence
metadata, output shape, and post-step state are unchanged. The experiment
passed all correctness checks but failed the latency criterion, so it is not
retained.

### analysis.md

Iteration 01 candidate analysis

The candidate diagnostic profile was obtained with the same fixed workload and
then routed through VibeSim. `simulate?prediction=.:before` returned the fixed
prediction `p_c7687e585abe4b5fbacdc89caf0c9761`; operator/run-summary/
iteration/optimality were rerun, and the MLA leaf was queried with
`kernel_set=unified.mla.attention.mla_decode_attention` because this service
requires `kernel_set` for the kernels verb.

The VibeSim values are unchanged because this container serves a fixed
before-state prediction: MXFP4 fused MoE remains 0.379161 ms (32.72%) and MLA
decode remains 0.358836 ms (30.96%). The MLA roofline is R0=0.000358836 s,
R5=0.000227559 s, R6/R7=0.000454164 s, R0/R5=1.577, necessary_share=1.266,
with a cached alternative. The actual post-edit profile changed the MLA
kernel family to the CuteDSL implementation and showed 29 launches; the
measured graph replay was 308.736 us at B=1/L=1048576, 489.856 us in the
diagnostic B=128 profile, and 307.648 us at B=16. The plain replay was 308.704,
493.056, and 304.640 us respectively.

The alternative preserved numerics: replay CHECK passed at all points and
`state_ok=true`; max relative output error was 0.0, 0.013761, and 0.004852.
It was nevertheless rejected because the scored point regressed 15.3% and
both secondary points were also slower than the before reference. The source
selection change is reverted.

### result.json

```
{
  "candidate": "cute-dsl MLA decode backend",
  "before_plain_latency_us": {
    "1,1048576": 267.64801144599915,
    "128,8192": 485.85599660873413,
    "16,65536": 300.57600140571594
  },
  "after_plain_replay_latency_us": {
    "1,1048576": 308.7039887905121,
    "128,8192": 493.0559992790222,
    "16,65536": 304.639995098114
  },
  "checks": {
    "1,1048576": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.01376146687989234, "state_ok": true, "pass": true},
    "16,65536": {"max_rel_err": 0.0048515978190613466, "state_ok": true, "pass": true}
  },
  "retained": false,
  "replay_exit_code": 0
}
```

(diff.patch: 16 lines, files: python/sglang/srt/layers/attention/cutedsl_mla_backend.py)

## iter_02
### hypothesis.md

Iteration 02 hypothesis

The top VibeSim node is the TRT-LLM MXFP4 fused MoE. Its short expert GEMMs
currently enable PDL for every required decode batch. CUDA graph replay already
captures the dependency chain, so removing PDL setup might reduce launch
overhead without changing any arithmetic or buffers. The experiment was
correctness-preserving but slower and is not retained.

### analysis.md

Iteration 02 candidate analysis

The candidate profile was measured after the edit and then routed through a
fresh `simulate?prediction=.:before` fetch followed by operator, run-summary,
optimality, and MoE-kernel analysis. The fixed prediction remained
`p_c7687e585abe4b5fbacdc89caf0c9761`.

VibeSim still ranks `unified.mla.moe.mxfp4_fused_moe` first: R0=0.000379160
s, R5=0.000008573 s, R6/R7=0.000227756 s, R0/R5=44.23, necessary_share=
0.6007, with no cached alternative. The predicted operator share is 32.72%
(0.379161 ms). The kernel drill-down identifies the production shape as
112 local experts, top-k 2, hidden 3584, intermediate 3072, BF16 input,
MXFP4 E2M1/UE8M0 weights, and backend `sglang_trtllm_mxfp4`.

The source experiment set `_TRTLLM_MOE_PDL_MAX_TOKENS=0`, disabling PDL for
the TRT-LLM MoE launch. The post-edit profile still launched the expected
MXFP4 path (28/29 launches), but the plain replay was slower at every point:
269.856 us vs 267.648 us, 488.992 us vs 485.856 us, and 302.592 us vs
300.576 us. CHECK passed exactly at all three points (`max_rel_err=0`,
`state_ok=true`). Since PDL is necessary for this short expert chain on the
measured B200, the edit is reverted.

### result.json

```
{
  "candidate": "disable TRT-LLM MoE PDL",
  "before_plain_latency_us": {
    "1,1048576": 267.64801144599915,
    "128,8192": 485.85599660873413,
    "16,65536": 300.57600140571594
  },
  "after_plain_replay_latency_us": {
    "1,1048576": 269.8560059070587,
    "128,8192": 488.99200558662415,
    "16,65536": 302.592009305954
  },
  "checks": {
    "1,1048576": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "16,65536": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "retained": false,
  "replay_exit_code": 0
}
```

(diff.patch: 15 lines, files: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)

## iter_03
### hypothesis.md

Iteration 03 hypothesis

The VibeSim MoE breakdown leaves shared-down work independent of the routed
MXFP4 expert branch. The fused K3 front already has a dedicated side stream,
so issuing the shared branch there and joining before the common tail should
reduce the critical path without changing arithmetic, buffer ownership, or
reduction order. It was exact numerically but harmed the primary single-request
graph, so the edit is not retained.

### analysis.md

Iteration 03 candidate analysis

The candidate profile was measured after editing the K3 fused-front dataflow,
then routed through `simulate?prediction=.:before` and the operator,
run-summary, optimality, and MoE-kernel verbs. The fixed prediction remained
`p_c7687e585abe4b5fbacdc89caf0c9761`.

VibeSim still ranks `unified.mla.moe.mxfp4_fused_moe` first (R0=0.000379160
s, R5=0.000008573 s, R6/R7=0.000227756 s, R0/R5=44.23,
necessary_share=0.6007), with `shared_down` the largest modeled independent
shared branch at 0.065361 ms. The actual candidate profile confirms the same
TRT-LLM MXFP4 leaf and its production 112-expert/top-k-2 shape.

The edit overlaps `_forward_shared` on the existing K3 MoE side stream with
`_forward_routed` on the current stream, joining before the shared latent tail.
Replay CHECK passed exactly at all points (`max_rel_err=0`, `state_ok=true`),
and the secondary graph timings improved in the replay to 476.736 us (B=128)
and 290.400 us (B=16). The primary B=1/L=1048576 replay regressed to
304.320 us from 267.648 us because the extra captured stream dependency costs
more than the hidden shared work for one request. The unconditional overlap is
therefore rejected despite correct output/state; the profile showed 28/29
launches as expected.

### result.json

```
{
  "candidate": "overlap fused-front shared experts with routed MXFP4 experts",
  "before_plain_latency_us": {
    "1,1048576": 267.64801144599915,
    "128,8192": 485.85599660873413,
    "16,65536": 300.57600140571594
  },
  "after_plain_replay_latency_us": {
    "1,1048576": 304.32000756263733,
    "128,8192": 476.73600912094116,
    "16,65536": 290.3999984264374
  },
  "checks": {
    "1,1048576": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "128,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "16,65536": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "retained": false,
  "replay_exit_code": 0
}
```

(diff.patch: 26 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_04
### hypothesis.md

Iteration 04 hypothesis

Select FlashInfer's FP16-softmax TRT-LLM MLA cubin to reduce the dominant
long-context attention launch while leaving its inputs, KV writes, output
shape, and state unchanged. The installed B200/SM100 FlashInfer build does
not provide this tactic; the hypothesis is rejected at the smoke gate.

### analysis.md

Iteration 04 candidate analysis

The retained baseline prediction is `p_c7687e585abe4b5fbacdc89caf0c9761`.
Before this edit, VibeSim had identified `unified.mla.attention.mla_decode_attention`
as the actionable attention leaf (R0=0.000358836 s, R5=0.000227559 s,
R0/R5=1.577, necessary_share=1.266, cached alternative available), and the
actual profile showed the TRT-LLM `fmhaSm100fKernel...` launch dominating the
long-context point.

The proposed FlashInfer MLA dispatcher call with `use_fp16_softmax=True` was
not a valid B200 candidate. The required smoke failed inside the layer with
`ValueError: use_fp16_softmax is only supported on SM107 (Rubin); the current
device is sm100`. No valid post-edit profile or replay CHECK was produced,
and the source was reverted immediately.

### result.json

```
{
  "candidate": "TRT-LLM MLA FP16 softmax tactic",
  "smoke": {
    "layer_point": "1,8192",
    "error": "ValueError: use_fp16_softmax is only supported on SM107 (Rubin); current device is sm100",
    "pass": false
  },
  "replay": null,
  "retained": false
}
```

(diff.patch: 22 lines, files: /tmp/iter04_candidate_trtllm_mla_backend.py)

## iter_05
### hypothesis.md

Hypothesis: pass `is_var_seq=False` to the TRT-LLM MLA decode wrapper for the
fixed-context decode points. The driver gives every request in a point the same
context length, so the flag should select a lower-overhead fixed-length launch
schedule while preserving the same page table, KV rows, softmax, and output.

The candidate was rejected. It passed the smoke and all replay checks, but the
profile still launched the same `...VarSeq...` FMHA kernel and the primary
replay was 267.808 us versus the 267.648 us baseline.

### analysis.md

Iteration 05 candidate analysis

The previous measured profile was routed through VibeSim before deciding on
this candidate. A fresh candidate profile was then built with
`simulate?prediction=.:before` and analyzed with the operator, run-summary,
iteration, optimality, and kernel verbs. The service returned the fixed
prediction `p_c7687e585abe4b5fbacdc89caf0c9761`.

VibeSim still ranks `unified.mla.moe.mxfp4_fused_moe` first:
R0=0.000379160 s, R5=0.000008573 s, R6/R7=0.000227756 s,
R0/R5=44.23, necessary_share=0.6007. The MLA decode leaf remains
R0=0.000358836 s, R5=0.000227559 s, R6/R7=0.000454164 s,
R0/R5=1.58, necessary_share=1.2657, with a cached alternative. The
candidate profile's attention table still showed the same TRT-LLM
`fmhaSm100fKernel_...VarSeq...` launch at 130.134 us for B=1/L=1048576,
so `is_var_seq=False` did not select a different kernel in this dispatch.

Replay against the golden passed exactly at all points, but timings were
267.808 us, 486.944 us, and 300.544 us for B=1/L=1048576, B=128/L=8192,
and B=16/L=65536. The primary result is not an improvement, so the source
was reverted and the post-revert smoke passed.

### result.json

```
{
  "candidate": "TRT-LLM MLA fixed-sequence dispatch flag",
  "retained": false,
  "baseline_graph_latency_us": {
    "1,1048576": 267.64801144599915,
    "128,8192": 485.85599660873413,
    "16,65536": 300.57600140571594
  },
  "candidate_replay": [
    {
      "point": [1, 1048576],
      "latency_us": 267.8079903125763,
      "check": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
    },
    {
      "point": [128, 8192],
      "latency_us": 486.9439899921417,
      "check": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
    },
    {
      "point": [16, 65536],
      "latency_us": 300.54399371147156,
      "check": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
    }
  ],
  "profile_graph_latency_us": {
    "1,1048576": 267.8079903125763,
    "128,8192": 481.82401061058044,
    "16,65536": 303.6159873008728
  }
}
```

(diff.patch: 11 lines, files: /tmp/iter05_candidate_trtllm_mla_backend.py)

## iter_06
### hypothesis.md

Hypothesis: turn off PDL for the TRT-LLM MLA decode launch. The kernel's
arithmetic and all tensor arguments are unchanged, so outputs and KV state
should remain identical; only the dependent-launch mechanism changes.

The candidate was rejected. It passed all checks, but graph replay rose from
267.648 to 271.840 us at the primary point and from 485.856 to 488.992 us at
B=128. PDL was restored.

### analysis.md

Iteration 06 candidate analysis

The PDL candidate profile was routed through VibeSim after
`simulate?prediction=.:before`. The fixed prediction was
`p_c7687e585abe4b5fbacdc89caf0c9761`; operator, run-summary, optimality, and
kernel analysis returned the same before-state prediction.

The modeled top node remains `unified.mla.moe.mxfp4_fused_moe` with
R0=0.000379160 s, R5=0.000008573 s, R6/R7=0.000227756 s, R0/R5=44.23,
and necessary_share=0.6007. MLA decode remains R0=0.000358836 s,
R5=0.000227559 s, R6/R7=0.000454164 s, R0/R5=1.58, and
necessary_share=1.2657. Its cached TRT-LLM alternative is already the
measured backend; the candidate only removed PDL from that launch.

The candidate profile showed graph latencies 269.760 us, 481.792 us, and
303.488 us for B=1/L=1048576, B=128/L=8192, and B=16/L=65536. The plain
replay against the golden measured 271.840 us, 488.992 us, and 300.512 us,
with `max_rel_err=0`, `state_ok=true`, and `pass=true` at every point. PDL
overlap is therefore useful for the scored long-context point and was
restored; the post-revert smoke passed.

### result.json

```
{
  "candidate": "disable TRT-LLM MLA PDL",
  "retained": false,
  "baseline_graph_latency_us": {
    "1,1048576": 267.64801144599915,
    "128,8192": 485.85599660873413,
    "16,65536": 300.57600140571594
  },
  "candidate_replay": [
    {
      "point": [1, 1048576],
      "latency_us": 271.84000611305237,
      "check": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
    },
    {
      "point": [128, 8192],
      "latency_us": 488.99200558662415,
      "check": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
    },
    {
      "point": [16, 65536],
      "latency_us": 300.5119860172272,
      "check": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
    }
  ],
  "profile_graph_latency_us": {
    "1,1048576": 269.76001262664795,
    "128,8192": 481.79200291633606,
    "16,65536": 303.48798632621765
  }
}
```

(diff.patch: 15 lines, files: /tmp/iter06_candidate_trtllm_mla_backend.py)
