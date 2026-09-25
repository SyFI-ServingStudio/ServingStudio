# MLA trial 8: agent iterations

## agent log.md

iter_00: baseline; graph latency 304.64 -> 304.64 us (B1/L1M), 493.06 us (B128/L8K), 302.56 us (B16/L65K); golden captured and VibeSim selected MLA cached alternative for iter_01.
iter_01: routed no-DCP cutedsl_mla decode to TRT-LLM MLA; plain graph 304.64 -> 267.78 us, 493.06 -> 485.98 us, 302.56 -> 300.48 us; replay CHECK pass on all points and state_ok true.

## iter_00
### hypothesis.md

No source change in iter_00. Baseline hypothesis for iter_01: when DCP is
disabled, the `cutedsl_mla` backend can use the existing TRT-LLM MLA decode
alternative through the same parent call. The query layout, FP8 KV cache,
sequence metadata, scale, output tensor, and cache write remain unchanged, so
the numerical output and post-step state should match the captured golden.
The explicit CuteDSL call in the DCP-only path remains untouched.

### analysis.md

Prediction built: `p_c7687e585abe4b5fbacdc89caf0c9761` via
`GET /api/v1/simulate?prediction=.:before` (`vibesim:k3_mla`). The analysis
endpoints use the current fixed `k3_mla` workspace when no selector is passed;
their raw responses are saved beside this file.

VibeSim operator summary: total modeled kernel time 1.1589 ms. The largest
operators are `unified.mla.moe.mxfp4_fused_moe` at 0.3792 ms (32.7%) and
`unified.mla.attention.mla_decode_attention` at 0.3588 ms (31.0%). The
iteration tree models four cases, including the scored B=1/L=1,048,576 case
at 255 us, where MLA attention is 169 us and its decode leaf is 139 us.

Optimality (`scope=iter`) selected two relevant candidates:

- `unified.mla.moe.mxfp4_fused_moe`: R0=379.16 us, R5=8.57 us,
  R6/R7=227.76 us, `necessary_share=0.6007`, and the largest R0-R5
  headroom. Its kernel drill-down reports `has_cached_alternative=false`.
- `unified.mla.attention.mla_decode_attention`: R0=358.84 us, R5=227.56 us,
  R6/R7=454.16 us, `necessary_share=1.2657`. Its drill-down reports the
  alternatives `sglang_cutedsl_mla` and `sglang_trtllm_mla`, with
  `has_cached_alternative=true`.

The aggregate ladder favors the MXFP4 leaf, but the measured primary point is
dominated by the single CuteDSL MLA kernel (166.70 us of 304.61 us graph
latency), and the MLA leaf has a cached implementation alternative. The first
experiment therefore targets the MLA dispatch while retaining the fixed public
`cutedsl_mla` entry point. The full per-kernel profiles are
`profile_B1_L1048576.json`, `profile_B128_L8192.json`, and
`profile_B16_L65536.json`; `profile.json` is the run summary containing all
three point tables and latencies.

Baseline plain graph latencies: B=1/L=1,048,576: 304.64 us; B=128/L=8,192:
493.06 us; B=16/L=65,536: 302.56 us.

### result.json

```
{
  "mode": "baseline_capture",
  "prediction_id": "p_c7687e585abe4b5fbacdc89caf0c9761",
  "points": [
    {"B": 1, "seq_len": 1048576, "latency_us": 304.639995098114},
    {"B": 128, "seq_len": 8192, "latency_us": 493.0559992790222},
    {"B": 16, "seq_len": 65536, "latency_us": 302.5600016117096}
  ],
  "check": "not applicable: this iteration captured the golden reference"
}
```

(diff.patch: 1 lines, files: )

## iter_01
### hypothesis.md

Hypothesis: route ordinary (`cp_world <= 1`) `cutedsl_mla` decode through the
cached `trtllm-gen` MLA implementation. VibeSim identifies this as the only
cached alternative for the measured `mla_decode_attention` leaf, and the
primary profile shows that leaf is the largest single launch.

The change is dispatch-only. Both implementations consume the same prepared
FP8 `[q_nope | q_rope]` query, paged FP8 latent KV cache, block table, sequence
lengths, and BMM1 scale, and both return the same MLA output. KV writes and all
metadata preparation occur before `_run_decode_kernel` and are unchanged. For
DCP (`cp_world > 1`), the existing explicit CuteDSL call is still selected.

### analysis.md

Prediction refreshed after the iter_01 profile: `p_c7687e585abe4b5fbacdc89caf0c9761`
via `GET /api/v1/simulate?prediction=.:before`. The service documents this
as a fixed before-state prediction, so its modeled operator values are
unchanged; the post-edit raw responses are saved in the `vibesim_*.json`
files here.

The ladder still identifies aggregate `unified.mla.moe.mxfp4_fused_moe` as
the largest headroom candidate (R0=379.16 us, R5=8.57 us, R6/R7=227.76 us,
`necessary_share=0.6007`, no cached alternative). The MLA decode leaf remains
the measured primary-point target (R0=358.84 us, R5=227.56 us, R6/R7=454.16
us, `necessary_share=1.2657`) and still advertises the cached
`sglang_trtllm_mla` alternative. The operator and iteration analyses still
show the modeled B=1/L=1,048,576 attention subtree as the dominant primary
path; the real profile confirms the selected kernel changed from the
CuteDSL split-KV kernel to `fmhaSm100fKernel_...ForGen`.

Post-edit diagnostic profile facts: B=1/L=1,048,576 has 29 launches,
310.83 us kernel sum, and 267.78 us graph latency; B=128/L=8,192 has 29
launches, 532.82 us kernel sum, and 479.84 us graph latency; B=16/L=65,536
has 29 launches, 348.50 us kernel sum, and 303.62 us graph latency. These
profiled values include `--split` and are diagnostic only. The plain replay
latencies in `result.json` are the timing reference.

The remaining aggregate VibeSim headroom is concentrated in the MXFP4 MoE
leaf, but that leaf has no cached alternative and is a larger contributor at
the B=128 point than at the scored B=1 long-context point. The current change
already removes the only cached primary MLA alternative and improves all
three plain replay points, so no speculative MoE rewrite is justified by the
available evidence.

### result.json

```
{
  "mode": "replay",
  "prediction_id": "p_c7687e585abe4b5fbacdc89caf0c9761",
  "plain_timing_reference": {
    "B1_L1048576": 267.7760124206543,
    "B128_L8192": 485.9839975833893,
    "B16_L65536": 300.4800081253052
  },
  "points": [
    {
      "B": 1,
      "seq_len": 1048576,
      "latency_us": 267.2640085220337,
      "check": {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_rel_err": 0.0, "nan": false, "state_ok": true, "pass": true}
    },
    {
      "B": 128,
      "seq_len": 8192,
      "latency_us": 484.8639965057373,
      "check": {"max_abs_err": 0.1875, "max_rel_err": 0.01376146687989234, "mean_rel_err": 0.0009116546716541052, "nan": false, "state_ok": true, "pass": true}
    },
    {
      "B": 16,
      "seq_len": 65536,
      "latency_us": 300.54399371147156,
      "check": {"max_abs_err": 0.0703125, "max_rel_err": 0.005136985926064956, "mean_rel_err": 0.00035468171699903905, "nan": false, "state_ok": true, "pass": true}
    }
  ]
}
```

(diff.patch: 14 lines, files: python/sglang/srt/layers/attention/cutedsl_mla_backend.py)
