# KDA_VERIFY_CLAUDE trial 3: agent iterations

## agent log.md

# KDA TARGET_VERIFY (vk3) optimization log
- iter_00: baseline 488.9 / 655.3 / 299.4 us (64x8k / 128x8k / 16x64k). Top: MoE bmm 252us (closed cubin), verify recurrence 64us (2x off roofline).
- iter_01: CUDA JIT verify recurrence (kda_verify_recurrent.cuh) -> 472.5 / 620.1 / 292.3 us, CHECK pass all (rel 0.0099/0.0118/0.0066).
- iter_02: token-major conv1d out= (drop transpose copy) -> 472.4 / 622.9 / 292.2 us, no gain (copy not from that reshape); REVERTED. Verify-kernel store-policy/occupancy/TMA variants all neutral or worse.
- final: replay (identical cmd) 472.5 / 620.0 / 292.3 us, CHECK pass all (rel 0.0099/0.0118/0.0066, state_ok). Tree = iter_01.

## iter_00
### analysis.md

# iter_00 baseline
VibeSim: the container serves one fixed prediction, `k3_kda` (p_38ccb83a6c4e44298f45b2251bd4cf54, decode workload; the
`prediction=.:before` handle does not resolve, use `run=k3_kda`). optimality (scope=iter), top nodes:
- mxfp4_fused_moe R0 412us / R5 9.5us (r0/r5 43.6), necessary_share n/a: closed TRT-LLM cubin; prior trials (kda_b512_*, TECHNIQUES B) found no lever.
- qkvbfg r0/r5 3.0; kda_recurrent_decode r0/r5 5.2 (the recurrence node); merged_front 1.32; shared_down 1.49.
run_summary: hardware_gap 61%, batching 29%.
The prediction models decode, not TARGET_VERIFY, so it is used only for its per-node ladder. The measured verify profile
(B=64 vk3, 256 tokens) is: MoE bmm 166.6+85.9us, fused_sigmoid_gating_delta_rule_update 64.5us (B=128: 118us), front GEMMs, ...
Verify recurrence mandatory traffic: bf16 state read 25MB + fp32 4-step checkpoints 201MB = 226MB -> ~30-35us at HBM BW,
measured 64us (~4 TB/s): the node that maps to kda_recurrent_decode has the largest code-controllable headroom (R0/R5 ~2).
Target: the verify recurrence kernel.

### result.json

```
JSON {"B": 64, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 256, "ok": true, "latency_us": 488.95999789237976, "latency_mode": "graph", "us_step": 1468.8960313796997, "us_step_graph": 488.95999789237976, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [64, 8192, "vk3"]}
JSON {"B": 128, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 512, "ok": true, "latency_us": 655.8719873428345, "latency_mode": "graph", "us_step": 1442.3359632492065, "us_step_graph": 655.8719873428345, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192, "vk3"]}
JSON {"B": 16, "seq_len": 65536, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 64, "ok": true, "latency_us": 299.3920147418976, "latency_mode": "graph", "us_step": 1660.3519916534424, "us_step_graph": 299.3920147418976, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [16, 65536, "vk3"]}
```

## iter_01
### hypothesis.md

# iter_01: CUDA JIT kernel for the KDA chain-verify recurrence
TARGET_VERIFY on K3 (bf16 state) takes the unfused path: SGLANG_OPT_FUSED_KDA_VERIFY=0 in the judge env, and the fused
chain-verify needs fp32 state anyway. The Triton `fused_sigmoid_gating_delta_rule_update` (one warp per 128x32 tile,
serial load->compute->store per step) is store-stream bound at ~4 TB/s. A Triton config sweep (BV 16/32/64, warps 1/2/4,
stages, .cs stores) only reached -2..3%.
New kernel (kda_verify_recurrent.cuh): 128-thread CTA, 16 row groups x 8 lanes (16 k per lane, 3-shuffle reductions),
all draft tokens' q/k/exp(g)/beta staged in smem up front, state rows in registers, coalesced 128B st.global.cs float4
checkpoint stores. Microbench: 63.1->38.2us (N=64), 113.5->82.0 (N=128), 19.9->11.3 (N=16).
Numerics: same fp32 formulas and op order as the Triton kernel (gate lower_bound*sigmoid(exp(A_log)*(a+dt)), beta,
l2norm eps 1e-6, h*=exp(g), delta, o=h.q). Only the reduction tree differs: checkpoints rel 2e-7, output <=1 bf16 ulp.
Committed state is not written (same as disable_state_update). Tree verify / ReplaySSM ring / other shapes fall back to Triton.
Prior trials: A5/A7 (kda_22, kda_11) tuned the decode recurrence; nothing covered the verify kernel.

### analysis.md

Same VibeSim prediction as iter_00 (k3_kda; fixed, no re-simulation available in this container). Target node:
kda_recurrent_decode (R0/R5 5.2 in the decode prediction), which is fused_sigmoid_gating_delta_rule_update_kernel in the verify profile (64.5us of 489us @B=64).

### result.json

```
CHECK {"max_abs_err": 0.15625, "max_rel_err": 0.009881, "mean_rel_err": 0.000381, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.00835, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 64, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 256, "ok": true, "latency_us": 472.54401445388794, "latency_mode": "graph", "us_step": 1215.9680128097534, "us_step_graph": 472.54401445388794, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [64, 8192, "vk3"], "correctness": {"max_abs_err": 0.15625, "max_rel_err": 0.009881422299989104, "mean_rel_err": 0.000381160294637084, "nan": false, "rows": 256, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.008349798619747162, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.1875, "max_rel_err": 0.011811, "mean_rel_err": 0.000572, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.008858, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 128, "seq_len": 8192, "mixed": false, "tag": "vk3", "prefill": false, "kind": "verify", "prefix_len": null, "chunk": null, "draft_tokens": 4, "decode_requests": null, "num_tokens": 512, "ok": true, "latency_us": 620.1279759407043, "latency_mode": "graph", "us_step": 1195.3920125961304, "us_step_graph": 620.1279759407043, "us_attn": null, "us_moe": null, "us_norms": null, "finite": true, "graph_finite": true, "attention_backend": null, "attn_heads": 12, "seed": 0, "error": null, "point": [128, 8192, "vk3"], "correctness": {"max_abs_err": 0.1875, "max_rel_err": 0.011811022878045804, "mean_rel_err": 0.0005720747285522521, "nan": false, "rows": 512, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.008858266286551952, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}}
CHECK {"max_abs_err": 0.09375, "max_rel_err": 0.006637, "mean_rel_err": 0.000136, "nan": false, "rows": 64, "rows_over_tol": 0, "frac_rows_over_tol": 0.0, "p99_row_rel_err": 0.006637, "rule": "max_rel_err<=tol", "state_ok": true, "rel_err_max": 0.02, "pass": true}
JSON {"B": 16, "seq_len": 65536, "m
[... truncated]
```

(diff.patch: 542 lines, files: b/kernels/jit/csrc/attention/kda_verify_recurrent.cuh, b/kernels/ops/attention/kda_verify_recurrent.py, b/srt/layers/attention/linear/kernels/kda_triton.py)

## iter_02
### hypothesis.md

# iter_02: token-major causal_conv1d_update output on the KDA verify path
Hypothesis: mixed_qkv is a slice of the fused in-proj output, so torch.empty_like(x) inside causal_conv1d_update
falls back to a dim-major (batch, dim, T) output; the following transpose(1,2).reshape would then be the 4.3us
direct_copy, and the conv's output stores are 4-element strided. Passing a preallocated token-major `out=` should
drop the copy and coalesce the stores.
Builds on iter_01 (verify recurrence kernel, kept). Ruled out before trying this: store-policy / occupancy / TMA-store
variants of the verify kernel (iter_02/analysis.md).
Result: NEGATIVE. Replay 472.4 / 622.9 / 292.2us (vs 472.5 / 620.1 / 292.3), CHECK pass; the profile still shows the
direct_copy (4.3us) and conv 7.7us, so the copy does not come from this reshape (a TorchDispatchMode trace of the
forward found no Python-dispatched copy). Reverted; diff.patch records the attempted change.

### analysis.md

# iter_02 analysis (after iter_01, B=64 latency 472.5us)
VibeSim k3_kda (fixed decode prediction) unchanged; ladder from the measured profile:
- MoE MXFP4 bmm 169 + 87us: closed trtllm cubin (ruled out in opt_history).
- nvjet GEMMs 56 + 3x16.5 + 28 + 22us: cuBLAS, at roofline for m=256.
- kda_verify_recurrent 46us (38us hot-L2 microbench). Diagnosed with a dirty-L2 microbench (300MB fill before each call):
  45.7us, matching in-layer, i.e. ~4.4-4.9 TB/s on the 201MB fp32 checkpoint stream vs 7.3 TB/s for a plain fill.
  Store policy sweep (st.cs / plain / L1::no_allocate / st.wt): all within 1us. Higher occupancy
  (__launch_bounds__ minBlocks 6/8 -> spills): 49.7 / 58.1us, worse. TMA bulk (cp.async.bulk) checkpoint stores via
  a 64KB smem stage: 58us (3 CTAs/SM), worse. Current kernel kept.
- causal_conv1d_update 7.6us + at::direct_copy 4.3us: tried a token-major conv output (see hypothesis).

### result.json

```
{
 "status": "reverted (no gain)",
 "before_us": {
  "64,8192,vk3": 472.5,
  "128,8192,vk3": 620.1,
  "16,65536,vk3": 292.3
 },
 "results": [
  {
   "point": [
    64,
    8192,
    "vk3"
   ],
   "latency_us": 472.4160134792328,
   "check": {
    "max_abs_err": 0.15625,
    "max_rel_err": 0.009881,
    "mean_rel_err": 0.000381,
    "nan": false,
    "rows": 256,
    "rows_over_tol": 0,
    "frac_rows_over_tol": 0.0,
    "p99_row_rel_err": 0.00835,
    "rule": "max_rel_err<=tol",
    "state_ok": true,
    "rel_err_max": 0.02,
    "pass": true
   }
  },
  {
   "point": [
    128,
    8192,
    "vk3"
   ],
   "latency_us": 622.9119896888733,
   "check": {
    "max_abs_err": 0.1875,
    "max_rel_err": 0.011811,
    "mean_rel_err": 0.000572,
    "nan": false,
    "rows": 512,
    "rows_over_tol": 0,
    "frac_rows_over_tol": 0.0,
    "p99_row_rel_err": 0.008858,
    "rule": "max_rel_err<=tol",
    "state_ok": true,
    "rel_err_max": 0.02,
    "pass": true
   }
  },
  {
   "point": [
    16,
    65536,
    "vk3"
   ],
   "latency_us": 292.1920120716095,
   "check": {
    "max_abs_err": 0.09375,
    "max_rel_err": 0.006637,
    "mean_rel_err": 0.000136,
    "nan": false,
    "rows": 64,
    "rows_over_tol": 0,
    "frac_rows_over_tol": 0.0,
    "p99_row_rel_err": 0.006637,
    "rule": "max_rel_err<=tol",
    "state_ok": true,
    "rel_err_max": 0.02,
    "pass": true
   }
  }
 ]
}
```

(diff.patch: 52 lines, files: kernels/ops/mamba/causal_conv1d_triton.py, srt/layers/attention/linear/kda_backend.py)
