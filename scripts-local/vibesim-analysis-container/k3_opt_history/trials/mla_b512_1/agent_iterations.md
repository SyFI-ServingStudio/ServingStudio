# MLA_B512 trial 1: agent iterations

## agent log.md

iter_00: baseline plain graph 978.464 / 638.496 / 273.888 us (B=512 / B=256 / B=16 mixed); VibeSim selected the TRT-LLM MXFP4 MoE leaf, and iter_01 tests a large-batch tactic ceiling.

## iter_01
### hypothesis.md

# Iteration 01 hypothesis

The iteration-00 VibeSim analysis ranked `unified.mla.moe.mxfp4_fused_moe`
first, with `R0=776.872 us`, `R5=46.037 us`, `R0/R5=16.88`, and
`necessary_share=31.62%`; its kernel drill-down reported backend
`sglang_trtllm_mxfp4` and no cached alternative. The real profile confirmed
the two dominant TRT-LLM MXFP4 BMM launches at B=512 (`259.548 us` and
`119.356 us`) and B=256 (`179.785 us` and `92.397 us`).

The prior KDA trial `kda_11` tested a larger TRT-LLM tuning ceiling at B=128
and found only noise, while the warm-start notes explicitly identify B>=256
as a new compute-bound regime. This trial re-tests that lever only for the
new B=256/512 decode buckets: pass `2 * next_power_of_2(M)` as
`tune_max_num_tokens` for 256 <= M <= 512, retaining the existing ceiling for
B=16 and for all larger chunked-prefill calls.

The call still receives the same FP32-produced routed rows, MXFP8 activation
quantization, top-k IDs/weights, MXFP4 expert weights/scales, SiTU constants,
and destination buffer. Only TRT-LLM tactic selection changes, so the layer
output, MLA KV writes, and any recurrent state should remain identical; the
full replay will verify those contracts.

### analysis.md

# Iteration 01 pre-edit analysis

Prediction: `p_6b53ccf740d742fd8779d087584820d7`, built by
`simulate?prediction=.:before` for `vibesim:k3_mla_b512`.

VibeSim operator analysis modeled total kernel time at `1.936794 ms`.
The largest operators were `mxfp4_fused_moe` at `0.776872 ms` (40.11%) and
`mla_decode_attention` at `0.686446 ms` (35.44%), followed by merged front at
`0.188123 ms` (9.71%). The run-summary necessary ratio was `0.2436`, with
batching as the largest modeled gap bucket (`47.86%`).

The iteration-level tree modeled the B=512 point at `1025 us`, with MoE at
`558 us` and attention at `467 us`. Optimality ranked the MoE leaf first:
`R0=776.872 us`, `R5=46.037 us`, `R6=245.662 us`, `R7=245.662 us`,
`R0/R5=16.88`, and `necessary_share=0.3162`. Its drill-down shape is
hidden=3584, intermediate=3072, 112 local experts, top-k=2, BF16 input,
MXFP4 E2M1/UE8M0 weights, SiTU, and group size 32; it has no cached
alternative. The MLA leaf has a cached TRT-LLM alternative and lower
headroom (`R0/R5=1.37`), so this iteration targets the MoE tactic selector.

Profiler confirmation: B=512 launches the TRT-LLM MLA kernel at `384.476 us`
and MXFP4 BMMs at `259.548 us` and `119.356 us`; B=256 launches the same
families at `195.654 us`, `179.785 us`, and `92.397 us`. The source call is
`Mxfp4FlashinferTrtllmMoEMethod.apply` ->
`_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen` in
`srt/layers/moe/moe_runner/flashinfer_trtllm.py`.

### result.json

```
{
  "candidate": "large_decode_trtllm_tuning_ceiling",
  "before_latency_us": {"512,8192": 978.464, "256,8192": 638.496, "16,65536,mix": 273.888},
  "after_latency_us": {"512,8192": 3501.632, "256,8192": 640.544, "16,65536,mix": 271.904},
  "checks": {
    "512,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "16,65536,mix": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "verdict": "rejected_performance; rollback_required",
  "smoke_before_rollback": "LAYER_SMOKE_OK"
}
```

## iter_02
### hypothesis.md

# Iteration 02 hypothesis

Iteration 00's actual profile showed the accepted shared-expert overlap is
active: the B=512 graph launches the routed MXFP4 BMMs (`259.548 us` and
`119.356 us`) while the shared-down path is also present, and VibeSim ranks
the MXFP4 fused MoE as the main necessary-work node. The warm-start notes say
the overlap gain shrinks when large batches make both branches compute-bound.
Iteration 01 ruled out a vendor tuning-ceiling change: correctness was exact,
but B=512 regressed from `978.464 us` to `3501.632 us`.

For decode batches of 256 and 512 rows, serialize the shared-down branch after
the fused front and before routed experts, avoiding SM contention between the
large dense BF16 GEMM and the MXFP4 expert GEMMs. Keep the existing side-stream
overlap for smaller decode batches and for all non-decode `ForwardMode.EXTEND`
chunked-prefill calls. The serialized and overlapped schedules consume the
same completed front slices, write disjoint output slices, and join before the
same reduction/tail, so output and MLA KV state should be unchanged.

### analysis.md

# Iteration 02 pre-edit analysis

Prediction remains `p_6b53ccf740d742fd8779d087584820d7`, fetched with
`simulate?prediction=.:before`; the VibeSim service exposes the fixed
`vibesim:k3_mla_b512` before-state prediction. Operator analysis modeled
`mxfp4_fused_moe` at `776.872 us` / `40.11%`, MLA decode attention at
`686.446 us` / `35.44%`, and merged front at `188.123 us` / `9.71%`.

The top optimality leaf remains `unified.mla.moe.mxfp4_fused_moe`:
`R0=776.872 us`, `R5=46.037 us`, `R6=245.662 us`, `R7=245.662 us`,
`R0/R5=16.88`, `necessary_share=0.3162`, with no cached alternative. The
profiled kernel table maps it to the TRT-LLM MXFP4 BMM pair; the shared-down
BF16 BMM is the adjacent source-level schedule branch in
`KimiK3MoE._forward_fused`.

The test is intentionally a dataflow schedule change rather than a kernel
arithmetic change, and it is mode-aware so prefill remains on the existing
overlap path.

### result.json

```
{
  "candidate": "serialize_shared_down_for_large_decode",
  "before_latency_us": {"512,8192": 978.464, "256,8192": 638.496, "16,65536,mix": 273.888},
  "after_latency_us": {"512,8192": 992.736, "256,8192": 644.544, "16,65536,mix": 271.840},
  "checks": {
    "512,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "16,65536,mix": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "verdict": "rejected_performance; rollback_required",
  "smoke_before_rollback": "LAYER_SMOKE_OK"
}
```

## iter_03
### hypothesis.md

# Iteration 03 hypothesis

The measured B=512 profile still contains separate route and MXFP8 preparation
launches (`routingIndicesClusterKernel` about `6.15 us` and
`per_token_group_quant_flat_kernel` about `6.41 us`). The current tree already
has a correctness-checked local 112-expert/top-2 `route_quant_fused` kernel,
but its Python coverage cap is 64 rows, so B=256/512 fall back to that chain.

Prior trial `mla_26` only verified the cap-64 local path at B=1/16 and
explicitly left B=128 on the external chain; prior cap increases were done
before the corrected local specialization and did not reach the live path.
Raise the local fusion cap to 512 so the exact B=256 and B=512 decode shapes
use the already-built route+pack+quant implementation. It preserves FP32
scores, correction-bias ranking, top-2 weights/IDs, packed routing, MXFP8
group scales, and all expert/attention/state tensors; only preparation launch
fusion changes.

### analysis.md

# Iteration 03 pre-edit analysis

Prediction `p_6b53ccf740d742fd8779d087584820d7` was fetched again with
`simulate?prediction=.:before`. VibeSim still ranks
`unified.mla.moe.mxfp4_fused_moe` first at `R0=776.872 us`, `R5=46.037 us`,
`R6=245.662 us`, `R7=245.662 us`, `R0/R5=16.88`, and
`necessary_share=0.3162`, with no cached alternative. Its operator share is
`40.11%`; MLA attention is second at `35.44%` and has a cached alternative.

The actual profile confirms the source-level preparation chain inside the
MoE leaf: the B=512 table has separate routing and quantization kernels, while
the local fused kernel is present only at the cap-64 shapes. The changed file
is `kernels/ops/moe/moe_route_quant_fused.py`; the CUDA route and quant code is
unchanged.

### result.json

```
{
  "candidate": "local_route_quant_cap_64_to_512",
  "before_latency_us": {"512,8192": 978.464, "256,8192": 638.496, "16,65536,mix": 273.888},
  "after_latency_us": {"512,8192": 992.768, "256,8192": 643.552, "16,65536,mix": 282.112},
  "diagnostic_profile_latency_us": {"512,8192": 999.776, "256,8192": 653.728, "16,65536,mix": 285.088},
  "checks": {
    "512,8192": {"max_rel_err": 0.008032128, "state_ok": true, "pass": true},
    "256,8192": {"max_rel_err": 0.008771929, "state_ok": true, "pass": true},
    "16,65536,mix": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "profile_confirmation": {
    "B512_launches": 27,
    "B256_launches": 28,
    "vibesim_prediction_id": "p_6b53ccf740d742fd8779d087584820d7"
  },
  "verdict": "rejected_performance_and_drift; rollback_required",
  "smoke_before_rollback": "LAYER_SMOKE_OK"
}
```

## iter_04
### hypothesis.md

# Iteration 04 hypothesis

The B=512 profile maps the dominant `mxfp4_fused_moe` node to two large
TRT-LLM BMMs. Their standard routed call currently enables PDL through
`trtllm_moe_enable_pdl`; the preceding `kda_11`/`mla_21` history rejected a
global PDL-off policy on smaller decode batches, but did not test the new
B=256/512 compute-bound regime.

Disable PDL only for 256 <= M <= 512 in the standard K3 routed call. The
expert weights, MXFP8 activations/scales, packed top-k IDs/weights, SiTU
epilogue, output pointer, and all layer state remain unchanged; only the
producer/consumer launch dependency mode changes. B=16 and other shapes keep
the inherited PDL setting.

### analysis.md

# Iteration 04 pre-edit analysis

The fixed VibeSim prediction is `p_6b53ccf740d742fd8779d087584820d7`,
re-fetched with `simulate?prediction=.:before`. The operator breakdown is
`mxfp4_fused_moe=776.872 us` / `40.11%`, MLA attention `686.446 us` /
`35.44%`, and merged front `188.123 us` / `9.71%`.

The top roofline leaf remains `unified.mla.moe.mxfp4_fused_moe` with
`R0=776.872 us`, `R5=46.037 us`, `R6=245.662 us`, `R7=245.662 us`,
`R0/R5=16.88`, and `necessary_share=0.3162`; its kernel drill-down has no
cached alternative. The real table confirms the target is the standard
TRT-LLM routed MXFP4 call in `flashinfer_trtllm.py`.

### result.json

```
{
  "candidate": "disable_trtllm_moe_pdl_for_256_512",
  "before_latency_us": {"512,8192": 978.464, "256,8192": 638.496, "16,65536,mix": 273.888},
  "after_latency_us": {"512,8192": 995.840, "256,8192": 650.752, "16,65536,mix": 287.200},
  "checks": {
    "512,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "256,8192": {"max_rel_err": 0.0, "state_ok": true, "pass": true},
    "16,65536,mix": {"max_rel_err": 0.0, "state_ok": true, "pass": true}
  },
  "verdict": "rejected_performance; rollback_required",
  "smoke_before_rollback": "LAYER_SMOKE_OK"
}
```

## iter_05
### hypothesis.md

# Iteration 05 hypothesis

The iteration-00 profile's B=512 table has
`set_mla_kv_concat_q_fp8_kernel<8, true, long>` at `17.862 us`; this is a
source-owned preparation leaf on the MLA path. The tree already contains the
16-warp CUDA instantiation, used for the B=1 13-item case. A large uniform
batch has thousands of independent KV-row/query-head items, so a 16-warp CTA
may reduce CTA scheduling overhead without changing any elementwise operation.

Select 16 warps for total preparation items >=1024, which covers B=256/512
with 12 heads. Keep B=1's existing 16-warp case, B=16 mixed's 8-warp case,
and all prefill calls unchanged. The kernel still performs identical BF16 to
FP8 conversion, KV row writes, query concatenation, and location mapping.

### analysis.md

# Iteration 05 pre-edit analysis

Prediction `p_6b53ccf740d742fd8779d087584820d7` remains the fetched
`vibesim:k3_mla_b512` before-state. VibeSim's largest leaf is still
`unified.mla.moe.mxfp4_fused_moe` (`R0=776.872 us`, `R5=46.037 us`,
`R0/R5=16.88`, `necessary_share=0.3162`, no cached alternative), followed by
MLA decode attention (`R0=686.446 us`, `R5=502.730 us`, cached alternative).

The actual per-kernel table identifies the preparation leaf and its source
selector: `set_mla_kv_concat_q_fp8_kernel<8, true, long>` at B=512 and
`17.094 us` at B=256. This is a narrow scheduling experiment on a small
critical-path leaf; the dominant MoE and attention arithmetic are untouched.

## iter_06
### hypothesis.md

Hypothesis: emit the Kimi-K3 fused MoE front in BF16 for large uniform decode.

The initial VibeSim prediction is p_6b53ccf740d742fd8779d087584820d7. Its
largest leaf is unified.mla.moe.mxfp4_fused_moe (R0=776.872 us, R5=46.037 us,
necessary_share=0.3162, no cached alternative), and the live B=512 profile
shows the routed MXFP8 quantizer consuming an FP32 front slice. The K3
SM100 TRT-LLM path already accepts BF16 routed activations after the front;
the FP32 front is retained for exact router logits, but the routing API
converts BF16 logits to FP32 before selection.

For decode-only batches with at least 256 rows, select the existing BF16-output
variant of the merged front GEMM. This halves front output traffic and lets
the quantizer consume BF16 rows. The change is gated on ForwardMode.DECODE and
does not affect chunked prefill, the B=16 mixed decode point, or the small
decode buckets. It changes representation at a GEMM boundary, so replay must
verify the output and every post-step state against the original golden.

This builds on the live profiler/VibeSim MoE mapping. The earlier
opt_history/kda_25 BF16 MXFP4 activation trial is ruled out because the
installed TRT-LLM build has no MxE2m1 x BF16 activation tactic; this trial
keeps the existing MXFP8 MoE kernel and changes only its upstream front
buffer. opt_history/kda_7 considered the same front boundary but did not
retain a judged change, so this large-decode-only guard is being measured
against the current MLA workload.

### analysis.md

Pre-edit analysis

- VibeSim workspace-info was called first. The fixed initial prediction is
  p_6b53ccf740d742fd8779d087584820d7.
- Operator analysis models total kernel time at 1.936794 ms. The largest
  operators are mxfp4_fused_moe at 0.776872 ms / 40.11%, MLA decode at
  0.686446 ms / 35.44%, and merged_front at 0.188123 ms / 9.71%.
- Run-summary ladder: R0=1.936794 ms, R5=0.471763 ms, R6=R7=0.471763 ms,
  optimality ratio 0.243579; batching is 47.856% and hardware gap is 8.70%.
- Iteration optimality ranks mxfp4_fused_moe first with R0/R5=16.8751 and
  necessary_share=0.3162188; its kernel drill-down reports backend
  sglang_trtllm_mxfp4 and no cached alternative. The profiler maps this to
  the two TRT-LLM MXFP4 BMMs plus the FP32 per-token-group quantizer.
- Source mapping is KimiK3MoE._forward_fused in
  python/sglang/srt/models/kimi_k3.py, which calls the SM100 runner in
  python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py. Only the
  caller-side output dtype is being tested; the closed vendor MoE leaf and
  its routing/weights remain unchanged.

## iter_07
### hypothesis.md

Hypothesis: overlap the independent MLA output-gate projection at B=256/512.

The fixed VibeSim prediction p_6b53ccf740d742fd8779d087584820d7 ranks MLA
decode as the second operator (0.686446 ms / 35.44%) and reports a cached
alternative, while the real profile confirms a separate K3 output-gate
projection and fused sigmoid/multiply in the MLA path. KimiK3MLAAttention
currently permits the alternate-stream gate precompute only through 128
decode rows. The gate GEMM reads the same hidden states as the MLA query
projection and has no dependency on the attention result; it is joined before
the exact gate multiply and o_proj.

Raise the Blackwell MLA gate precompute limit to 512. This changes only CUDA
stream scheduling: the same gate GEMM, sigmoid/multiply kernel, output tensor,
KV writes, and post-step state are retained. B=16 mixed decode already takes
the precompute path, and chunked prefill is not captured, so those paths keep
their existing behavior. Replay must prove output and MLA KV state equality.

Prior opt_history/mla_26 output-gate trial changed the gate kernel block size
and was rejected; this experiment retains the tested 256-thread kernel and
tests the separate large-batch overlap decision. It also builds on the live
profile rather than the old small-batch gate result.

### analysis.md

Pre-edit analysis

- VibeSim workspace-info was already called at loop start. The fixed initial
  prediction is p_6b53ccf740d742fd8779d087584820d7.
- Operator breakdown: mxfp4_fused_moe 0.776872 ms / 40.11%, MLA decode
  0.686446 ms / 35.44%, merged_front 0.188123 ms / 9.71%.
- The MLA decode leaf has R0=686.446 us, R5=502.730 us, R0/R5=1.3654 and
  `has_cached_alternative=true`; its necessary share is lower than the MoE
  leaf, so this is a scheduling probe limited to the newly relevant large
  batches rather than a replacement for the dominant vendor MoE kernel.
- The B=512 profiler maps the MLA-side work to the TRT-LLM FMHA kernel plus
  the K3 gate path. Source mapping is `_precompute_output_gate` and the
  `_gated_o_proj_forward` wrapper in
  `python/sglang/srt/models/kimi_k3.py`. The guard is capture-only, so eager
  chunked prefill remains unchanged.

## iter_08
### hypothesis.md

Hypothesis: extend the existing K3 local route+MXFP8-quant fusion to B=256/512.

The fresh restored plain baseline is B=512 992.736 us, B=256 644.448 us,
and mixed B=16 282.080 us. VibeSim still ranks
unified.mla.moe.mxfp4_fused_moe first (R0=776.872 us, R5=46.037 us,
necessary_share=0.3162, no cached alternative). The current profiler maps
the large-batch front handoff to separate route/pack/quant launches. The
existing local 112-expert/top-2 fused kernel is already exact and accepted for
small decode; extend only its token cap to 512 so the B=256 and B=512 shapes
use the same route, packed IDs, and MXFP8 quantization in one launch.

The earlier opt_run/iter_03 attempt used this same lever and was rejected
against a noisy 978.464 us B=512 reference. This is a workload-specific
re-measurement per the opt_history guidance: it will require a fresh profile,
VibeSim simulate/analyze/optimality/kernels pass, and an identical full replay
against the original golden. The mixed B=16 path remains on the accepted
small-batch kernel. The route/quant kernel preserves the routing contract and
the MoE input representation; output and MLA KV state must remain within the
driver checks.

### analysis.md

Pre-edit analysis

- VibeSim workspace-info was called first and the initial prediction is
  p_6b53ccf740d742fd8779d087584820d7.
- Operator analysis: total modeled kernel time 1.936794 ms; mxfp4 fused MoE
  0.776872 ms / 40.11%, MLA decode 0.686446 ms / 35.44%, merged front
  0.188123 ms / 9.71%.
- Run-summary: R0=1.936794 ms, R5=R6=R7=0.471763 ms, optimality ratio
  0.243579; batching gap 47.856%, hardware gap 8.70%.
- MoE optimality: R0/R5=16.8751 and necessary_share=0.3162188, with no
  cached alternative. Kernel drill-down identifies the production 112-local-
  expert/top-2 MXFP4 SiTU TRT-LLM leaf. The accepted local route+quant source
  is `moe_route_quant_fused.py` plus its JIT CUDA implementation and the K3
  TopK handoff.
- The fresh baseline is a plain fixed-flag timing only; the post-edit
  `--profile-kernels` run will be saved here and routed through all VibeSim
  analysis verbs before interpreting the result.
