# MLA trial 4: agent iterations

## agent log.md

iter_00 baseline: plain graph 680.416/345.536/357.920 us for 128x8192/1x1048576/16x65536; VibeSim ranked unified.mla.moe.mxfp4_fused_moe first.
iter_01 shared/routed alternate-stream overlap: replay 672.192/347.520/359.840 us, all CHECKs passed, rejected below target and regressed long contexts.
iter_06 route+quant fused-token cap 64->128: replay 676.288/305.664/356.704 us, all CHECKs passed, profile showed the fused route kernel was not launched; reverted.
iter_07 TRT-LLM B=128 tactic bucket 128->256: replay 676.320/305.600/356.704 us, all CHECKs passed, no kernel or latency change; reverted.
iter_08 MLA CUTLASS backend: smoke rejected by FlashInfer (backend not supported), restored TRT-LLM and passed smoke; no profile.

## iter_00
### hypothesis.md

# Baseline hypothesis for Iteration 01

The fused K3 front produces `gate_up`, `router_logits`, and `routed_input` before either tail. The shared-expert activation/down projection writes a disjoint slice of the pair buffer, while routed MXFP4 experts write the latent slice. Launching the shared branch on the existing alternate stream and joining before the pair is consumed should preserve every operation and value while reducing the critical path.

### analysis.md

# Iteration 00: baseline

- Measured source tree: pristine checkout; diagnostic profile in `profile.json` and point profiles.
- Required simulation call: `simulate?prediction=.:before` returned fixed run `k3_kda`, prediction id `p_38ccb83a6c4e44298f45b2251bd4cf54`. The MLA run was selected with the workspace handle `.:k3_mla` because the endpoint ignored `k3_mla` in the simulate selector; its workspace-info id was `p_c7687e585abe4b5fbacdc89caf0c9761`.
- MLA VibeSim run summary: R0 `0.001364681324` GPU-s, optimality ratio `0.200057`; batching was `73.68%` of R0.
- Operator analysis ranked `unified.mla.moe.mxfp4_fused_moe` first at `0.527255 ms` / `38.64%`, followed by MLA decode at `0.358836 ms` / `26.29%` and merged front at `0.158805 ms` / `11.64%`.
- Optimality for `unified.mla.moe.mxfp4_fused_moe` (`iter/0/0/1/3`): R0 `527.255 us`, R5 `8.955 us`, R6/R7 `227.756 us`, R0/R5 `58.88`, necessary share `43.20%`; no cached alternative.
- MLA decode (`iter/0/0/0/7`) had R0 `358.836 us`, R5 `227.559 us`, R6/R7 `454.164 us`, and a cached `sglang_trtllm_mla` alternative.
- Live B=128 profile confirmed the expected launches: MXFP4 expert BMMs `283.801 us` and `142.868 us`, CuTeDSL MLA `108.122 us`, 29 launches, kernel sum `755.471 us`.

The change target was the MXFP4 MoE. The first hypothesis was to overlap the independent shared-expert tail with the routed expert path after the fused front.

(diff.patch: 1 lines, files: )

## iter_01
### hypothesis.md

# Hypothesis

After the fused front, `shared_experts` and the routed MXFP4 path have disjoint outputs. Running `_forward_shared` on `alt_stream` and `_forward_routed` on the main stream, followed by the existing stream join, should change scheduling only. The same kernels, inputs, weights, reduction, norm, and output buffers are used, so numerical output and post-step state should remain identical.

The diagnostic profile showed the shared-down work is small relative to the routed MoE, so the expected gain was limited. The plain replay confirmed that assessment.

### analysis.md

# Iteration 01: shared/routed overlap

- Prediction build/fetch: `simulate?prediction=.:before` returned the fixed KDA id `p_38ccb83a6c4e44298f45b2251bd4cf54`; MLA analysis used `.:k3_mla` / `/opt/vibesim/repo/logs/k3_mla` as in Iteration 00.
- The required `analyze` verbs and `optimality?prediction=.:k3_mla&scope=iter` were rerun after the profile. The top node stayed `unified.mla.moe.mxfp4_fused_moe`: R0 `527.255 us`, R5 `8.955 us`, R6/R7 `227.756 us`, necessary share `43.20%`, no cached alternative.
- The live profile still launched 29 kernels; diagnostic B=128 graph latency was `695.744 us`, with split MoE `693.760 us`.
- The kernel drill-down still identified `sglang_trtllm_mxfp4` for MXFP4 and `sglang_cutedsl_mla` for MLA; the latter still reported cached `sglang_trtllm_mla`.

The overlap did not change VibeSim's modeled bottleneck and did not meet the required speedup, so this iteration is not accepted as the final optimization.

### result.json

```
{
  "baseline_plain_graph_us": {
    "128,8192": 680.4159879684448,
    "1,1048576": 345.5359935760498,
    "16,65536": 357.91999101638794
  },
  "replay_json": [
    {"B": 128, "seq_len": 8192, "latency_us": 672.1919775009155, "latency_mode": "graph"},
    {"B": 1, "seq_len": 1048576, "latency_us": 347.51999378204346, "latency_mode": "graph"},
    {"B": 16, "seq_len": 65536, "latency_us": 359.8400056362152, "latency_mode": "graph"}
  ],
  "checks": [
    {"point": [128, 8192], "max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": [1, 1048576], "max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"point": [16, 65536], "max_abs_err": 0.0, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "accepted": false,
  "reason": "1.21% B=128 improvement in this replay, below the 5% target, with long-context regressions"
}
```

(diff.patch: 25 lines, files: /workspace/opt_run/iter_01/current_source/kimi_k3.py)

## iter_06
### hypothesis.md

The fused route+quant helper was capped at 64 tokens while the scored batch is 128. Extending its coverage to 128 could replace route selection, packed-id construction, and activation quantization with one launch, while preserving bits because the kernel implementation and all inputs were unchanged. The profile showed that this coverage predicate was not reached for the actual layer tensors, so there was no performance or correctness benefit. The source was restored after the required smoke and replay checks.

### analysis.md

## Iteration 06: route+quant fused-token cap

- Profile: `profile.json` is the diagnostic run from the fixed three-point command; B=128 still launched 29 kernels and the two routed MXFP4 BMMs measured 284.963 us and 143.088 us.
- VibeSim build/fetch: `simulate?prediction=.:before` returned the fixed KDA prediction `p_38ccb83a6c4e44298f45b2251bd4cf54`; the MLA analysis handle was `.:k3_mla` (`/opt/vibesim/repo/logs/k3_mla`).
- Re-ran `analyze` at `operator`, `run_summary`, and `iteration`, plus `optimality?scope=iter` and the `kernels` drill-downs for `mxfp4_fused_moe` and `mla_decode_attention`.
- VibeSim continued to rank `unified.mla.moe.mxfp4_fused_moe` first: path `iter/0/0/1/3`, R0 `527.255 us`, R5 `8.955 us`, R6/R7 `227.756 us`, R0/R5 `58.879`, necessary share `43.20%`, and no cached alternative. MLA decode remained second at R0 `358.836 us`, R5 `227.559 us`, R6/R7 `454.164 us`, with the TRT-LLM alternative cached.
- The source cap change did not activate `moe_route_quant_fused` in the actual 128-token run: the profile had no fused route kernel and remained 29 launches. Plain replay was `676.288 / 305.664 / 356.704 us` for `128x8192 / 1x1048576 / 16x65536`, so the candidate was rejected and reverted.

### result.json

```
{
  "candidate": "route_quant_fused_max_tokens_128",
  "replay": [
    {"B": 128, "seq_len": 8192, "latency_us": 676.2880086898804, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 1, "seq_len": 1048576, "latency_us": 305.6640028953552, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 16, "seq_len": 65536, "latency_us": 356.7039966583252, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "decision": "rejected_no_activation_and_no_qualifying_speedup"
}
```

(diff.patch: 0 lines, files: )

## iter_07
### hypothesis.md

The two dominant TRT-LLM MXFP4 BMMs are selected using a maximum-token tuning bucket. Supplying a 256-token bucket only for the B=128 graph might select a more suitable Blackwell CTA/tile configuration without changing inputs, weights, routing, or arithmetic. The runtime accepted the hint and exact replay checks passed, but the kernel table and latency were unchanged, so the source was restored.

### analysis.md

## Iteration 07: TRT-LLM tactic bucket

- Profile: `profile.json` is the fixed diagnostic profile; B=128 remained 29 launches and measured graph `699.840 us` with the same dominant routed BMM pair.
- VibeSim build/fetch: `simulate?prediction=.:before` returned `p_38ccb83a6c4e44298f45b2251bd4cf54`; MLA analysis used `.:k3_mla`. I reran `analyze` at all three levels, `optimality`, and both MoE/MLA `kernels` drill-downs.
- The prediction remained unchanged: `unified.mla.moe.mxfp4_fused_moe` was R0 `527.255 us`, R5 `8.955 us`, R6/R7 `227.756 us`, necessary share `43.20%`, with no cached alternative. The MLA node remained the only cached backend alternative.
- Forcing the B=128 standard routed call to `tune_max_num_tokens=256` did not alter the launched kernels or graph latency. Plain replay was `676.320 / 305.600 / 356.704 us`; all checks passed, so the hint was reverted.

### result.json

```
{
  "candidate": "trtllm_tune_max_num_tokens_256_for_b128",
  "replay": [
    {"B": 128, "seq_len": 8192, "latency_us": 676.3200163841248, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 1, "seq_len": 1048576, "latency_us": 305.59998750686646, "max_rel_err": 0.0, "state_ok": true, "pass": true},
    {"B": 16, "seq_len": 65536, "latency_us": 356.7039966583252, "max_rel_err": 0.0, "state_ok": true, "pass": true}
  ],
  "decision": "rejected_no_kernel_or_latency_change"
}
```

(diff.patch: 0 lines, files: )

## iter_08
### hypothesis.md

FlashInfer exposes multiple MLA decode backend names. CUTLASS might have provided a faster B=128 decode kernel while preserving the same mathematical operation. This build does not support that backend name for `trtllm_batch_decode_with_kv_cache_mla`, so the hypothesis was invalidated before timing or correctness replay.

### analysis.md

## Iteration 08: unsupported MLA CUTLASS backend

- Hypothesis was tested by changing only the ordinary `CuteDslMLABackend` decode backend from `trtllm-gen` to `cutlass`.
- The mandatory smoke reached the MLA decode call and failed with `ValueError: Backend cutlass not supported` from FlashInfer 0.6.18. The driver still exits zero after reporting a failed point, so the traceback was treated as a failed smoke.
- No valid layer profile was produced and no VibeSim analysis was claimed for this failed build. The supported TRT-LLM backend was restored and the smoke then passed.

### result.json

```
{
  "candidate": "cutlass_mla_decode_backend",
  "smoke": {"pass": false, "error": "ValueError: Backend cutlass not supported"},
  "decision": "reverted_before_profile"
}
```

(diff.patch: 0 lines, files: )
