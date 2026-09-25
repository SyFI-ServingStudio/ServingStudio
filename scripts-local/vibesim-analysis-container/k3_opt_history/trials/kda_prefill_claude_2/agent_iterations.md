# KDA_PREFILL_CLAUDE trial 2: agent iterations

## agent log.md

# opt log (latency_us: 1x16384pf / 1x16384pf49152 / 4x4096pf)
- iter_00: baseline 8533 / 8742 / 8369 (capture run 8607 / 8636 / 8414)
- iter_01: CUDA kda_chunk_h h-scan (bit-exact, small-grid gate) -> 8402 / 8515 / 8599 (A/B vs gate-off: -150 / -260 / 0 noise); h kernel 516->281us; CHECK pass (0.0)

## iter_00
### hypothesis.md

# iter_00
Baseline only, no edit. The seed tree contains trial kda_prefill_claude_1 (iter_01..04: h-kernel BV16, host-sync removal, attn_res fused add, strided copies removal; verdict PASS 9072->8564).

### analysis.md

# iter_00 baseline analysis
VibeSim prediction id `prediction=.:k3_kda_prefill`; endpoints: simulate, analyze(level=run_summary|operator), optimality(scope=iter), kernels(kernel_set=...). Raw: vs_*.json.

## VibeSim
- run_summary: optimality 0.070, hardware_gap 85%, batching 2.4% (all in kda_chunk_prefill: 1.16 of 2.89 ms), imbalance 5.9%.
- optimality r0/r5: merged_front 3.36, mxfp4 moe 2.97, qkvbfg 6.06/3.17, shared_down 3.35, latent_up 3.15, o_proj 3.08, **kda_chunk_prefill 37.9** (worst ratio, no cached alternative).
- GEMM nodes report has_cached_alternative=true, but microbenchmarks (tools/bench_gemm*.py) of cuBLAS / DeepGEMM bf16 / flashinfer mm_bf16 (cudnn/cutlass/cublaslt) show no faster kernel (all ~1.5-1.6 PFLOPS, cutlass 3-4x slower).

## Measured (profile.json, 1x16384pf, us)
nvjet tst x4 2585 (qkvbfg 875, shared_down 898, latent_up 493, o_proj 215); nvjet tss front fp32 2366; MoE bmm1 929, bmm2 346; **h kernel 516**; attn_res x2 395; situ 253; add3 134; intra 131; conv 130; solve 113; chunk_o 102; recompute_w_u 94; finalize 72; cumsum 50; quant 49.
Timeline serial (span ~= kernel sum). h kernel: grid 8x12=96 CTAs, sequential 256 chunk steps, ~1.6 us/step: tcgen05 MMA + TMEM round trip per tiny dot -> latency bound.

### result.json

```
{"iter": 0, "latency_us": {"1,16384,pf": 8533.2, "1,16384,pf49152": 8742.1, "4,4096,pf": 8369.3},
 "capture_run_us": {"1,16384,pf": 8607.0, "1,16384,pf49152": 8635.6, "4,4096,pf": 8413.6},
 "check": "golden captured from start tree (/tmp/golden_*.pt)"}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis
The KDA chunk h-recurrence (Triton `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`) is a serial 256-step scan with 96 CTAs at N=1. Each step goes through tcgen05/TMEM issue->commit->wait round trips for four tiny dots, so the kernel is latency bound at ~1.6 us/step.

Plan: replace it with a CUDA JIT kernel. The state stays in mma.sync accumulator registers and the chunk operands stream through a cp.async ring. Each dot keeps the Triton reduction order:
- w@h^T is one fp32 chain over K in k16 steps;
- k^T@v is a fresh chain over tokens, added after the exp2 decay with separate mul and add.

With that order the result is bit-identical.

Prior trials:
- kda_prefill_claude_1 iter_01 already retuned the Triton tile (BV16/nw4/ns3, 630->516 us in-layer), and that is in the start tree. It noted that changing num_warps changes the accumulation order. This kernel keeps the order explicitly instead.
- No trial replaced the kernel.

Gate: covered shapes (K=128, gk/exp2, bf16 operands, no track_state) and `cdiv(V,16)*N*H <= #SMs`. Microbenchmarks show CUDA wins only while the grid fits one wave: N=1 223 vs 398 us, but N=4 155 vs 136 us. Everything else stays on Triton.

Expected: about 200-290 us at the primary point, the same at pf49152, and 0 at 4x4096 (Triton path).

### analysis.md

# iter_01 analysis
VibeSim prediction id `prediction=.:k3_kda_prefill`. Endpoints: simulate, analyze(run_summary|operator), optimality(scope=iter), kernels(kernel_set=unified.kda.attention.kda_chunk_prefill). Raw output is in vs_*.json here and in iter_00.

## Node selection
- `unified.kda.attention.kda_chunk_prefill` has r0/r5 = 37.9, the worst in the layer. r0 is 2.89 ms, r5 is 76 us, and there is no cached alternative.
- Its batching gap (1.16 ms) is the only non-hardware bucket in run_summary.
- The GEMM nodes are about 3x r5 and have no faster library kernel (iter_00 microbenchmarks).
- Mapped to source: `kernels/ops/attention/fla/chunk_delta_h.py::chunk_gated_delta_rule_fwd_h`. The largest kernel in the node is the h-scan: 516 us in-layer, grid 8x12, 256 serial steps.

## Kernel work (micro/)
- Microbenchmark of mma.sync on B200 (micro/mmabench): a dependent HMMA chain costs about 23 cycles per HMMA; independent chains reach about 8.5 cycles per HMMA per SMSP.
- The new kernel `csrc/attention/kda_chunk_h.cuh` is 0.0 error against Triton on h, v_new and state (bf16 and fp32 state, varlen tails, index -1). Standalone: N=1 T16384 398 -> 223 us. N=4 is 136 -> 155 us, so the gate keeps N=4 on Triton.
- ncu, per step:
  - about 1700 cycles at 1.1 GHz;
  - the critical path is barrier -> ldmatrix h -> an 8-deep HMMA chain -> u subtract -> STS -> barrier -> a 4-deep HMMA chain -> decay/add;
  - dominant stalls are short_scoreboard and wait.

## Result (profile.json = 1x16384pf, --split diagnostic)
- The h kernel is `kda_chunk_h_kernel<bf16,3,true>`: 281 us, down from 516 us. Total kernel time is 8499 -> 8171 us; launch count is unchanged at 34.
- Interleaved plain A/B, gate off vs on, two runs each:

  | Point | Gate off (us) | Gate on (us) |
  |---|---|---|
  | 1x16384pf | 8525, 8585 | 8386, 8431 |
  | pf49152 | 8736, 8832 | 8502, 8552 |
  | 4x4096 | 8660, 8601 | 8534, 8590 (Triton path, noise) |

### result.json

```
{
 "iter": 1,
 "edit": "CUDA JIT h-scan kernel (kda_chunk_h) for chunk_gated_delta_rule_fwd_h, small-grid gate, Triton fallback",
 "before_us": {"1,16384,pf": 8533.2, "1,16384,pf49152": 8742.1, "4,4096,pf": 8369.3},
 "after_us": {"1,16384,pf": 8402.2, "1,16384,pf49152": 8514.7, "4,4096,pf": 8598.5},
 "after_runs": {"1,16384,pf": [8402.2, 8488.0, 8385.8, 8430.7], "1,16384,pf49152": [8514.7, 8467.8, 8502.4, 8551.6], "4,4096,pf": [8598.5, 8537.4, 8534.1, 8590.3]},
 "ab_gate_off_runs": {"1,16384,pf": [8525.0, 8584.6], "1,16384,pf49152": [8735.8, 8832.2], "4,4096,pf": [8660.1, 8600.8]},
 "check": {"all_pass": true, "max_rel_err": 0.0, "state_ok": true},
 "note": "4x4096 does not take the new path (N*H*V/16=384 > 148 SMs); its before/after difference is run noise (A/B off 8601-8660)"
}
```

(diff.patch: 667 lines, files: b/python/sglang/kernels/jit/csrc/attention/kda_chunk_h.cuh, b/python/sglang/kernels/ops/attention/fla/chunk_delta_h.py, b/python/sglang/kernels/ops/attention/kda_chunk_h.py)
