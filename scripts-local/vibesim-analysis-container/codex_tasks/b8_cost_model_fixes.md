# Task: close the top K3 cost-model gaps found by alignment (branch `kimi-k3-arch`)

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust`, branch `kimi-k3-arch` (do not cd
to `.../main`). CPU only; `uv run --no-sync ...`; `cargo` at `~/.cargo/bin`; `just test-cpu` is the gate (the
4 `test_energy_*` failures are pre-existing). `VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db`
for every launcher/analyzer command. Commit on `kimi-k3-arch` with trailer
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Skills: `impl-compose-worklet`, `impl-compose-op`,
`operate-run-alignment`, `top-align-with-framework`. Read `doc/alignment/kimi_k3_single_layer.md` (the
alignment you are fixing) and `presets/alignment/kimi_k3_single_layer/*`.

## Findings to fix (KDA rank-1 probe, B=128 @ kv 8192, measured vs simulated per step)
1. `unified.kda.attention.qkvbfg_a_proj`: measured 21.8 us vs simulated 64.8 us (+198%). Two causes: (a) the
   leaf costs ONE `single_gemm` with the fused-QKVG shape plus the `[f_a|b]` GEMV serially, while sglang runs
   `fused_qkvg_proj` (7168 -> 4*1536 at 12 heads) and the small `[f_a|b]` GEMV (7168 -> 128+12(+pad)) on a
   side stream (`_bfa_alt_stream`, engaged under CUDA-graph capture: `kimi_k3.py:1943-1951`) — model them as
   two leaves with the side-stream one overlapped (the worklet/op layer has an overlap construct — find how
   other worklets express concurrent streams; if none, cost the small GEMV at zero-critical-path with the
   bytes still accounted and document); (b) confirm the exact production shapes from the measured kernel
   table `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/kda_prod_v3_bf16state_B128_L8192.json`
   (nvjet GEMM names + the driver source `kimi_single_layer_decode.py`) and make the cost row shape match
   (n=6144,k=7168 for q/k/v/g at 12 heads, i.e. 4*12*128; check what the current leaf requests — 7692 or 16768
   rows were seen in the branch DB).
2. `unified.kda.moe.mxfp4_fused_moe`: −10.8% (sim 342.9 vs meas 384.4 us). The profiler row assumes uniform
   `per_expert_batches`; the real step also includes the routing/finalize kernels the fused MoE launches
   (`routingIndicesClusterKernel`, `finalizeKernelVecLoad` ~10 us, the bf16 `bmm_Bfloat16_MxE2m1` gemm2 145 us
   + the fp8 gemm1 288 us). Check the `mxfp4_fused_moe` runner times the same set of kernels as one launch
   group the way sglang calls `trtllm_fp4_block_scale_moe` (it should: one call); if the gap is routing
   skew, add the skewed-popularity case the arch already supports (`routing`/ppm) or document why uniform.
3. `unified.kda.attention.kda_recurrent_decode`: +18% (sim 30.1 vs meas 25.5 us) and `kda_conv_decode` +20%:
   the cached rows were measured at a different shape/state dtype than the probe (bf16 state, B=128, 12
   heads, 4608 channels). Verify the row keys the arch looks up match the probe exactly; if the runner's
   input layout differs from the real call (e.g. `a`/`b` gate tensors, `cache_indices` mapping), fix it.
4. MLA probe: sim 584.6 vs meas 677.5 us (−13.7%) — identify the two largest per-op gaps from the MLA
   alignment payloads and fix them the same way if they fall into the classes above.

## Verify
Re-run the KDA and MLA alignment (`launcher alignment analyze presets/alignment/kimi_k3_single_layer/{kda,mla}_analyze.yaml`
after re-predicting with the warm branch DB; if a new row shape is needed and cannot be measured here on
CPU, leave the exact `kernel-profile run` command in the report for the operator). Target: per-op |error|
≤ 20% on every leaf ≥ 5 us, duration-weighted |error| ≤ 10% for both probes. Report the new table.
