# B12c — Kimi-K3 chunked-prefill prediction fidelity: prefix-length scaling + GEMM/MoE prefill rows

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust` (branch `kimi-k3-arch`; HEAD = B12
+ the long-context presets `presets/predict_kimi_k3_b200_rank1_layer_{kda,mla}_lcprefill*.json` and their baked
predictions under `logs/`). Read `CLAUDE.md`, then the skills `top-add-kernel`, `impl-register-kernel`,
`impl-wire-kernel-to-rust`, `impl-compose-worklet`, `operate-run-timing-predict`, `operate-run-alignment`.
Commit when green (`just test-cpu`; 4 known `test_energy_*` host failures), English messages, report in English.

## GPU rules (hard) — changed since B12
GPU work goes ONLY through slurm partition `main`, one job per profiling step, never directly on a device:
```
sbatch --wait --partition=main --job-name=b12c_<what> --output=/raid/tmp/yilegu_k3_tmp/b12c_<what>.out \
  /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/slurm_gpu.sh \
  /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/k3_slurm_step.sh \
  <your command>
```
`slurm_gpu.sh` exports `SLURM_GPU_UUID` / `DOCKER_GPU_ARG` for the granted device; inside the job set
`VIBESIM_PROFILE_GPUS=$SLURM_GPU_UUID`. Ready-made: `run_k3_jit_fill.sh <preset...>` in the same directory runs
`timing-predict` (JIT-filling rows) that way — `sbatch --wait ... slurm_gpu.sh k3_slurm_step.sh
<dir>/run_k3_jit_fill.sh presets/<p>.json`. Always
`VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db`
(branch DB; never the shared `profiling/profile.db`), `TMPDIR=/raid/tmp/yilegu_k3_tmp`. Never `--gpus all`,
never a GPU index by hand, never kill anything. Jobs are limited to 2 h; keep each fill short.

## The problem (measured vs predicted, pristine-equivalent tree, one B200, eager prefill step)
Points are `B,L,pf<prefix>` = 1 request × L new tokens on a `prefix`-token cached context. Measured step medians
(judge, 5 reps) and the current `timing-predict` totals (`logs/predict_kimi_k3_b200_rank1_layer_*_lcprefill/reports/iter_breakdown.ans`):

| layer | point | measured step | predicted total | predicted composition (wrong parts in bold) |
|---|---|---|---|---|
| MLA | 16k @ prefix 245,760 | 27.69 ms | 24.61 ms | **mla_prefill_attention_prefix 9394 µs** (measured prefix FMHA 16.7 ms), **merged_front_prefill 5609** (measured ≈2.3 ms), **mxfp4_fused_moe_prefill 2861** (measured ≈1.2 ms) |
| MLA | 16k @ prefix 131,072 | 18.37 ms | 24.61 ms | **prefix attention 9394 µs — identical to the 245k case** (measured 8.9 ms: it halves with the prefix) |
| MLA | 32k @ prefix 229,376 | 48.36 ms | 46.34 ms | prefix attention 17967 (scales with the chunk only) |
| MLA | 32k @ prefix 131,072 | 35.09 ms | 46.34 ms | same total as 229k |
| KDA | 16k @ prefix 245,760 | 9.03 ms | 16.88 ms | **merged_front_prefill 5609**, **mxfp4_fused_moe_prefill 2861**, kda_chunk_prefill 1445 (measured chunk kernels ≈0.7 ms) |
| KDA | 32k @ prefix 229,376 | 17.94 ms | 33.54 ms | ×2 of the above |
| KDA | 16k @ prefix 131,072 | 9.20 ms | 16.70 ms | |
| KDA | 16k first chunk (prefix 0), earlier campaign | 9.07 ms | 16.49 ms (`..._kda_prefill`) | same GEMM/MoE overestimate — it was there in B12 already |

Measured per-kernel tables (driver `--profile-kernels`, read-only): 
`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/iter_opt_eval_k3_{mla,kda}_lcprefill_claude/trial_1_opt_run/iter_00/profile_B1_L16384pf{245760,131072}.json`
(fields `kernels[].{name,count,total_us}`). Key rows at MLA 16k@245k: `fmhaSm100aKernel_…DenseVarSeq…PersistentContext`
16,695 µs ×2 (prefix attention, two prefix chunks), causal FMHA 645 µs, `nvjet_sm100_tss_128x256…` 2,259 µs ×1
(merged front, fp32 out), `nvjet…tst_128x256…` 2,124 µs ×6 (the other projections / shared experts),
`bmm_MxE4m3_MxE2m1MxE4m3…` 855 µs + `bmm_Bfloat16_MxE2m1…` 322 µs (MXFP4 MoE), `attn_res_fused_tma` 352 µs ×2.
At 128k prefix the prefix FMHA is 8,914 µs ×1. KDA 16k: nvjet 2,473 ×4 + 2,279 ×1, MoE bmm 888 + 333,
`chunk_gated_delta_rule_fwd_kernel_h` 479, `chunk_kda_fwd_kernel_intra_sub_chunk` 121, `_causal_conv1d_fwd_kernel` 122.
Pristine-tree profiles of the ≤48k points: `.../kimi_single_layer/profiles/prefill_{kda,mla}_*.json`.

## Tasks
1. **Prefix-length scaling of `mla_prefill_attention` (prefix branch).** The prefix pass attends L query tokens
   against `prefix` cached tokens (non-causal, fp8 K/V from the latent cache, `kv_b_proj` per prefix chunk); its
   FLOPs and bytes scale with `L × prefix`, and sglang issues one launch per prefix chunk (chunk size from the
   scheduler; the harness plants `num_prefix_chunks` = ceil(prefix / prefix_chunk_len)). Make the kind's args carry
   the prefix length (and prefix chunk count) and the runner profile it, or model it from the causal-row cost with an
   explicit `L × prefix / (L × L / 2)` ratio if rows at 128k/245k are too slow to fill — state which. Target: 8.9 ms
   at 128k, 16.7 ms at 245k for L = 16k (±15%), scaling to L = 32k.
2. **GEMM prefill rows.** `merged_front_prefill` (`gemm_fp32_output`, m = 16384/32768 × k 7168 × n 22000) is
   predicted 5,609 µs vs 2,259 measured (cuBLAS `nvjet tss` fp32-out on B200 ≈ 2.3 PFLOP/s effective); the bf16
   projections (`single_gemm` rows at these m) likewise. Find why the rows are ~2.4× slow (wrong backend in the
   runner — e.g. a torch fp32 GEMM instead of the sglang/cuBLAS bf16→fp32 path — a stale row, or a shape bucket
   mismatch) and fix the runner/rows. Re-profile only the shapes the four presets need.
3. **`mxfp4_fused_moe_prefill`.** Predicted 2,861 µs vs measured ≈1,180 µs (two `bmm_Mx…` kernels + routing glue)
   for 16k tokens × local top-2 = 32k rows over 112 experts. Check the prefill runner's row (expert-popularity
   histogram, batch of rows per expert, whether it includes activation quant + finalize twice) and fix.
4. **KDA `kda_chunk_prefill`** 1,445 µs vs measured ≈720 µs (h-scan 479 + intra 121 + conv 122): check what the
   runner times (it may include the unfused l2norm/cumsum glue that the measured tree already removed — if so keep
   the row but say so in the report; do not tune to the optimized tree).
5. Re-run `timing-predict` for all six prefill presets (`*_prefill`, `*_lcprefill`, both layers) through slurm,
   compare per case against the measured column above (and the ≤48k measured steps: KDA 9.07 / 8.87 / 8.57 ms,
   MLA 12.12 / 8.13 / 7.79 ms for the three `*_prefill` cases). Acceptance: every case within ±15% AND the ranking
   of the top-3 nodes matches the measured tables (prefix attention first for MLA long context, GEMMs first for
   KDA). Put the comparison table in the commit message or a short `doc/` note.
6. Decode predictions must stay bit-identical (`just test-cpu` green; do not touch decode rows/branches).
7. Commit on `kimi-k3-arch`. The operator re-bakes the oracle image (`build_context_k3.sh`, BAKE map already lists
   `k3_{kda,mla}_lcprefill`) and restarts the 8805–8808 containers — do not run docker builds yourself.

Time box: 3 hours of wall clock. If a task cannot be finished, commit what is green and leave a `B12C_STATUS.md`
in the worktree root with what remains and why.
