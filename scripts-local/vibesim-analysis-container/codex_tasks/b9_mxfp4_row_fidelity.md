# B9 — close the remaining Kimi-K3 single-layer prediction gap (mxfp4_fused_moe row + glue)

You are in the VibeSim worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust`
(branch `kimi-k3-arch`, HEAD 07d5ee16 = your B8 commit). Read `CLAUDE.md`, then the skills
`operate-run-alignment`, `top-add-kernel`, `impl-register-kernel`, `operate-profile-existing-kernel`.
Work on this branch, commit when green (`just test-cpu`), and END with a short report in English.
No GPU is available to you: never run `kernel-profile run`, `timing-predict` with a cold cache on a
GPU, or docker. If a row must be (re)measured, print the exact command for the operator, who runs it
on GPU 3 via `scripts-local/vibesim-analysis-container/run_k3_jit_fill.sh` (`K3_GPU_INDEX=3`).

## Where we are (post-B8, all rank-1 single-layer presets, B200, branch DB
`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db`)

| layer | point | measured CUDA-graph step (us) | simulated (us) | error |
|---|---|---|---|---|
| KDA | B=1 / 32 / 128 @ L=8k | 179.6 / 310.7 / 556.5 | 141 / 277 / 495 | -21% / -11% / -11% |
| MLA | 128x8k / 1x1M / 16x64k | 680.4 / 345.5 / 357.9 | 569 / 274 / 317 | -16% / -21% / -11% |

Your B8 fixed the qkvbfg leaf (now ~21 us, was 65). The per-op alignment tables in
`doc/alignment/kimi_k3_single_layer.md` say the remaining under-prediction is dominated by:
1. `moe.mxfp4_fused_moe`: measured 384.4 us (KDA, B=128) vs row 342.9 (-10.8%); MLA B=128 421.2 vs
   342.9 (-18.6%). Same row for both layers although the measured numbers differ by 37 us — so the
   row does not depend on something the layer changes (routing distribution? PDL? which of the two
   BMMs? the fused routing/finalize inside `trtllm_fp4_block_scale_moe`?). The runner times ONE call
   with `do_finalize=True`; the layer's kernel table (probe) shows the routed MXFP4 BMMs as two
   launches (284.0 + 143.0 us at B=128) plus routing/finalize kernels.
2. Unmapped ~17 us/step: two `attn_res_fused_tma_kernel` launches (the attention-residual stream,
   `attn_res_block_size=12`) + CUDA-graph glue. Decide: fold into an existing leaf, add an
   `elementwise`-kind leaf with a measured backend, or document as an explicit floor.
3. At B=1 the KDA gap is -21% (38 us): check launch-bound floors — with 26-29 kernels/step in the
   graph, per-kernel minimum durations (~2-3 us each under graph replay) may be below what the rows
   return for m=1; look at how other archs handle the small-m floor.

Evidence available (read-only): probe captures + parsed kernel sequences under
`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/align/{kda,mla}_b128_l8192/`
(`parsed.json`, `kernel_sequences.json`, `metrics.jsonl`), per-kernel torch-profiler tables in
`.../kimi_single_layer/profiles/*.json`, alignment payloads under `logs/alignment_kimi_k3_single_layer_{kda,mla}/`,
and the driver `scripts-local/vibesim-analysis-container/kimi_single_layer_decode.py` (how the
layer is built and routed: seeded random router logits, 112 local experts, top-16 + 2 shared).

## Deliverables
- Root-cause each of the three items with numbers from the evidence; for (1) determine what the
  runner measures differently from the layer (compare `profiling/runners/moe/mxfp4_fused_moe.py`
  arguments/shape/routing/PDL against the driver's real call) and fix the runner and/or the Rust
  kernel model (`simulator/src/timing/kernels/mxfp4_fused_moe.rs`, the K3 MoE worklet) so the SAME
  row explains both layers' measurements, or add the missing input dimension to the kind.
- If (1) needs new rows, keep the change JIT-fillable by the existing presets
  (`presets/predict_kimi_k3_b200_rank1_layer_{kda,mla}.json`) and print the fill command.
- Re-run the CPU side of the alignment pipeline (`launcher alignment analyze` against the cached
  rows) and update the tables in `doc/alignment/kimi_k3_single_layer.md`; state clearly which numbers
  are post-fix vs still waiting on a GPU fill.
- Also answer, without changing analyzer semantics: does the `CostNode::Max` used for the stream
  overlap hide the non-critical leaves (`mla_cache_append`, `output_gate`) from the `optimality`
  ladder that agents read? If yes, propose (in the doc) the smallest analyzer/worklet change that
  keeps their bytes/time visible.
- `just test-cpu` green (the 4 `test_energy_*` failures are a known host issue). Commit.
