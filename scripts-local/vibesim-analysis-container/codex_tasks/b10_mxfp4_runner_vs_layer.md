# B10 — make the `mxfp4_fused_moe` runner reproduce the Kimi-K3 layer's real MoE launch (GPU microbench)

Worktree: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust` (branch `kimi-k3-arch`,
HEAD b088a6eb = your B9). Read `CLAUDE.md`, skills `operate-profile-existing-kernel`,
`impl-validate-kernel-cache`, `top-add-kernel`. Commit when green (`just test-cpu`), report in English.

## GPU rules (hard)
You may use exactly ONE GPU: NVIDIA B200 index 3, UUID `GPU-019267a2-092a-6798-3cc5-57ffc761e004`.
- Only through VibeSim's profiling env: `export VIBESIM_PROFILE_GPUS=GPU-019267a2-092a-6798-3cc5-57ffc761e004
  VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db
  TMPDIR=/raid/tmp/yilegu_k3_tmp` (the `sglang_k3_env` docker env binds that UUID). For ad-hoc
  scripts use `docker run --rm --gpus '"device=GPU-019267a2-092a-6798-3cc5-57ffc761e004"' --shm-size 32g
  -v /raid/yilegu/roofline_guided_agent/VibeSimWorkspace:/ws -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer
  lmsysorg/sglang:v0.5.20 python3 ...`. Never `--gpus all`, never other indices/UUIDs, never slurm.
- Before each GPU step check `nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader` < 1000 MiB.
- Do not write to `profiling/profile.db`; only the branch DB above.

## The discrepancy (B9's fix went from 18% under to 37% over)
Rank-1 KDA layer, B=128 tokens, 112 local experts, top-16, MXFP4 (`flashinfer_mxfp4`), CUDA-graph replay.
Measured per step (torch profiler on the graph, `scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/kda_prod_B128_L8192.json`):
`bmm_MxE4m3_MxE2m1MxE4m3_Fp32_...t128x16x256u2_s3...` 264.8 us + `bmm_Bfloat16_MxE2m1MxE4m3_...` 134.7 us +
`finalizeKernelVecLoad` 10.2 us + `routingIndicesClusterKernel` 7.7 us ≈ **418 us** (alignment leaf 384 us).
Branch-DB rows for the same kind (`mxfp4_fused_moe`, B200, num_tokens=128): **num_experts=896 → 342.9 us**
(pre-B9, only 258 of 2048 assignments local) and **num_experts=112 → 527.1 us** (your B9 row, per_expert_batches
uniform 22/21). Post-B9 alignment: KDA leaf +37% (layer total +22.8%), MLA leaf +25% (total +11.1%).
So neither runner configuration launches what the layer launches. Candidates to test, each as a controlled
A/B on GPU 3 with the exact layer call as the reference:
1. Routing distribution: the layer routes 2048 seeded-random assignments (Poisson-like loads, mean 18.3) —
   the runner pins uniform 22/21 (sum 2368 ≠ 2048?). Check the sum, the padding to the 16-row tile, and
   whether `per_expert_batches` should be derived from the driver's actual histogram (the driver
   `scripts-local/vibesim-analysis-container/kimi_single_layer_decode.py` can dump it; add a flag if needed).
2. Tactic/autotune: the layer's kernel names encode `t128x16x256u2_s3_et128x16`; compare with the runner's
   launched kernel names (torch profiler or nsys inside the same container). Both pass
   `tune_max_num_tokens=next_pow2(128)`; check whether the layer runs under FlashInfer's `autotune()` context
   (sglang `flashinfer_trtllm.py`) while the runner does not, or vice versa.
3. Activation/input format and `do_finalize`, `routing_method_type` (layer: DeepSeekV3 via `kimi_k3.py:482`),
   `routed_scaling_factor`, `gemm1_alpha/clamp`, `n_group/topk_group`, weight layout/shuffle
   (`sglang.srt.layers.quantization.mxfp4` post-load) — any of these changes the tactic or adds kernels.
4. PDL / graph vs eager: the runner times an eager loop; the layer number is a graph replay. Quantify the
   difference for the same call.
Reference implementation of the layer's call: `sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` +
`KimiK3MoE._forward_fused` in `sglang/srt/models/kimi_k3.py` (pristine v0.5.20 tree at
`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/iter_opt_eval_k3_kda/pristine_tree/`).

## Deliverables
- A table: for each candidate, runner-as-is vs modified, kernel names + us, vs the layer's 418 us reference.
- Fix `profiling/runners/moe/mxfp4_fused_moe.py` (and the Rust kind/worklet if an input dimension is
  missing) so the row for the rank-1 K3 call is within 10% of the layer's measurement at B=128 and B=32,
  and refill the 112-expert rows the rank-1 presets need (`presets/predict_kimi_k3_b200_rank1_layer_{kda,mla}.json`
  via `uv run --no-sync python -m launcher timing-predict <preset>` with the env above). Then re-run
  `launcher alignment timing-predict` + `alignment analyze` for
  `presets/alignment/kimi_k3_single_layer/{kda,mla}_{timing_predict,analyze}.yaml` and update the tables in
  `doc/alignment/kimi_k3_single_layer.md` (state clearly what is measured post-fix).
- Also confirmed by the post-B9 analyze: `mla_cache_append`, `output_gate`, `kv_a_layernorm` show −100%
  (hidden by the `CostNode::Max` attribution). Implement the smallest change that keeps them visible in the
  per-op alignment and the `optimality` ladder without breaking the critical-path total (e.g. attribute
  the Max's children proportionally, or model the overlap on the worklet's non-critical side as a labeled
  zero-critical-time leaf), with tests.
- `just test-cpu` green (4 known `test_energy_*` host failures). Commit on `kimi-k3-arch`.
