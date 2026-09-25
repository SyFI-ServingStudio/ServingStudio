# B12 — Kimi-K3 chunked-prefill support: kernel kinds, worklet prefill branches, rank-1 prefill presets

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust` (branch `kimi-k3-arch`,
HEAD 51810876 = B11 + the b512 presets). Read `CLAUDE.md`, then the skills `top-split-model-into-kernels`,
`top-add-kernel`, `orchestrator-add-kernel-to-python-profile`, `impl-register-kernel`, `impl-wire-kernel-to-rust`,
`impl-compose-worklet`, `impl-wire-new-arch`, `operate-run-timing-predict`. Commit when green (`just test-cpu`;
4 known `test_energy_*` host failures), report in English.

## GPU rules (hard)
Only GPU index 7 / UUID `GPU-c88e489a-0693-2c29-a1e3-30952377f742` (user-authorized 2026-09-24; SHARED with
the user's other runs and with our optimizer-loop trials — do not wait for it to be idle, do not kill anything
on it), only via `VIBESIM_PROFILE_GPUS=GPU-c88e489a-0693-2c29-a1e3-30952377f742
VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db
TMPDIR=/raid/tmp/yilegu_k3_tmp` or `docker run --gpus '"device=GPU-c88e489a-0693-2c29-a1e3-30952377f742"'`.
Never other GPUs (0-6), never `--gpus all`, never slurm, never the shared `profiling/profile.db`. Expect a few %
timing noise from co-tenants; use medians.

## Context
The optimizer loop so far only exercised DECODE (`kimi_k3_sglang` arch, rank-1 KDA/MLA layer presets). The
harness now also drives CHUNKED PREFILL of the same layer (sglang v0.5.20, one B200 = one rank of the cookbook
TP8/EP8 recipe: 12 heads, 112 local MXFP4 experts with the local top-2 share, fp8-e4m3 latent KV, bf16 KDA
state). Driver: `scripts-local/vibesim-analysis-container/kimi_single_layer_decode.py` (workspace branch
`kimi-k3-loop`), point syntax `B,L,pf[<prefix>]` = B requests x L NEW tokens each on a <prefix>-token cached
context, `ForwardMode.EXTEND`, timed eagerly (sglang does not graph-capture prefill).

What sglang runs per prefill step (all file:line in the sglang tree
`scripts-local/vibesim-analysis-container/iter_opt_eval_k3_mla/pristine_tree/`, read-only):
- KDA layer: `KimiK3DeltaAttention.forward` (srt/models/kimi_k3.py:1992-2055) -> `forward_qkvbfg` (one wide
  GEMM + side GEMV, as in decode) -> `KDAAttnBackend.forward_extend` (srt/layers/attention/linear/kda_backend.py:
  795-953): `causal_conv1d_fn` over the T tokens with the conv state (kernels/ops/mamba/causal_conv1d_triton.py),
  `chunk_kda` (kernels/ops/attention/fla/kda.py:1202-1249 -> chunk_kda_fwd 1083-1199: l2norm, cumsum/gating,
  chunk_gated_delta_rule_fwd_h with the per-request initial state, output kernel) then the Python o_norm
  (gated RMSNorm, T rows) and o_proj. Then the MoE at T tokens x local top-2 = 2T routed rows (32k rows for a
  16k chunk) + shared experts + latent up/down.
- MLA layer: `KimiK3MLAAttention` -> `DeepseekV2AttentionMLA` with the trtllm_mla handler ->
  `AttnForwardMethod.MHA_CHUNKED_KV` (srt/models/deepseek_common/attention_backend_handler.py:171-181,
  attention_forward_methods/forward_mha.py:179-393): q/kv projections at T rows, fp8 latent-KV write of the T new
  rows (`set_mla_kv_buffer_triton`), `trtllm_ragged_attention_deepseek` (flashinfer; fp8 q/k/v because the KV
  cache is fp8: srt/layers/attention/trtllm_mla_backend.py:1085-1128) causal over the chunk, and for prefix > 0
  one pass per prefix chunk (gather the cached latent rows, `kv_b_proj` GEMM to K/V, non-causal ragged attention,
  `merge_state`); prefix chunk length = 128k // batch (forward_batch_deepseek_mha_mixin.py:162-251).

Measured evidence (GPU 7, eager, driver `--profile-kernels`, read-only):
`scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/prefill_kda_*.json`,
`.../profiles/prefill_mla_*.json` (per-kernel tables per point) and
`.../kimi_single_layer/result_prefill_{kda,mla}.json` (per-point `us_step`, `num_tokens`, `prefix_len`).
Points: KDA `1,16384,pf` / `1,16384,pf49152` / `4,4096,pf`; MLA `1,16384,pf49152` / `1,16384,pf` / `4,4096,pf`
(16384 = the B200 default `chunked_prefill_size`, srt/arg_groups/memory_hook.py:88-179).

## Deliverables
1. **Phase-1 decision table** (`top-split-model-into-kernels`) for the prefill step of both layer types from
   the profiles: per launched kernel -> reuse kind / new kind / fold into a neighbour / elementwise placeholder.
   Expected new kinds (decide from the evidence, not from this list): `kda_chunk_prefill` (the chunk_kda
   kernel group incl. l2norm/cumsum/h-recurrence/output, args: tokens, heads=12, head_dim=128, prefix present or
   not, state dtype), `causal_conv1d_prefill` (or a backend of an existing conv kind), `mla_prefill_attention`
   (trtllm ragged fp8 DeepSeek attention: q_len, kv_len, heads=12, qk 192 / v 128, causal or prefix-chunk),
   plus `merge_state` and the prefix gather if they are not negligible. Existing kinds cover the GEMMs
   (`single_gemm` sglang backends at m = T), the MoE (`mxfp4_fused_moe` with m = 2T local rows over 112 experts,
   histogram-aware as in B10/B11), `mla_cache_append` at T rows, norms.
2. **Kinds + runners + Rust wiring** for the new kinds (`impl-register-kernel`, `impl-wire-kernel-to-rust`),
   profiled in `sglang_k3_env` by calling the same sglang/flashinfer entry points the layer calls.
3. **Worklet prefill branches**: `kimi_k3_kda_local`, `kimi_k3_mla_local`, `kimi_k3_moe_local` (and the dense
   front/back) consume `prefill_chunk_pairs` groups (context = prefix, append = chunk) — the arch already
   validates them (simulator/src/arch/kimi_k3_sglang.rs:1084-1140) but the worklets leave prefill leaves at zero
   (kimi_k3_kda_local.rs:5-7). Keep the decode paths bit-identical (existing tests must stay green; add tests
   in the style of the existing worklet tests: source order, prefill-vs-decode leaf selection, prefix>0 path).
4. **Presets** `presets/predict_kimi_k3_b200_rank1_layer_{kda,mla}_prefill.json` (+ `_cases.json`):
   KDA cases `[[0,16384]]`, `[[49152,16384]]`, 4 x `[[0,4096]]`; MLA cases `[[49152,16384]]`, `[[0,16384]]`,
   4 x `[[0,4096]]` (one group each; same arch block as the decode rank-1 presets: heads_per_rank 12,
   local_experts 112, local_top_k 2, kda_state_dtype bf16, sim_{kda,mla}_layers 1/0 or 0/1).
   JIT-fill the rows on GPU 7 into the branch DB, run `timing-predict` for both presets, and compare the
   layer totals per case with `result_prefill_{kda,mla}.json` (`us_step`). Target: within 15% per case; report
   the per-leaf comparison against the profile tables in `doc/alignment/kimi_k3_single_layer.md` under a new
   "chunked prefill" section (this is layer-level kernel evidence from the driver, not a server alignment).
5. `just test-cpu` green; commit on `kimi-k3-arch` (one or a few commits, English messages). The operator bakes
   the two predictions into the oracle image afterwards (`logs/predict_kimi_k3_b200_rank1_layer_{kda,mla}_prefill`
   must contain `prediction.meta.json` + `raw/params.json`).

## Do not
- Do not touch the shared `profiling/profile.db`, `main/` (the user's checkout), or the driver/judge.
- Do not change the decode presets or decode prediction numbers (B11 baselines are in use by running trials).
- Do not re-capture nsys probes (nsys needs the operator).
