
## ADDENDUM (operator)
- You are in the git WORKTREE `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust` on branch
  `kimi-k3-arch` (a clean checkout of `kimi-k3` at 368af34a with the correct `req-frontend` submodule, so
  `cargo` builds here). Commit on `kimi-k3-arch`; the operator merges into `kimi-k3`. Do not cd into
  `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main` (its submodule is rewound and Rust fails there).
- `.venv` here is a symlink to the main checkout's venv: run Python as `uv run --no-sync python ...` /
  `uv run --no-sync pytest ...` (never plain `uv sync`). `cargo` is at `~/.cargo/bin/cargo`.
- `profiling/profile.db` has NO B200 rows yet for the new K3 kinds (they are measured later on GPU by the
  operator); so deliverable 5 (timing-predict reaching the cache-miss stage) is exactly what is expected.
- The K3 kinds/backends on this branch are: kinds `kda_recurrent_decode` (backends `torch`, `sglang_triton`),
  `kda_fused_decode` (`sglang_fused`), `mla_decode_attention` (`sglang_cutedsl_mla`, `sglang_trtllm_mla`,
  `sglang_triton`), `mxfp4_fused_moe` (`sglang_trtllm_mxfp4`); backends `gdn_causal_conv_decode:sglang_triton`,
  `gdn_gated_rms_norm:sglang_triton`, `batched_gemm:sglang_k3_absorb`. Read their Rust twins under
  `simulator/src/timing/kernels/` for the exact KernelConfig fields before wiring ops. Production KDA at 12
  heads runs the FUSED kernel (`kda_fused_decode`), so the KDA decode worklet should use that kind (conv +
  recurrence + gated norm in one leaf) with the split kinds as the alternative backend path if the config
  supports backend lists.
- The measured per-kernel launch tables live at
  `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/{kda,mla}_prod_v2_B*_L*.json`.
