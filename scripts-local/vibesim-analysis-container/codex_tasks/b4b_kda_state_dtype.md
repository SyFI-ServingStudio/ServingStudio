# Task: Kimi-K3 KDA worklet — pick the decode kernel chain by recurrent-state dtype (branch `kimi-k3-arch`)

You are in the git worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust` on branch
`kimi-k3-arch` (clean checkout with a working `req-frontend` submodule; `cargo` at `~/.cargo/bin`; Python via
`uv run --no-sync ...`, never `uv sync`). Do not cd to `.../main`. Commit on `kimi-k3-arch` with the trailer
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Follow `skills/impl-compose-worklet/SKILL.md` and
`skills/impl-compose-arch/SKILL.md` conventions; keep `just test-cpu` green (the `tests/test_l1_single_gemm.py::
test_energy_*` failures are pre-existing on this host — ignore them).

## Facts (verified on B200 with the real layer, 2026-09-22)
- sglang's fused KDA decode kernel (`kda_fused_decode`, our kind `kda_fused_decode:sglang_fused`) is only
  covered for an **fp32** recurrent state (`sglang/kernels/ops/attention/kda_fused_decode.py::covered` requires
  `ssm_states.dtype == float32`).
- The cookbook B200 recipes run `--mamba-ssm-dtype bfloat16`. With a bf16 state the KDA decode is the **unfused
  Triton chain**: `causal_conv1d_update` (kind `gdn_causal_conv_decode:sglang_triton`, channels = 3*heads*128),
  `fused_sigmoid_gating_delta_rule_update` (kind `kda_recurrent_decode:sglang_triton`, state_dtype bf16) and
  `FusedRMSNormGated` (kind `gdn_gated_rms_norm:sglang_triton`, hidden 128, m = tokens*heads). Measured per
  step at B=128, 12 heads: recurrent 29 us, conv 7 us, gated norm ~3 us (fused kernel would be 39 us).
- Both are production paths (fp32 state = default/low-latency recipe; bf16 = balanced/high-throughput recipe).

## Change
1. Add a `kda_state_dtype` (fp32|bf16) parameter to the K3 arch selector (`IterArchSel::KimiK3Sglang` in
   `simulator/src/arch/config.rs`, ProviderSchema, `build.rs` builder) and the KDA worklet config, defaulting
   to **bf16** (the cookbook recipe). Thread it into the recurrent-state byte accounting (state bytes per
   request = heads*128*128*elem) and KV/state reporting.
2. KDA decode leaf selection: `kda_state_dtype == fp32` -> the single `kda_fused_decode` leaf (as now);
   `bf16` -> three leaves `kda_conv_decode` (gdn_causal_conv_decode), `kda_recurrent_decode`
   (kda_recurrent_decode, state_dtype bf16) and `kda_gated_norm` (gdn_gated_rms_norm). Leaf names must stay
   stable per mode; update the location map (`model/work/location_maps/kimi_k3_sglang_unified.json`) so BOTH
   modes' leaves are listed and every `model.work` semantic row is consumed exactly once in each mode — if the
   map format is per-arch-type and cannot branch on a parameter, register two arch_type tags or two maps and
   explain which the analyzer will pick; run the analyzer location tests.
3. Presets: `predict_kimi_k3_b200_sglang_tp8ep8pp2.json` and `predict_kimi_k3_b200_rank1_layer_kda.json` use
   bf16 (cookbook); add `predict_kimi_k3_b200_rank1_layer_kda_fp32state.json` for the fused path.
4. Tests: slot/leaf-name freeze for both modes; state bytes for both dtypes.
5. Verify: `just test-cpu`; `uv run --no-sync python -m launcher timing-predict presets/predict_kimi_k3_b200_rank1_layer_kda.json`
   on CPU must reach the cache-miss listing and show the three split leaves (paste the listing).

Report: files changed, test results, the two miss listings (bf16 and fp32 presets), and any question.
