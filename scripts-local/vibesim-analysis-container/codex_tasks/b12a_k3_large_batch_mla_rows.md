# B12a — Kimi-K3 rank-1 presets at B=256/512: the MLA decode-attention leaf mispredicts by 20-50x

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust` (branch `kimi-k3-arch`,
HEAD 51810876). Read `CLAUDE.md`, skills `operate-run-timing-predict`, `impl-register-kernel`,
`impl-wire-kernel-to-rust`. Commit when green (`just test-cpu`; 4 known `test_energy_*` host failures),
report in English. This is a small, focused fix — do it before anything else.

## GPU rules (hard)
Only GPU index 7 / UUID `GPU-c88e489a-0693-2c29-a1e3-30952377f742` (user-authorized 2026-09-24; SHARED with
the user's runs and with our optimizer-loop trials — do not wait for it to be idle, do not kill anything on
it), only via `VIBESIM_PROFILE_GPUS=GPU-c88e489a-0693-2c29-a1e3-30952377f742
VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db
TMPDIR=/raid/tmp/yilegu_k3_tmp`. Never other GPUs (0-6), never `--gpus all`, never slurm, never the shared
`profiling/profile.db`. Expect timing noise from co-tenants; use medians.

## Symptom
New presets (committed, HEAD): `presets/predict_kimi_k3_b200_rank1_layer_{kda,mla}_b512.json` with cases
KDA 128/256/512 @ 8k and MLA 512 @ 8k, 256 @ 8k, 16 @ 40960 (decode). `timing-predict` on both finished in
7 s (no JIT fill happened) and produced:
- KDA: 417 / 514 / 751 us at B=128/256/512 — plausible (MoE 264/308/389, recurrent 30/58/113).
- MLA: **16614 us at 512x8k and 5854 us at 256x8k**, of which `mla_decode_attention` = 15998 / 5402 us
  (`logs/predict_kimi_k3_b200_rank1_layer_mla_b512/reports/iter_breakdown.ans`). The branch DB has
  `mla_decode_attention` rows `sglang_cutedsl_mla` at batch_size 256 / kv_len 8192 = 288 us (and 128/8192 =
  142 us; `sglang_trtllm_mla` 128/8192 = 104.5 us, no trtllm rows above batch 128, no rows at all for
  batch 512), so the leaf is neither reading the 256-row nor triggering a fill for 512: it extrapolates.
  Expected physics: KV read = B x 8192 x 576 B -> 2.4 GB at B=512 = 345 us at 7 TB/s; measured layer step
  at 128x8k is 493 us with attention ~105 us, so attention at 512 should be ~400-600 us, not 16 ms.

## Deliverables
1. Find why the `mla_decode_attention` predictor (simulator/src/timing/kernels/mla_decode_attention.rs and
   the K3 MLA worklet's input construction in simulator/src/worklet/kimi_k3_mla_local.rs) does not hit the
   B=256 row and extrapolates instead of JIT-filling at B=512 — check the input it builds for a decode group
   (batch_size vs decode_kv_total vs per-request kv_len, backend choice trtllm vs cutedsl, page/dtype keys),
   and the lookup/extrapolation policy. Fix it so that out-of-grid shapes are FILLED (JIT, GPU 7) rather than
   extrapolated across an order of magnitude, or at worst extrapolated along the measured axis.
2. JIT-fill the rows both b512 presets need (`kernel-profile count-missing` = 0), for both MLA backends the
   worklet can select (trtllm + cutedsl) at batch 256 and 512 with kv_len 8192, and 16 @ 40960.
3. Re-run `timing-predict` for `presets/predict_kimi_k3_b200_rank1_layer_mla_b512.json` (and the KDA one to
   confirm it is unchanged) and compare with the measured pristine step times the operator will drop at
   `scripts-local/vibesim-analysis-container/iter_opt_eval_k3_mla_b512/seed_transfer_verdict.json`
   (`points.<key>.before_us`; keys "512,8192", "256,8192", "16,65536mix") and
   `.../iter_opt_eval_k3_kda_b512/seed_transfer_verdict.json` ("512,8192", "256,8192", "128,8192").
   Target: layer totals within 15%. Report the per-case comparison.
4. Add a regression test that a decode group above the profiled batch grid does not silently extrapolate
   the MLA attention leaf (whatever policy you implement), keep the existing decode predictions for the
   B=1/32/128 presets unchanged, `just test-cpu` green, commit on `kimi-k3-arch`.

Do not touch `main/` (the user's checkout), the shared `profiling/profile.db`, the driver/judge, or the
decode presets' numbers. When done, write a 10-line summary to
`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/codex_runs/b12a_summary.md`.
