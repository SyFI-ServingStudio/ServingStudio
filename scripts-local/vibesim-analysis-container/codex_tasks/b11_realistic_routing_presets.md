# B11 — mirror the driver's realistic routing in the Kimi-K3 rank-1 presets + re-align

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust` (branch `kimi-k3-arch`,
HEAD ef202ca2 = your B10). Read `CLAUDE.md`, skills `operate-run-alignment`, `operate-run-timing-predict`,
`impl-compose-arch`. Commit when green (`just test-cpu`), report in English.

## GPU rules (hard)
Only GPU index 7 / UUID `GPU-c88e489a-0693-2c29-a1e3-30952377f742` (user-authorized 2026-09-24; it is
SHARED with a light-load run of the user's — do not wait for it to be idle, do not kill anything on it),
only via `VIBESIM_PROFILE_GPUS=GPU-c88e489a-0693-2c29-a1e3-30952377f742
VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db
TMPDIR=/raid/tmp/yilegu_k3_tmp` or `docker run --gpus '"device=GPU-c88e489a-0693-2c29-a1e3-30952377f742"'`.
Never other GPUs (0-6), never `--gpus all`, never slurm, never `profiling/profile.db`. Expect a few % timing
noise from the co-tenant; use medians.

## What changed on the harness side
The driver (`scripts-local/vibesim-analysis-container/kimi_single_layer_decode.py`, commit e4a3c25 on the
workspace branch `kimi-k3-loop`) now models an EP8 rank realistically:
- `--local-topk 2`: K3 routes top-16 over 896 global experts; a rank holds 112, so its expected share is
  2 assignments per token. The single-process driver cannot run sglang's EP group, so it emulates the share
  by routing top-2 over the 112 local experts (128 tokens -> 256 local rows instead of the previous 2048).
- `--hidden-scale 1.0`: unit-scale decode inputs so the per-token identity dominates the pre-MoE norm; the
  previous 0.02 inputs made every token route to the same ~20 experts (B10 found the collapsed histogram).
The case configs will carry these flags; the judge's baselines/goldens are re-captured with them. Measured with the new flags (GPU 7, CUDA-graph replay, 2026-09-24 01:22): **KDA 252.4 / 174.6 / 139.7 us at
B=128/32/1 @8k** (old flags: 556.5 / 310.7 / 179.6); **MLA 352.6 (128x8k) / 306.7 (1x1M) / 264.7 (16x64k)**
(old: 680.4 / 345.5 / 357.9). Full JSON: `scripts-local/vibesim-analysis-container/kimi_single_layer/result_{kda,mla}_realistic_*.json`;
per-kernel tables: `.../kimi_single_layer/profiles/*realistic*` (read-only evidence).

## Deliverables
1. Rank-1 presets `presets/predict_kimi_k3_b200_rank1_layer_{kda,mla}.json` + the K3 arch/MoE worklet:
   model the rank's routed load as 128 x 2 = 256 rows over 112 local experts with a realistic (Poisson-like,
   NOT collapsed, NOT perfectly balanced) per-expert distribution; keep the production TP8/EP8/PP2 preset
   at the global top-16-of-896 semantics (routing_experts=896, local share 1/8). Make the histogram source
   explicit (measured histogram from the alignment payload when present; the analytic distribution otherwise).
2. Refill whatever new `mxfp4_fused_moe` rows the presets need (GPU 3, gated), re-run the rank-1 predictions
   and compare with the realistic measurements per point (B=1/32/128 @8k for KDA; 128x8k, 1x1M, 16x64k for
   MLA). Target: layer totals within 10%.
3. Alignment: the existing nsys probes (`kimi_single_layer/align/*`) were captured with the OLD flags. Do NOT
   re-capture (needs nsys on GPU 3, the operator does that); instead document the new-flag comparison from
   the driver's `--profile-kernels` tables vs the prediction per leaf, and update
   `doc/alignment/kimi_k3_single_layer.md` with a "realistic routing" section (clearly separated from the
   old-flag alignment tables).
4. `just test-cpu` green (4 known `test_energy_*` host failures). Commit on `kimi-k3-arch`.
