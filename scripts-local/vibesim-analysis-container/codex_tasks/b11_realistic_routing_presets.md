# B11 — mirror the driver's realistic routing in the Kimi-K3 rank-1 presets + re-align

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust` (branch `kimi-k3-arch`,
HEAD ef202ca2 = your B10). Read `CLAUDE.md`, skills `operate-run-alignment`, `operate-run-timing-predict`,
`impl-compose-arch`. Commit when green (`just test-cpu`), report in English.

## GPU rules (hard) — same as B10
Only GPU index 3 / UUID `GPU-019267a2-092a-6798-3cc5-57ffc761e004`, only via
`VIBESIM_PROFILE_GPUS=<that UUID> VIBESIM_PROFILE_DB=<branch db> TMPDIR=/raid/tmp/yilegu_k3_tmp` or
`docker run --gpus '"device=<that UUID>"'`; check `nvidia-smi -i 3` memory.used < 1000 MiB before EVERY GPU
step and wait (sleep 60, retry) if not — the user's own jobs come and go on this device. Never other GPUs,
never slurm, never `profiling/profile.db`.

## What changed on the harness side
The driver (`scripts-local/vibesim-analysis-container/kimi_single_layer_decode.py`, commit e4a3c25 on the
workspace branch `kimi-k3-loop`) now models an EP8 rank realistically:
- `--local-topk 2`: K3 routes top-16 over 896 global experts; a rank holds 112, so its expected share is
  2 assignments per token. The single-process driver cannot run sglang's EP group, so it emulates the share
  by routing top-2 over the 112 local experts (128 tokens -> 256 local rows instead of the previous 2048).
- `--hidden-scale 1.0`: unit-scale decode inputs so the per-token identity dominates the pre-MoE norm; the
  previous 0.02 inputs made every token route to the same ~20 experts (B10 found the collapsed histogram).
The case configs will carry these flags; the judge's baselines/goldens are re-captured with them. Measured
numbers with the new flags will be in `scripts-local/vibesim-analysis-container/kimi_single_layer/result_{kda,mla}_realistic_*.json`
and kernel tables in `.../kimi_single_layer/profiles/{kda,mla}_realistic.json` (read-only evidence; if they
do not exist yet, the operator's measurement is still queued behind a busy GPU 3 — do the CPU work first).

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
