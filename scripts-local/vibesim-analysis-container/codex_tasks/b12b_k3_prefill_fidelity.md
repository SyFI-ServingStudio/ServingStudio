# B12b — Kimi-K3 chunked-prefill predictions are ~60% below the measured layer step: close the gap

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust` (branch `kimi-k3-arch`; HEAD = the
B12 commit "chunked-prefill support"). Read `CLAUDE.md`, `doc/alignment/kimi_k3_single_layer.md` (B12 section),
skills `operate-run-timing-predict`, `impl-register-kernel`, `impl-compose-worklet`. Commit when green
(`just test-cpu`; 4 known `test_energy_*` host failures), report in English, and write a 10-line summary to
`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/codex_runs/b12b_summary.md`.

## GPU rules (hard)
Only GPU index 7 / UUID `GPU-c88e489a-0693-2c29-a1e3-30952377f742` (user-authorized, SHARED; do not wait for
idle, do not kill anything), only via `VIBESIM_PROFILE_GPUS=GPU-c88e489a-0693-2c29-a1e3-30952377f742
VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db
TMPDIR=/raid/tmp/yilegu_k3_tmp`. Never other GPUs, never `--gpus all`, never slurm, never the shared
`profiling/profile.db`. You may ALSO run the driver itself in a container to get ground truth (see below).

## Symptom
`timing-predict` on `presets/predict_kimi_k3_b200_rank1_layer_{kda,mla}_prefill.json` gives
KDA 8894 / 8972 / 8537 us for cases `[[0,16384]]` / `[[49152,16384]]` / 4x`[[0,4096]]` and
MLA 11792 / 8002 / 7557 us for `[[49152,16384]]` / `[[0,16384]]` / 4x`[[0,4096]]`.
Measured eager layer step (driver, GPU 7, `--profile-kernels`): KDA 22132 / 22535 / 22111 us,
MLA 29538 / 20585 / 17763 us. The per-kernel tables show the measured step is ~95% kernel time (KDA 20959 of
22132 us; MLA 24785 of 29538 us) with 41-75 launches, so launch overhead is NOT the gap. Top kernels (KDA
1x16384 first chunk): `nvjet_sm100_tst_128x256_64x6_2x1_2cta` 6111 us, `nvjet_sm100_tss_128x256_64x6_2x1_2cta`
5817 us, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32` (routed MXFP4 MoE) 2963 us, `sglang::attn_res_fused_tma_kernel`
1883 us. MLA prefix case adds `fmhaSm100aKernel_QkvE4m3OBfloat16HQk19...` 6144 us (prefix attention).
Tables: `scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/prefill_{kda,mla}_B*_L*pf*.json`
(read-only), step totals in `.../kimi_single_layer/result_prefill_{kda,mla}.json`.

## What to do
1. Reconcile leaf by leaf against the profile tables (kernel name -> leaf): for each large kernel group
   decide whether the predictor (a) has no leaf for it, (b) has the leaf but with the wrong shape (e.g. the
   fused front GEMM at T=16384 rows is one 7168 -> 15984+6016 GEMM, the shared experts run at T rows, the routed
   MoE physically processes 2T = 32768 rows, latent up/down at 2T or T), or (c) has the right shape but the row
   was measured under different conditions than the layer (e.g. cuBLAS `nvjet` heuristics at m = 16384 pick a
   2-CTA 128x256 tile that runs at ~50% of peak: a 16384 x 7168 x 15984 GEMM is 3.75 TFLOP; 6.1 ms means
   ~0.6 PFLOP/s). Prefer measuring the exact production call (same shapes, dtypes, out_dtype, weights layout)
   in the runner over adjusting numbers.
2. Where the leaf is right but the measurement differs, verify with a direct probe in the sglang container:
   `docker run --rm --gpus '"device=GPU-c88e489a-0693-2c29-a1e3-30952377f742"' -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 --shm-size 32g -v /raid/hf/hub:/raid/hf/hub:ro -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer -v /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer_decode.py:/tmp/kimi_single_layer_decode.py:ro --workdir /tmp lmsysorg/sglang:v0.5.20 python3 /tmp/kimi_single_layer_decode.py --point 1,16384,pf --attn-type kda --moe-backend flashinfer_mxfp4 --attn-heads 12 --experts 112 --ep 8 --local-topk 2 --hidden-scale 1.0 --bf16-gemm-init --iters 10 --warmup 3 --profile-kernels /tmp/p.json`
   (MLA: `--attn-type mla --attention-backend cutedsl_mla --kv-cache-dtype fp8_e4m3`, points `1,16384,pf49152`
   etc.). `--bf16-gemm-init` reproduces the production scheduler's bf16 GEMM backend init (auto -> cutedsl on
   SM100); the earlier tables were taken WITHOUT it (cuBLAS-only), so re-profile with it and compare.
3. Target: each of the six cases within 15% of the measured step, and the top-5 kernels within 25% each.
   Update the alignment doc section with the per-leaf table (predicted vs measured, before/after).
4. Keep decode predictions unchanged (B=1/32/128 and the b512 presets); `just test-cpu` green; commit.

Do not touch `main/`, the shared `profiling/profile.db`, the driver/judge. Do not run while a judge trial
container (`judge_k3_*`, `iteropt_k3_*`) is on GPU 7 if you can avoid it: bursty profiling inflated a judge
measurement 3x earlier today; prefer short bursts and medians.
