# Task: run VibeSim's alignment on the Kimi-K3 single-layer probe (branch `kimi-k3-arch`)

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust`, branch `kimi-k3-arch` (do not cd to
`.../main`). CPU only; `uv run --no-sync ...`; `cargo` at `~/.cargo/bin` (retry on build-dir lock).
Set `VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db`
for every launcher/analyzer command (the K3 rows live in that branch DB, not in profiling/profile.db).
Follow `skills/operate-run-alignment/SKILL.md` and `skills/top-align-with-framework/SKILL.md`; read
`alignment/README.md` (kernel-align path: profile_log_dir + timing_predict_log_dir + labeled kernel sequences)
and `alignment/labeling/README.md`. Commit artifacts that belong in the repo (label rules, the alignment yaml,
a short markdown note) on `kimi-k3-arch` with trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`;
large logs stay out of git.

## What exists
- Measured side (a LAYER-LEVEL DEBUG PROBE — one KimiK3DecoderLayer driven by a synthetic driver under nsys,
  20 decode iterations, B=128 requests at kv 8192, production TP8/EP8 rank shape, bf16 KDA state, fp8 MLA KV):
  `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/align/kda_b128_l8192/`
  and `.../mla_b128_l8192/`, each with `profile.nsys-rep`, `profile.sqlite`, `records/server.log`
  (VibeSimAlignmentWorker + VibeSimAlignmentIteration records, adapter sglang_text), `metrics.jsonl`,
  `parsed.json` and `kernel_sequences.json` (already produced with `python -m alignment parse --sqlite
  --metrics --server-log --range-mode forward --default-stage decode`). 26 (KDA) / 29 (MLA) kernels per step.
  Per-kernel torch.profiler tables for the same points: `.../kimi_single_layer/profiles/kda_prod_v3_bf16state_B128_L8192.json`
  and `mla_prod_v2_B128_L8192.json`.
- Simulated side: `presets/predict_kimi_k3_b200_rank1_layer_kda.json` (rank-1 topology, 1 KDA layer, cases
  B=1/32/128 @8192) already predicted at `logs/predict_kimi_k3_b200_rank1_layer_kda` (warm branch DB) — the
  B=128 case is the one to align (predicted 542us vs measured graph 623us / nsys busy ~600us). The MLA preset
  `predict_kimi_k3_b200_rank1_layer_mla.json` is being filled on GPU right now; do the KDA alignment first and
  leave the MLA one ready to run (yaml + rules) — run it too if `logs/predict_kimi_k3_b200_rank1_layer_mla`
  is complete when you get there (check for `reports/iter_breakdown.ans`).

## Do
1. Write `profile_result.json` for each probe dir in the shape `launcher/alignment.py` requires (`log_dir`
   equal to the profile dir, `sqlite`, `metrics_jsonl`, `parsed_nsys`, `kernel_sequences`, `gpu`,
   `server_tp_size`=1, `server_dp_size`=1 ...) — read `alignment/profiler/runner.py:640-678` for the key list.
2. `python -m alignment label initialize <kernel_sequences.json> <labeled.json>` then write label rules
   (`label apply`) mapping every measured kernel of the KDA step to the modelled leaves of the K3 arch
   (`unified.kda.attention.{input_layernorm,qkvbfg_a_proj,kda_conv_decode,kda_recurrent_decode,kda_gated_norm,
   o_proj,post_attention_layernorm}`, `unified.kda.moe.{merged_front,shared_gate_up_activation,shared_down,
   mxfp4_fused_moe,routed_norm,latent_up,add3}`); kernels with no modelled leaf (e.g. the two
   `attn_res_fused_tma_kernel` launches of the attention-residual stream, cuda-graph glue) stay unmapped and
   must be REPORTED with their per-step time. `label check` / `coverage` must pass.
3. `launcher alignment analyze <yaml>` with `profile_log_dir` = the probe dir and `timing_predict_log_dir` =
   the KDA prediction dir, iteration mode; select the B=128 case. Read
   `reports/alignment_iteration_report.json` + `payloads/alignment_iteration_breakdowns.jsonl`.
4. Report a table: per operation measured_ms vs simulated_ms, relative error, and the unmapped measured time;
   the duration-weighted error; and the top 3 leaves to fix in the cost model (with the concrete suspected
   cause, e.g. a kernel row measured at the wrong shape, a missing leaf, or the graph side-stream overlap the
   probe measures but the cost tree sums serially). Mark clearly that this is layer-level probe evidence, not a
   full-model alignment (no K3 checkpoint exists).
