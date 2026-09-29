# Handoff: merging the Kimi-K3 optimization loop into VibeOptimizer

*Written 2026-09-28 for the intern taking over. Everything referenced here is committed on the branches in §1;
nothing in this document depends on the author's session.*

> **Merge from VibeOptimizer branch `kimi-k3-loop`, folder `kimi-k3-loop/`** (a self-contained snapshot of this
> directory's loop files, 2026-09-29). This workspace copy keeps the full history and the one-off campaign scripts.

## 0. What the K3 loop is, in one paragraph

An open-ended optimization loop for ONE Kimi-K3 decoder layer (KDA linear-attention layers and MLA full-attention
layers, sglang v0.5.20, one B200 of the cookbook TP8/EP8 recipe). There is no PR answer key: an agent (Claude Opus 5.5
via Bedrock, earlier Codex) gets a real sglang checkout, a synthetic single-layer driver that times the layer's step
(CUDA-graph replay for decode and speculative-verify points, eager for prefill and mixed batches) and checks numerics
against goldens of the pristine tree, plus a VibeSim roofline oracle baked for the workload. A judge re-measures the
agent's tree and the tree it started from (5 reps, same slurm job), accepts any gain above max(3σ, 0.5%) on the
primary point with no secondary regression and CHECK pass on every point, and promotes the tree; rounds continue from
the best tree. Results vs pristine, all seed-rechecked (seeds 1, 2, fresh goldens):

| workload (primary point) | KDA | MLA |
|---|---|---|
| decode 1×1M | 136.6 → 101.8 µs (−25.5%) | 304.5 → 216.4 µs (−28.9%) |
| decode 128×8k | −7.1% | −6.2% |
| large batch 512×8k | −12.1% | −10.1% |
| chunked prefill 16k @ 245k prefix | 9.03 → 8.13 ms (−10.0%) | 27.69 → 20.45 ms (−26.1%) |
| speculative verify 64×4 @8k (graph) | −5.6% | −6.1% |
| mixed 64 decodes + 16k chunk @8k | 10.1 → 8.35 ms (−17.5%) | 21.2 → 8.3 ms (**−61%**; −90% @128 dec + 4k) |

Write-ups: `K3_PIPELINE_AND_RESULTS.md` (campaigns 1–5), `K3_CAMPAIGN6_SHAPES_LCPREFILL.md` (campaign 6 + follow-ups,
§9.6 = verify/mixed), `K3_TECHNIQUES_ALL_SHAPES.md` (every accepted lever × shape, dead ends, transfer rules).

## 1. Where the code is

| what | repo / branch | path |
|---|---|---|
| loop harness, cases, prompts, judge, patches, KB, write-ups | `VibeSimWorkspace` branch `kimi-k3-loop` (github serendipity-zk/VibeSimWorkspace) | `scripts-local/vibesim-analysis-container/` |
| VibeSim K3 support (arch recipe, kernel kinds + runners, model.work label, presets, alignment docs, oracle predictions) | `VibeSim` branch `kimi-k3` (github SyFI-VibeSim/VibeSim; the local checkout is `VibeSimWorkspace/main/`, Rust built in worktree `main-k3-rust` = same commits) | whole repo; K3 entry points: `simulator/src/arch/kimi_k3_sglang*.rs`, `worklet/kimi_k3_*`, `profiling/runners/attention/kimi_k3_*.py`, `presets/predict_kimi_k3_*.json`, `doc/alignment/kimi_k3_*.md` |
| B200 profile rows for the K3 kernel kinds | NOT in git: `scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db` (branch DB; merging into VibeSim's shared `profiling/profile.db` needs Yile's approval) | |
| the sglang optimizations | no sglang fork. Accepted trees: `iter_opt_eval_k3_<case>/best_tree` (untracked, on /raid). Reviewable form: `patches/sglang/{mla,kda,claude}/*.patch` (tracked; `export_k3_patches.sh` regenerates and validates every chain against its best tree) | apply with `patch -p1` inside `python/sglang` of `lmsysorg/sglang:v0.5.20` |
| per-trial evidence (agent transcripts, verdicts, goldens) | untracked on /raid: `iter_opt_eval_k3_*/` (trial_k_agent.log, trial_k_verdict.json, trial_k_tree_judged, rounds.jsonl), `/raid/yilegu/eval_goldens/golden_k3_<case_key>/` | |

The loop's files, by role (all in `scripts-local/vibesim-analysis-container/`):

- **Driver / extractor** `kimi_single_layer_decode.py` — builds one layer with seeded weights, drives decode /
  prefill / mixed / verify steps through the real sglang backends, `--cuda-graph`, `--verify-graph`, `--capture` /
  `--replay` goldens, `--profile-kernels`, `--nvtx-align`. Its bytes are part of the judge's case key.
- **Judge** `judge_k3.py` — pristine goldens + baseline per case key; measures agent tree and baseline tree in one job;
  gates: correctness (strict max-rel ≤ 0.02 for decode/verify, row-wise rule for prefill/mixed), latency ≥
  max(3σ, 0.5%), no secondary regression.
- **Runner** `run_iter_opt_eval.sh` (one trial: seed tree, agent container, rebuild smoke, judge), `run_k3_continuous.sh`
  (rounds with best_tree promotion, `rounds.jsonl`), `run_k3_pair_slurm.sh`, `resume_mixed_verify.sh`.
- **GPU policy (slurm mode)** `k3_gpu_shim.py` (in-container proxy replacing the driver), `k3_gpu_broker.py` (host side,
  one `sbatch` per measurement), `k3_slurm_step.sh`, `slurm_gpu.sh`, `gpu_run.sh`. The agent container has no GPU.
- **Cases** `issue_k3_<layer>_<workload>_claude.json` (points, driver args, oracle URL, prompt render values).
- **Prompt** `agent_task_k3_opt.md`, `gpu_note_slurm.md`; **KB** `k3_opt_history/` built by `build_opt_history.py`
  + curated `opt_history_techniques.md` (mounted read-only into the agent container as the warm start).
- **Oracle** `build_context_k3.sh` → image `vibesim-analysis:k3`, one container per baked workload
  (`VIBESIM_PREBAKED_RUN`, ports 8801–8810); `bake_lcprefill_oracles.sh`, `rebake_prefill_oracles_b12c.sh`.
- **Hardening** `recheck_best_trees_seeds.sh` (seeds 1, 2 vs pristine), `rejudge_near_misses.sh` (15 reps),
  `bedrock_gate_and_resume.sh`, `redo_void_rounds.sh`, `finish_and_resume.sh`.

## 2. How this maps onto VibeOptimizer

VibeOptimizer today = a *PR case* (before/after commits, answer key, full engine, serving benchmark). The K3 loop is an
*open-ended single-layer case*. Same three-container shape, different case type.

| K3 loop piece | VibeOptimizer counterpart | what changes |
|---|---|---|
| `issue_k3_*.json` | `cases/<id>/case.yaml` + `benchmark.yaml` | new case type `open_ended_layer`: no `after` commit, no `ground_truth.yaml` targets; `points{primary, secondary}`, `driver_args`, `reps`, `rel_err_max`, `seed`; workload tags `mix` / `pf<prefix>` / `mx<chunk>[p<prefix>]` / `vk<k>` |
| `lmsysorg/sglang:v0.5.20` used as-is | `rga-local/<case>:<arch>-before` from `Dockerfile.<arch>` | the engine image is the upstream image; the "build" is `find … __pycache__ -prune -exec rm -rf {} +` + a one-point smoke (`LAYER_SMOKE_OK`) — no compile step |
| `kimi_single_layer_decode.py` | `assets/probes/extract_and_profile.py` | the probe is the whole workload: it builds the layer itself instead of hooking a live engine. Keep it as a separate probe; do not fold it into `extract_and_profile.py` |
| `k3_gpu_shim.py` + `k3_gpu_broker.py` | `broker.py` verbs `build · profile · smoke` | same idea; K3's broker additionally runs each verb as its own `sbatch` job (GPU held only while measuring). The shim rewrites `/tmp/<x>` → `/workspace/opt_run/tmp/<x>` for path flags only |
| `judge_k3.py` | `target_probe.py` + `acceptance.py` | K3 rules to port: case key = hash(driver bytes, image, driver_args, points, seed, reps); pristine goldens cached per key; baseline = the round's start tree measured in the same job; 3σ/0.5% gate; row-wise prefill rule; verify judged as graph replay |
| `run_k3_continuous.sh` + `best_tree` + `rounds.jsonl` | `campaign.py` attempts | attempts are *chained*: attempt k+1 seeds from the accepted tree of attempt k; the correctness reference is always pristine |
| `k3_opt_history/` warm start | `knowledge_bank/` | per-trial `incremental.diff` + `agent_iterations.md` + curated techniques; mounted read-only |
| `build_context_k3.sh` (prebaked prediction per workload) | `rga-local/vibesim-analysis:<case>` | one image, N containers (`VIBESIM_PREBAKED_RUN`); presets live on the VibeSim `kimi-k3` branch |
| seed rechecks, 15-rep re-judge, VOID rounds | `viability.py`, `report.py` | new: a null round whose tree equals its start tree is VOID (re-run), not a result |

## 3. Suggested merge plan (each step has a check you can run without the author)

1. **Read** `K3_CAMPAIGN6_SHAPES_LCPREFILL.md` §7 (harness lessons) and `K3_TECHNIQUES_ALL_SHAPES.md` §6 (transfer
   rules) before touching code. Check: you can explain why verify must be judged as a graph replay.
2. **Reproduce one measurement** on this node from the loop as it is (no VibeOptimizer yet):
   `sbatch slurm_gpu.sh k3_slurm_step.sh env CASES="mla_shapes_claude" ./recheck_best_trees_seeds.sh slurm 1`
   → `iter_opt_eval_k3_mla_shapes_claude/best_tree_seed1_verdict.json` must say PASS with 1×1M ≈ −28%.
3. **Case type** in VibeOptimizer: `models.py` gains `open_ended_layer`; `cases.py` loads `issue_k3_*.json`-equivalent
   YAML; leak scan trivially passes (no answer key). Check: `vibeoptimizer validate` on a ported `k3-mla-shapes` case.
4. **Probe + broker**: register the driver as a probe asset; broker verbs `profile` (driver run) and `smoke` (rebuild
   smoke); keep the per-verb `sbatch` mode as an option (`gpu_mode: slurm`). Check: the agent container has no GPU and
   a `profile` call produces the driver's `JSON` line.
5. **Judge**: port `judge_k3.py` rules into `target_probe.py`/`acceptance.py` (or call `judge_k3.py` as a first step).
   Check: re-judging `iter_opt_eval_k3_mla_verify_claude/trial_1_tree_judged` against its `pristine_tree` reproduces
   PASS ≈ −4% at 64×8k vk3 (graph mode).
6. **Chained attempts + VOID**: attempt k+1 seeds from attempt k's accepted tree; a non-PASS attempt with an unchanged
   tree is VOID and re-run. Check: a fake harness that edits nothing yields VOID, not FAIL.
7. **Oracle bake** as a VibeOptimizer step (`build_context_k3.sh` → `vibesim-analysis:k3`). Check: `workspace-info` and
   `optimality` on port 8801 return K3 nodes.
8. **One end-to-end round** of `k3_kda_shapes_claude` under VibeOptimizer; compare with `rounds.jsonl` of the original.
9. Only then: KB import, write-up consolidation, and (with Yile's approval) the branch profile DB merge.

## 4. Things that bit us (read before debugging)

- **Eager verify is 65% host time.** σ 90–400 µs made the 3σ gate demand 22–99% and reject a real −3.5% kernel; a
  −14.7% eager "win" was 0% on GPU. Always judge `vk` points with `--verify-graph`.
- **FlashInfer autotune**: production warms up with `autotune(True)` and a disk cache; the judge's cache mount must be
  read-write or the first new shape fails with EROFS (false FAIL). Flags: `--flashinfer-autotune` (+ cache path).
- **Driver bytes are in the case key.** Editing the driver re-keys every case's goldens; never do it while a trial is
  in flight (the judge snapshots the driver at start; the agent's copy is installed at trial start).
- **Mixed batch composition**: sglang rewrites MIXED→EXTEND; the decode rows of a mixed chunk went through the
  chunked-prefix MHA path in production's `handle_attention_trtllm_mla` too — that is where the −61% came from.
- **slurm-manager placeholders** (`gpumgmt-ph-*`, partition `placeholder`) take every free GPU within minutes and
  outrank `main`. An agent whose measurements queue edits blind. Gate rounds on a schedulable-GPU probe.
- **Bedrock Opus 5.5 had 503 storms for hours** (all regions). Gate on a live probe (15 green minutes) and treat rounds
  whose tree came back unchanged as VOID.
- **Accepted MLA decode trees regress verify** (+5.8…+31%): their small-m dispatch assumes one query row per request.
  Seed verify from pristine or gate those levers on `q_len`.
- **GPU policy on this node**: GPU work only via slurm partition `main`, hold a GPU only while measuring; never touch
  Yile's jobs; the judge and every measurement is one `sbatch`.
- **Between-job spread** of the same tree is ~2% (graph) / ~3% (mixed eager) on different GPUs; only same-job
  comparisons count. The seed rechecks are the cumulative numbers to quote.

## 5. Open decisions (Yile's)

- Merge `kimi_single_layer/k3_branch_profile.db` rows into VibeSim's shared `profiling/profile.db`.
- Where campaign 6 lands in `K3_PIPELINE_AND_RESULTS.md` (Yile's working copy of that file diverged from HEAD).
- Whether to create an sglang fork with one branch per patch chain.
- Stop the cron job `7870a345` (5-minute status loop) once the handoff is done.
