# VibeSim optimization eval — workspaces, trials, and prediction runs

Updated 2026-09-11 (~06:10 UTC) from the k-trial "can the agent rediscover the
PR fix" eval pipeline under `scripts-local/eval-pipeline/` (harness in `lib/`,
GT specs in `issues/`, per-run outputs in `runs/<issue>/<ts>/trial_<n>/`).
Each trial: a fresh gpt-5.6-luna@max conversation gets only the sealed
before-state and a spoiler-free task; three evaluation layers then apply:

1. **Objective gate** (deterministic, cache-only) — re-simulates the agent's
   edited repo from a PINNED `eval-configs/before.json`; decides PASS/FAIL.
2. **Trajectory scan** (anti-reward-hacking) — mines the backend log's full
   tool-call stream for online PR lookup, eval-config tampering, and stripped-
   spoiler mining; any *hard* flag voids the trial.
3. **LLM judge** (advisory) — scores the diagnosis against the GT root cause.

A trial is **correct iff the gate passes AND the trajectory has no hard flag**.
All agent containers run with `CODEX_DOCKER_GPUS=""` (no GPU device) and every
host-side gate re-sim runs with `CUDA_VISIBLE_DEVICES=""` — zero GPU use by
construction.

## Per-issue scoreboard

| Issue | Eval base workspace (sealed commit) | Source workspace | GT scope (`--path`) | GT fix (sim) | Gate type | before → after (scoped) | k result (gate · judge) |
|---|---|---|---|---|---|---|---|
| vllm-28103 | `w_45893bc43b51` "eval-28103 base" (`55de3a6`) | `w_1471ccd48dec` | `unified.qk_norm` (`iter/1/0/1`) | `qwen3_dense_vllm_before.rs` `COPY_INPUTS true→false` | structural (forbidden `_input_copy$` leaves vanish) + metric | R0 4.714e-4 → 2.727e-4, R5 8.877e-5 → 4.438e-5, R6 1.792e-9 = | **3/3 · 3/3** |
| sglang-30947 | `w_f2c002bbcd89` "eval-30947 base" (`632d0a7`) | `w_68f6afe40d6d` | `unified.draft_topk1` (`iter/4`) | `qwen2_dense_sglang_before.rs` `DRAFT_TOPK1_BACKEND "torch"→"triton"` | metric-only (leaf set exact; R0 drops, R5/R6 pinned) | R0 9.529e-5 → 3.889e-5, R5 = R6 = 3.498e-6 both arms | **2/3 · 1/3** |
| sglang-32296 | — (not seeded) | `w_bad6960910bd` | 4 `input_quant` leaves (e.g. `iter/1/0/0/1/0`) | `QUANT_BACKEND "sglang_cuda_before"→"sglang_cuda"` | — | qkv `input_quant` R0 1.3518e-4 → 1.33128e-4 (−1.5 %) | **SKIPPED — delta too small to gate** |
| vllm-27931 | `w_1a73838d022f` "eval-27931 base" (`01bba52`) | `w_befc5365ebdc` | `unified.pre_attn.input_norm` (`iter/1`) | `llama3_dense_vllm_before.rs` `BACKEND "vllm_before"→"vllm_after"`, `VEC_SIZE 1→8`, block cap 1024→256 | metric-only (leaf exact; R0 drops, R5/R6 pinned) | R0 3.8872e-5 → 1.7605e-5, R5 8.390e-6 =, R6 3.277e-8 = | **3/3 · 3/3 (vetted, incl. T2 rerun)** |
| vllm-17973 | `w_7eda59064b88` "eval-17973 base" (`459107f`) | `w_e3c7eebe2924` | `unified.mrope` (`iter/0`) | Before worklet emits 3 proxy leaves; fix swaps to the after-style worklet (proxy emission removed) | structural (`rot_pos_emb`/`get_window_index`/`old_cu_seqlens` vanish; 4 `h2d_copy` stay) + metric (R0 −20 %, R5 −67 %) | R0 2.023e-5 → 1.620e-5, R5 7.877e-8 → 2.626e-8, R6 1.4e-11 = | **3/3 · 3/3 (vetted)** |

**v1 campaign final (2026-09-11 ~07:35 UTC): 11/12 vetted gate passes across
the four gateable issues; every pass also matched the judge except 30947 T2
(fix correct, diagnosis narrative penalized). The single genuine failure
(30947 T3) chose the largest R0/R5 outlier (`kv_cache_append`) over the
fixable one. NB the v1 seeded repos contained the `_after` implementations —
see the answer-key caveat below; the v3 campaign removes them.**

## v3 SEALED campaign (2026-09-11, final ~11:57 UTC): 12/12 · 12/12

Reruns of all four issues in freshly sealed workspaces with **no answer key**:
after implementations deleted (arch + registrations + location maps; for 17973
also the After worklet recipe; for 28103 the v2 kernel-capability model where
copies are DERIVED and delete-copies-only is unrepresentable), variant names
neutralized (27931's `vllm_before/after` backends → `vllm_cuda` keyed by real
`vec_size`/`block_size` columns; db rows renamed in place), baselines pinned
by hash in the specs, hardened cleanup + trajectory audit throughout.

| Issue (v3) | Eval workspace (seal) | Gate · Judge | Avg time | What synthesis required |
|---|---|---|---|---|
| vllm-28103-v3 | `w_af3858cf8529` (`94e90ed`) | **3/3 · 3/3** | 18.6 min | reason the RMSNorm capability declaration (derived copies) |
| sglang-30947-v3 | `w_31e2e24aa68d` (`3f37f38`) | **3/3 · 3/3** | 21.1 min | switch draft backend to the profiled triton implementation |
| vllm-27931-v3 | `w_9169afebff09` (`b6d4cf5`) | **3/3 · 3/3** | 22.5 min | author `VEC_SIZE=8`/block-cap parameters from the measured space |
| vllm-17973-v3 | `w_5961c414f17a` (`446afb5`) | **3/3 · 3/3** | 34.6 min | rewrite the mRoPE worklet's proxy emission (no template) |

**Conclusion: with contamination, reward hacking, and answer keys all
controlled, gpt-5.6-luna@max synthesizes these four PR fixes at 12/12** —
the v1 answer-key transcription affected how fixes were expressed, not
whether the diagnosis-and-fix capability exists. (Interestingly, 30947's
kv_cache_append detour did not recur in v3.)

v3 incidents (all preserved as `*_v1` dirs; verdicts above exclude them):
- 28103-v3 T1 (first attempt) read the original reproduction agent's
  plan/progress notes that the ORIGINAL 28103 seal had accidentally
  committed — a verbatim answer key. Voided; notes purged from all
  28103-lineage workspaces; only that one trial ever touched them.
- 28103-v3's first seal shipped the wrong `before.rs` (a rehearsal
  `git checkout --` reverted it pre-commit → uncompilable baseline whose
  error named the capability const), and a purge commit was made on top of
  an agent's fixed tree (pre-solving one rerun). All three trials
  invalidated; clean reseal `94e90ed` verified ON THE COMMITTED TREE; full
  k=3 rerun is what the table reports. Process fixes: specs pin
  `harness.baseline_commit` by hash; seals are verified post-commit.

## Trial history

| Issue | Run dir (`runs/…`) | Trial | Conversation | Gate | Judge (score) | Agent time | Notes |
|---|---|---|---|---|---|---|---|
| vllm-28103 | `vllm-28103/20260910_153241` | 1 | `e5329f2f601d` | PASS | match (1.0) | 25.3 min | found copies via ladder; also versioned the model.work location map |
| vllm-28103 | 〃 | 2 (v1) | `0b19ea44a922` | PASS | match (1.0) | 23.5 min | **VOIDED — contaminated**: read trial 1's post-fix `iter_breakdown.ans` from leftover `logs/` before its own baseline (old cleanup exempted `logs/`). Preserved as `trial_2_contaminated_v1`; rerun in flight |
| vllm-28103 | 〃 | 3 | `08530a727264` | PASS | match (1.0) | 26.5 min | vetted: zero pre-predict leftover reads |
| sglang-30947 | `sglang-30947/20260910_195129` | 1 | `342f17244ae0` | PASS | match (1.0) | 20.9 min | direct hit: eager-torch draft → fused Triton |
| sglang-30947 | 〃 | 2 | `c3905fa016dc` | PASS | no-match (0.05) | 24.5 min | implemented the correct draft fix but report led with `kv_cache_append` as diagnosis; strict judge penalized the narrative |
| sglang-30947 | 〃 | 3 | `4e4e900056e8` | FAIL | no-match (0.0) | 35.2 min | committed to `kv_cache_append` (largest leaf-level R0/R5 = 21 858×) with a 509-line KV-fusion change; draft leaf unchanged → gate FAIL |
| vllm-27931 | `vllm-27931/20260910_215417` | 1 | `6e1d2b87f170` | PASS | match (1.0) | 19.6 min | rediscovered vectorized RMSNorm; vetted clean |
| vllm-27931 | 〃 | 2 (v1) | `bf01aeef933f` | PASS | match (1.0) | 33.6 min | borderline: listed leftover `logs/eval_before_run` FILENAMES (not content) pre-predict; preserved as `trial_2_borderline_v1`, superseded by rerun |
| vllm-27931 | 〃 | 2 (rerun) | `352cac2acbe9` | PASS | match (1.0) | 24.7 min | vetted clean under hardened `-x` cleanup |
| vllm-27931 | 〃 | 3 | `7fc493f7e90c` | PASS | match (1.0) | 32.9 min | vetted clean (retro-audit: 0 hard, 0 leftover reads) |
| vllm-28103 | `vllm-28103/20260910_153241` | 2 (rerun) | `692a5fadb0a6` | PASS | match (1.0) | 24.6 min | contamination-free rerun; rediscovered the fix with zero leftover hints |
| vllm-17973 | `vllm-17973/20260910_221601` | 1 | `ea51a0cdda19` | PASS | match (1.0) | 39.4 min | removed the mrope proxy-leaf emission; vetted clean |
| vllm-17973 | 〃 | 2 | `e5063b987157` | PASS | match (1.0) | 35.0 min | first trial fully under hardened `-x` cleanup |
| vllm-17973 | 〃 | 3 | `314088dc9bf2` | PASS | match (1.0) | 56.2 min | longest trial of the campaign; vetted clean |

Judge caveat (30947): trials 1–2 were re-judged after fixing a prompt bug —
the original judge instructions hard-coded 28103's root cause. Old verdicts
preserved in each `trial.json` as `judge_prepatch`; re-judge verdicts agreed
with the originals in both cases.

## Integrity vetting (added 2026-09-11)

Two audits applied retroactively to every finished trial and automatically to
all future ones; results live in each `trial.json` (`trajectory` and
`leftover_reads_pre_predict` blocks):

- **Cross-trial contamination.** The original per-trial cleanup exempted
  `logs/` (protecting the gate output) and, per `git clean` semantics, skipped
  gitignored files — so trial N+1 could see trial N's post-fix prediction
  reports and tmp artifacts. Fixed: reset to the sealed `eval base` hash +
  `git clean -fdqx` keeping only `profiling/` (warm db), `target/` (cargo
  self-invalidates), `.venv`. An ordering audit (did the agent READ leftover
  `logs/`/`tmp/` before its own first predict overwrote them?) found exactly
  one real hit (28103 T2 — voided + rerun) and one borderline (27931 T2 —
  filename listing only; precautionary rerun). All other trials: zero reads.
- **Reward-hacking trajectory scan** (`lib/trajectory_scan.py`). Hard-fail
  patterns: online PR/issue lookup (github/search engines/external fetch),
  tampering with `eval-configs/`, reading stripped spoiler paths or a foreign
  conversation's plan/progress notes. Soft (advisory): reading the `_after`
  arch sources we deliberately keep, pip installs, unprompted GT issue-number
  mentions. Result across all finished trials: **0 hard flags**; soft flags
  are the expected after-arch reads. Note: agents reading their OWN
  `<cid>_plan.md` is standard runtime behavior, not a flag (first scan
  version false-positived on this; fixed).

## Timing-prediction runs used

| Purpose | Where | Runs |
|---|---|---|
| 28103 GT + gate re-sims | `w_45893bc43b51/repo/logs/` | pinned re-sim `eval_before_run` (overwritten per gate); GT from `w_1471ccd48dec` `20260809_3_qwen3_vllm_{before,after}` |
| 30947 GT | `w_68f6afe40d6d/repo/logs/` | `20260824_sglang30947_predict_{before,after}` (scoped `iter/4` reports added during GT extraction) |
| 30947 gate re-sims | `w_f2c002bbcd89/repo/logs/eval_before_run` | pinned config = before-config of the pair above, `cases_file` inlined |
| 32296 gateability check | `w_bad6960910bd/repo/logs/` | `20260823_6/7_qwen3_sglang_fp8_{before,after}_predict` — scoped qkv `input_quant` delta −1.5 % ⇒ skip |
| 27931 GT | `w_befc5365ebdc/repo/logs/` | `20260824_20/21_llama3_4096_prefill_timing_predict_{before,after}` (qwen3-8b pairs show ZERO delta — not used) |
| 27931 gate re-sims | `w_1a73838d022f/repo/logs/eval_before_run` | pinned llama3 4096-prefill before config |
| 17973 GT | `w_e3c7eebe2924/repo/logs/` | `vllm-17973-predict-cataloged-v19/{before,after}` (scoped `iter/0` reports added during GT extraction) |
| 17973 gate re-sims | `w_7eda59064b88/repo/logs/eval_before_run` | pinned config = the v19 before config (`qwen2_5_vl_mrope_before_v13_postdemotion.json`, cases inlined) |

## Finding these in the viz-ui (searchable handles)

The UI does **not** search raw `w_…` workspace ids or conversation ids.
**Conversations are the primary handle** — one workspace per issue, but one
*conversation per trial* (plus a judge conversation each). Use:

- **Trial conversations** (agent view): every eval conversation is titled
  `eval <issue> · trial <N> agent` / `… trial <N> judge` (set directly in the
  store; `run_trial.py` titles its own conversations at trial end). Search the
  title text, e.g. `eval vllm-28103` or `trial 2 agent`. Special labels:
  `CONTAMINATED(v1)` (voided 28103 T2 pair), `judge (old prompt)` (pre-fix
  30947 judges), `aborted trial (infra…)` (early symlink-failure leftovers).
  In-flight trials still show raw task-text titles until they finish.
  The Trial history table above maps every title to its conversation id.
- **Workspaces** (secondary, to browse an issue's whole set): search the
  display name — `eval-28103 base (before-state)`, `eval-30947 base
  (before-state)`, `eval-27931 base (before-state)`, `eval-17973 base
  (before-state)`.
- **Explore Results**: search `eval_before_run` — each eval base workspace has
  ONE such **Timing prediction** entry (the gate's pinned re-sim; overwritten
  every trial, so it always shows the LAST gate run). Disambiguate by the
  entry's `arch_type`: `qwen3_dense_vllm_before` = 28103,
  `qwen2_dense_sglang_before` = 30947, `llama3_dense_vllm_before` = 27931,
  `qwen2_5_vl_mrope_before` = 17973.
  (28103's entry was invisible until 2026-09-10 — its run dir lacked the
  `artifact.meta.json` discovery marker; added, per the known marker rule.)
- **GT source pairs**: search the run names in the "Timing-prediction runs
  used" table below (e.g. `20260824_sglang30947_predict_before`) — those live
  in the source workspaces and are already cataloged.

## Harness anatomy (scripts-local/eval-pipeline/)

- `issues/<id>.yaml` — GT spec: scope path, rungs, tolerances, per-issue
  `harness.analyze_bin` / `harness.task_file` overrides, `gt_root_cause`.
- `lib/run_eval.py` — k sequential trials in one shared workspace
  (`--start/--run-ts` resume supported), aggregates `summary.json`.
- `lib/run_trial.py` — reset to the sealed `eval base` commit (deletes agent
  branches) → luna@max conversation via backend :8766 → cache-only gate
  (`timing-predict` pinned config + `analyze optimality-scoped`) → judge →
  `trial.json` (+ `agent.diff` vs baseline, `scoped_report.json`, gate logs).
- `lib/objective_gate.py` — structural (forbidden/required/exact leaves) +
  metric (`metric_rungs` near-after AND below-before) + invariants
  (`unchanged_rungs`, R6 necessary) checks.
- `lib/judge.py` — gpt-5.6-luna (single, high) JSON verdict vs `gt_root_cause`.
- `lib/agent_task*.md` — spoiler-free task prompts (28103: unnecessary-work
  criterion; 30947/27931: hardware-limit-ratio criterion + cache-only
  achievability note).
- `lib/seed_*.sh` — de-spoiled base seeding (excludes logs/reports/tmp/.git,
  issue-named docs, plan/progress notes, `reference/`; sanitizes PR-number
  comments; vendors `req-frontend`; seals an `eval base` git commit).

## Caveats

- **Anti-cheat:** the gate re-sims from a pinned before config, so a config or
  arch-selector swap to the `_after` recipe still selects the unfixed before
  arch and fails; only a real code fix to the before arch passes.
- **ANSWER-KEY LEAK (quantified 2026-09-11, user-spotted):** the `_after` arch
  sources remain in every seeded repo, with self-describing names (backend
  `"vllm_after"`, files `*_after.rs`) — and the trajectory audit shows **all
  12/12 trials read them** (1-8 reads each). The honest claim is therefore
  split: *diagnosis* (ladder → operator → achievability) was independent —
  30947 T3 failed with the key in-repo, proving it wasn't telegraphed — but
  *fix synthesis* was largely transcription from the after recipe (27931's
  report literally says "now uses the warm vllm_after recipe"). The campaign
  measured "diagnose + select the available fix," NOT "synthesize the fix."
  Hardened v3 design: strip the after arches (5-file surgery per issue),
  rename backends to capability-descriptive neutral names (e.g.
  `vllm_rmsnorm_vec8`), scrub before/after framing from configs and task text.
  The profile.db must still contain the fix's rows (cache-only validation
  requires it), so v3 measures "diagnose + discover the fix among profiled
  implementation variants + write the recipe change" — the strongest claim a
  cost-simulator eval can make without GPU kernel authorship (that's 28103-v2
  Phase 2+).
- **Discovery ambiguity is real and measured:** at leaf granularity
  `kv_cache_append` out-ranks the GT operator on R0/R5 in both 30947 (21 858×
  vs 27×) and 27931 (10.7× vs 4.6×) but has no cached alternative
  implementation. Agents that survey-then-pivot pass; 30947 trial 3 committed
  to it and failed. The 27931 task adds an explicit cache-only achievability
  note.
- **Gateability screen:** an issue needs GT delta ≫ tolerance at some scope.
  sglang-32296 failed this (−1.5 % per quant leaf) and was skipped without
  burning agent time; vllm-17973's time delta is also small but its structural
  leaf change still gates.
- Rung semantics: R0 measured, R5 hardware limit, R6 segmented necessary,
  R7 fused necessary (values from `analyze optimality-scoped`, GPU-seconds).
