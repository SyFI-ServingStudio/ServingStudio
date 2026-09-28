# Kimi-K3 loop — patches of what works (2026-09-28)

Regenerate with `./export_k3_patches.sh` (validates that each sglang chain reproduces the case's best tree).

## sglang (apply inside `python/sglang` of lmsysorg/sglang:v0.5.20 with `patch -p1`)
- `sglang/mla/01..06`: the six accepted MLA levers in the order they were accepted (each judged PASS against
  the pristine goldens; cumulative 304.6 -> 253.5 us @1x1M, -16.8%). `sglang/mla_best_tree_vs_pristine.patch`
  is the cumulative patch (identical result).
- `sglang/kda/01..02`: the KDA levers (stacked seed -4.8% + round 22 bf16-state fused decode kernel;
  cumulative 403.0 -> 379.4 us @B=128, -5.9%). `sglang/kda_best_tree_vs_pristine.patch` cumulative.
- These trees also passed: two extra seeds each (rel_err <= 0.011), the MLA mixed-length point, and the
  B=512/256 transfer checks (see K3_PIPELINE_AND_RESULTS.md).
- `sglang/claude/`: the Claude Opus 5.5 campaign chains on top of the decode best trees (seed = the same layer's
  best tree): MLA-b512 r1-r3 (976.3 -> 877.8 us @512x8k), KDA-b512 r1 (675.2 -> 600.4), MLA-prefill r1-r2
  (12128 -> 11229 us @16k chunk with 48k prefix), KDA-prefill r1 (9072 -> 8564 us). Each chain is validated against
  the case's best_tree. The r1 b512 patches include the one-shot MoE autotune that reproduces production's warmup
  (harness gap; see the write-up). UNACCEPTED_*: exact but below the judge's 3-sigma floor.
- `sglang/claude/{kda,mla}_shapes_*`, `{kda,mla}_lcprefill_*`: Campaign 6 (shape-matched decode incl. 1x1M,
  long-context chunked prefill at 128k-262k). `{kda,mla}_verify_*`: speculative-verify (TARGET_VERIFY, CUDA-graph
  metric; KDA seeded from the shapes tree, MLA from pristine). `{kda,mla}_mixed_*`: mixed prefill-chunk + decode
  batches (eager metric; seeded from the lcprefill trees). See K3_CAMPAIGN6_SHAPES_LCPREFILL.md §9.6.

## harness (workspace branch `kimi-k3-loop`, vs base `c252264`)
- `harness/01_driver_*.patch`: the single-layer extractor/driver (CUDA-graph metric, goldens, prefill points, ...).
- `harness/02_*`: judge, trial runner, agent prompt, case configs. `harness/03_*`: loop orchestration.
- `harness/04_*`: warm-start history builder + curated techniques. `harness/05_*`: oracle bake, Codex task prompts.
- `harness/COMMITS.txt`: the branch's commit log (per-commit patches: `git format-patch c252264..kimi-k3-loop`).

## VibeSim (repo `main/`, branch `kimi-k3`, vs `roofline-base`)
- `vibesim/kimi_k3_vs_roofline_base.patch` (+ `STAT.txt`, `COMMITS.txt`): K3 kernel kinds + runners,
  `kimi_k3_sglang` arch + worklets, model.work label, presets, alignment pack, oracle predictions; excludes the
  branch profile.db (rows live in kimi_single_layer/k3_branch_profile.db; merging into the shared db needs approval).
