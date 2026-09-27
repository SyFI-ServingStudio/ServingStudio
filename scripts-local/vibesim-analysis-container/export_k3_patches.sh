#!/usr/bin/env bash
# Export the WORKING Kimi-K3 changes as reviewable git patches under patches/:
#   patches/sglang/mla/NN_<lever>.patch  one patch per ACCEPTED round of the MLA case, in order
#                                        (diff of the judged tree against the tree the round started from)
#   patches/sglang/kda/NN_<lever>.patch  same for KDA (stacked seed + round 22)
#   patches/sglang/{mla,kda}_best_tree_vs_pristine.patch   cumulative (all accepted levers of a case)
#   patches/harness/*.patch              the eval-harness code vs the workspace branch base, per component
#   patches/vibesim/kimi_k3_vs_roofline_base.patch   VibeSim kimi-k3 branch (K3 arch, kinds, runners, presets,
#                                        alignment, docs) vs its base, plus COMMITS.txt
# Every sglang chain is VALIDATED: applying the per-round patches onto a copy of the pristine tree must
# reproduce the case's best_tree exactly (source files).
# Patches are `diff -ruN` with paths relative to python/sglang -> apply with `patch -p1` inside that dir.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/patches"; rm -rf "$OUT"; mkdir -p "$OUT/sglang/mla" "$OUT/sglang/kda" "$OUT/harness" "$OUT/vibesim"
SRC_RE='\.(py|cu|cuh|h|hpp|cc|cpp|inc|jinja)$'

# diff two sglang trees -> unified patch with a/ b/ prefixes (p1), source files only
tree_diff () { # $1 from-dir $2 to-dir $3 out
  local from="$1" to="$2" out="$3"
  # portable form: diff with explicit relative paths from a temp dir holding symlinks a/ and b/
  local tmp; tmp="$(mktemp -d)"; ln -s "$from" "$tmp/a"; ln -s "$to" "$tmp/b"
  (cd "$tmp" && diff -ruN -x __pycache__ -x '*.pyc' a/ b/ || true) \
    | python3 -c '
import sys, re
src = re.compile(r"'"$SRC_RE"'")
out, keep = [], False
for line in sys.stdin:
    if line.startswith("diff ") or line.startswith("Only in "):
        keep = bool(src.search(line.strip()))
    if keep: out.append(line)
sys.stdout.write("".join(out))' > "$out"
  rm -rf "$tmp"
  echo "  $(basename "$out"): $(grep -c '^diff ' "$out") files, $(grep -c '^[+-][^+-]' "$out") changed lines"
}

# apply a chain of patches to a copy of pristine and compare with the best tree
validate_chain () { # $1 pristine $2 best $3... patches
  local pristine="$1" best="$2"; shift 2
  local tmp; tmp="$(mktemp -d)"; cp -a "$pristine" "$tmp/tree"; find "$tmp/tree" -name __pycache__ -prune -exec rm -rf {} +
  for p in "$@"; do (cd "$tmp/tree" && patch -p1 -s < "$p") || { echo "!! $p failed to apply"; rm -rf "$tmp"; return 1; }; done
  local t2; t2="$(mktemp -d)"; ln -s "$tmp/tree" "$t2/a"; ln -s "$best" "$t2/b"
  local d; d="$(cd "$t2" && diff -rq -x __pycache__ -x '*.pyc' -x '*.orig' a/ b/ | grep -E "$SRC_RE" || true)"
  rm -rf "$tmp" "$t2"
  if [ -z "$d" ]; then echo "  chain reproduces $(basename "$(dirname "$best")")/$(basename "$best") exactly"; else echo "!! chain differs from best tree:"; echo "$d"; return 1; fi
}

echo "== sglang MLA chain"
M="$HERE/iter_opt_eval_k3_mla"
tree_diff "$M/pristine_tree"        "$M/trial_8_tree_judged"  "$OUT/sglang/mla/01_r08_cutedsl_to_trtllm_mla_decode.patch"
tree_diff "$M/trial_8_tree_judged"  "$M/trial_20_tree_judged" "$OUT/sglang/mla/02_r20_route_quant_fused_112_top2.patch"
tree_diff "$M/trial_20_tree_judged" "$M/trial_21_tree_judged" "$OUT/sglang/mla/03_r21_front_fp32_gemm_cute_tgv.patch"
tree_diff "$M/trial_21_tree_judged" "$M/trial_23_tree_judged" "$OUT/sglang/mla/04_r23_latent_up_shared_down_bf16_tgv.patch"
tree_diff "$M/trial_23_tree_judged" "$M/trial_24_tree_judged" "$OUT/sglang/mla/05_r24_shared_routed_alt_stream_overlap.patch"
tree_diff "$M/trial_24_tree_judged" "$M/trial_25_tree_judged" "$OUT/sglang/mla/06_r25_is_var_seq_persistent_kvconcat_warps.patch"
tree_diff "$M/pristine_tree"        "$M/best_tree"            "$OUT/sglang/mla_best_tree_vs_pristine.patch"
validate_chain "$M/pristine_tree" "$M/best_tree" "$OUT"/sglang/mla/0*.patch

echo "== sglang KDA chain"
K="$HERE/iter_opt_eval_k3_kda"
tree_diff "$K/pristine_tree"        "$K/stacked_tree_judged"  "$OUT/sglang/kda/01_stacked_fastpath_overlap_cutedsl_gemm_warps.patch"
tree_diff "$K/stacked_tree_judged"  "$K/trial_22_tree_judged" "$OUT/sglang/kda/02_r22_bf16_state_fused_kda_decode_kernel.patch"
tree_diff "$K/pristine_tree"        "$K/best_tree"            "$OUT/sglang/kda_best_tree_vs_pristine.patch"
validate_chain "$K/pristine_tree" "$K/best_tree" "$OUT"/sglang/kda/0*.patch

echo "== sglang Claude (Opus 5.5 via Bedrock) campaign chains (seed = the decode best tree of the same layer)"
mkdir -p "$OUT/sglang/claude"
CB="$HERE/iter_opt_eval_k3_mla_b512_claude"
tree_diff "$HERE/iter_opt_eval_k3_mla/best_tree" "$CB/trial_1_tree_judged" "$OUT/sglang/claude/mla_b512_01_r1_moe_autotune_kvconcat_satfinite.patch"
tree_diff "$CB/trial_1_tree_judged" "$CB/trial_2_tree_judged" "$OUT/sglang/claude/mla_b512_02_r2_persistent_decode_tail_split.patch"
tree_diff "$CB/trial_2_tree_judged" "$CB/trial_3_tree_judged" "$OUT/sglang/claude/mla_b512_03_r3_outgate_tail_overlap_attnres_fused_add.patch"
validate_chain "$HERE/iter_opt_eval_k3_mla/best_tree" "$CB/best_tree" "$OUT"/sglang/claude/mla_b512_0*.patch
CB="$HERE/iter_opt_eval_k3_kda_b512_claude"
tree_diff "$HERE/iter_opt_eval_k3_kda/best_tree" "$CB/trial_1_tree_judged" "$OUT/sglang/claude/kda_b512_01_r1_moe_autotune_fused_decode_vector_ldst_bfa_overlap.patch"
validate_chain "$HERE/iter_opt_eval_k3_kda/best_tree" "$CB/best_tree" "$OUT"/sglang/claude/kda_b512_0*.patch
CB="$HERE/iter_opt_eval_k3_mla_prefill_claude"
tree_diff "$HERE/iter_opt_eval_k3_mla/best_tree" "$CB/trial_1_tree_judged" "$OUT/sglang/claude/mla_prefill_01_r1_kv_pack_quantize_fp8_attnres_fused_add.patch"
tree_diff "$CB/trial_1_tree_judged" "$CB/trial_2_tree_judged" "$OUT/sglang/claude/mla_prefill_02_r2_prefix_attention_cutedsl_fmha.patch"
validate_chain "$HERE/iter_opt_eval_k3_mla/best_tree" "$CB/best_tree" "$OUT"/sglang/claude/mla_prefill_0*.patch
CB="$HERE/iter_opt_eval_k3_kda_prefill_claude"
tree_diff "$HERE/iter_opt_eval_k3_kda/best_tree" "$CB/trial_1_tree_judged" "$OUT/sglang/claude/kda_prefill_01_r1_strided_chunk_kernels_no_copies_no_host_sync.patch"
validate_chain "$HERE/iter_opt_eval_k3_kda/best_tree" "$CB/best_tree" "$OUT"/sglang/claude/kda_prefill_0*.patch
# unaccepted but exact and worth keeping (below the 3-sigma floor on the primary point):
tree_diff "$CB/best_tree" "$CB/trial_2_tree_judged" "$OUT/sglang/claude/UNACCEPTED_kda_prefill_r2_cuda_kda_chunk_h_hscan.patch"

echo "== sglang Claude SHAPE-MATCHED decode chains (seed = the b512 Claude best tree of the same layer; slurm mode)"
CB="$HERE/iter_opt_eval_k3_kda_shapes_claude"
if [ -d "$CB/trial_3_tree_judged" ]; then
  tree_diff "$HERE/iter_opt_eval_k3_kda_b512_claude/best_tree" "$CB/trial_1_tree_judged" "$OUT/sglang/claude/kda_shapes_01_r1_split_moe_front_small_m_alt_stream_tgv_fp32.patch"
  tree_diff "$CB/trial_1_tree_judged" "$CB/trial_2_tree_judged" "$OUT/sglang/claude/kda_shapes_02_r2_l2_prefetch_oproj_routed_front.patch"
  tree_diff "$CB/trial_2_tree_judged" "$CB/trial_3_tree_judged" "$OUT/sglang/claude/kda_shapes_03_r3_fused_decode_attnres_tuning.patch"
  validate_chain "$HERE/iter_opt_eval_k3_kda_b512_claude/best_tree" "$CB/best_tree" "$OUT"/sglang/claude/kda_shapes_0*.patch
fi
CB="$HERE/iter_opt_eval_k3_mla_shapes_claude"
if [ -d "$CB/trial_2_tree_judged" ]; then
  tree_diff "$HERE/iter_opt_eval_k3_mla_b512_claude/best_tree" "$CB/trial_1_tree_judged" "$OUT/sglang/claude/mla_shapes_01_r1_b1_kv_split8_lse_merge_kvconcat_q_replicas.patch"
  tree_diff "$CB/trial_1_tree_judged" "$CB/trial_2_tree_judged" "$OUT/sglang/claude/mla_shapes_02_r2_attnres_cluster_small_moe_finalize_rmsnorm.patch"
  validate_chain "$HERE/iter_opt_eval_k3_mla_b512_claude/best_tree" "$CB/best_tree" "$OUT"/sglang/claude/mla_shapes_0*.patch
fi
echo "== sglang Claude LONG-CONTEXT PREFILL chains (seed = the prefill Claude best tree; slurm mode)"
for c in kda mla; do
  CB="$HERE/iter_opt_eval_k3_${c}_lcprefill_claude"; prev="$HERE/iter_opt_eval_k3_${c}_prefill_claude/best_tree"; n=0; chain=()
  for k in 1 2 3; do
    t="$CB/trial_${k}_tree_judged"; v="$CB/trial_${k}_verdict.json"
    [ -d "$t" ] && [ -f "$v" ] && grep -q '"verdict": "PASS"' "$v" || continue
    n=$((n+1)); p="$OUT/sglang/claude/${c}_lcprefill_0${n}_r${k}.patch"; tree_diff "$prev" "$t" "$p"; chain+=("$p"); prev="$t"
  done
  [ "$n" -gt 0 ] && validate_chain "$HERE/iter_opt_eval_k3_${c}_prefill_claude/best_tree" "$CB/best_tree" "${chain[@]}"
done

echo "== harness (workspace branch kimi-k3-loop vs its base)"
BASE="$(git -C "$WS" merge-base kimi-k3-loop yilegu/dev)"
C=scripts-local/vibesim-analysis-container
git -C "$WS" diff "$BASE"..HEAD -- "$C/kimi_single_layer_decode.py" > "$OUT/harness/01_driver_kimi_single_layer_decode.patch"
git -C "$WS" diff "$BASE"..HEAD -- "$C/judge_k3.py" "$C/run_iter_opt_eval.sh" "$C/agent_task_k3_opt.md" "$C"/issue_k3_*.json > "$OUT/harness/02_judge_runner_prompt_cases.patch"
git -C "$WS" diff "$BASE"..HEAD -- "$C"/run_k3_*.sh "$C"/after_*.sh "$C"/resume_k3_loop_when_free.sh "$C"/restart_mla_b512_after_b12.sh "$C"/launch_b12*.sh "$C"/finish_b12.sh "$C"/recheck_best_trees_seeds.sh "$C"/rebake_mla_b512_oracle.sh "$C"/k3_prefill_profiles.sh > "$OUT/harness/03_loop_orchestration_scripts.patch"
git -C "$WS" diff "$BASE"..HEAD -- "$C/build_opt_history.py" "$C/opt_history_techniques.md" > "$OUT/harness/04_warm_start_history.patch"
git -C "$WS" diff "$BASE"..HEAD -- "$C/build_context_k3.sh" "$C"/codex_tasks/b*.md "$C/run_k3_jit_fill.sh" > "$OUT/harness/05_oracle_bake_and_codex_tasks.patch"
git -C "$WS" log --oneline --reverse "$BASE"..HEAD > "$OUT/harness/COMMITS.txt"
for f in "$OUT"/harness/*.patch; do echo "  $(basename "$f"): $(grep -c '^diff --git' "$f") files, $(grep -c '^[+-][^+-]' "$f") changed lines"; done

echo "== VibeSim (main/ branch kimi-k3 vs roofline-base)"
VB="$(git -C "$WS/main" merge-base kimi-k3 roofline-base)"
git -C "$WS/main" diff "$VB"..kimi-k3 --stat -- . ':!profiling/profile.db' ':!logs' ':!*.db' > "$OUT/vibesim/STAT.txt"
git -C "$WS/main" diff "$VB"..kimi-k3 -- . ':!profiling/profile.db' ':!logs' ':!*.db' ':!*.parquet' ':!*.png' > "$OUT/vibesim/kimi_k3_vs_roofline_base.patch"
git -C "$WS/main" log --oneline --reverse "$VB"..kimi-k3 > "$OUT/vibesim/COMMITS.txt"
echo "  kimi_k3_vs_roofline_base.patch: $(grep -c '^diff --git' "$OUT/vibesim/kimi_k3_vs_roofline_base.patch") files, $(wc -l < "$OUT/vibesim/COMMITS.txt") commits"

cat > "$OUT/README.md" <<EOF
# Kimi-K3 loop — patches of what works ($(date +%F))

Regenerate with \`./export_k3_patches.sh\` (validates that each sglang chain reproduces the case's best tree).

## sglang (apply inside \`python/sglang\` of lmsysorg/sglang:v0.5.20 with \`patch -p1\`)
- \`sglang/mla/01..06\`: the six accepted MLA levers in the order they were accepted (each judged PASS against
  the pristine goldens; cumulative 304.6 -> 253.5 us @1x1M, -16.8%). \`sglang/mla_best_tree_vs_pristine.patch\`
  is the cumulative patch (identical result).
- \`sglang/kda/01..02\`: the KDA levers (stacked seed -4.8% + round 22 bf16-state fused decode kernel;
  cumulative 403.0 -> 379.4 us @B=128, -5.9%). \`sglang/kda_best_tree_vs_pristine.patch\` cumulative.
- These trees also passed: two extra seeds each (rel_err <= 0.011), the MLA mixed-length point, and the
  B=512/256 transfer checks (see K3_PIPELINE_AND_RESULTS.md).
- \`sglang/claude/\`: the Claude Opus 5.5 campaign chains on top of the decode best trees (seed = the same layer's
  best tree): MLA-b512 r1-r3 (976.3 -> 877.8 us @512x8k), KDA-b512 r1 (675.2 -> 600.4), MLA-prefill r1-r2
  (12128 -> 11229 us @16k chunk with 48k prefix), KDA-prefill r1 (9072 -> 8564 us). Each chain is validated against
  the case's best_tree. The r1 b512 patches include the one-shot MoE autotune that reproduces production's warmup
  (harness gap; see the write-up). UNACCEPTED_*: exact but below the judge's 3-sigma floor.

## harness (workspace branch \`kimi-k3-loop\`, vs base \`$(git -C "$WS" rev-parse --short "$BASE")\`)
- \`harness/01_driver_*.patch\`: the single-layer extractor/driver (CUDA-graph metric, goldens, prefill points, ...).
- \`harness/02_*\`: judge, trial runner, agent prompt, case configs. \`harness/03_*\`: loop orchestration.
- \`harness/04_*\`: warm-start history builder + curated techniques. \`harness/05_*\`: oracle bake, Codex task prompts.
- \`harness/COMMITS.txt\`: the branch's commit log (per-commit patches: \`git format-patch $(git -C "$WS" rev-parse --short "$BASE")..kimi-k3-loop\`).

## VibeSim (repo \`main/\`, branch \`kimi-k3\`, vs \`roofline-base\`)
- \`vibesim/kimi_k3_vs_roofline_base.patch\` (+ \`STAT.txt\`, \`COMMITS.txt\`): K3 kernel kinds + runners,
  \`kimi_k3_sglang\` arch + worklets, model.work label, presets, alignment pack, oracle predictions; excludes the
  branch profile.db (rows live in kimi_single_layer/k3_branch_profile.db; merging into the shared db needs approval).
EOF
du -sh "$OUT"; echo "== OK: $OUT"
