#!/usr/bin/env bash
# Stage the Docker build context for the VibeSim analysis container (case: vllm-28103).
#
# The container is a FIXED before-state diagnosis oracle: it serves the 5 agent-facing
# vibesim_api verbs (workspace-info, simulate, analyze, optimality, kernels) against a
# baked before-state prediction. It does NOT re-simulate agent edits and needs NO Rust
# toolchain / PyO3 runtime -- only the prebuilt `analyze` binary + profile.db + assets.
#
# Answer-key hygiene (trust boundary): source files come from `git archive defcc63`
# (the working tree is on the AFTER state and is unsafe); the after arch recipe, the
# after work-map, and the *_plan/_progress answer keys are purged. The result carries
# NO after arch, NO after config, NO after run -- a true before-only workspace.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC_REPO="${SRC_REPO:-/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/agent-workspaces/w_45893bc43b51/repo}"
API_PKG="${API_PKG:-/raid/yilegu/roofline_guided_agent/vibesim-api-package}"
BASELINE="${BASELINE:-defcc63}"
# Genuine before-state prediction (copies present, R6 populated). NOTE: do NOT use
# $SRC_REPO/logs/eval_before_run -- that working tree is on the AFTER commit
# (81eec3f removed the copies from the "before" recipe), so a re-sim there yields
# after-state numbers. The canonical before prediction lives in the API test-bed.
BAKED_SRC="${BAKED_SRC:-/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/agent-workspaces/w_5609772a9151/repo}"
BAKED_RUN="${BAKED_RUN:-api_v0_before}"      # before-state prediction dir under $BAKED_SRC/logs
OUT="${OUT:-$HERE/context}"

REPO_OUT="$OUT/repo"
echo "== staging into $OUT (baseline $BASELINE from $SRC_REPO)"
rm -rf "$OUT"
mkdir -p "$REPO_OUT" "$OUT/app"

echo "== 1. tracked before-state ASSETS at $BASELINE (curated; avoids .venv/target/.git bloat)"
# Only the dirs the prebuilt `analyze` binary + shim actually read: gpu spec, model
# config/work assets, simulator source (marker + arch names for workspace-info),
# and the before config. NOT .venv (9G torch/cuda), target, alignment, .git, docs.
git -C "$SRC_REPO" archive "$BASELINE" -- gpu model simulator eval-configs | tar -x -C "$REPO_OUT"

echo "== 2. purge after-state / answer-key files (trust boundary)"
rm -f "$REPO_OUT/simulator/src/arch/qwen3_dense_vllm_after.rs" \
      "$REPO_OUT/model/work/location_maps/qwen3_dense_vllm_after_unified.json"
# any *_plan.md / *_progress.md that ride along in the tree
find "$REPO_OUT" -maxdepth 2 -type f \( -name '*_plan.md' -o -name '*_progress.md' \) -delete

echo "== 3. copy state-independent binary artifacts from the working tree"
mkdir -p "$REPO_OUT/profiling" "$REPO_OUT/target/release"
cp "$SRC_REPO/profiling/profile.db"        "$REPO_OUT/profiling/profile.db"
cp "$SRC_REPO/target/release/analyze"      "$REPO_OUT/target/release/analyze"

echo "== 4. copy the baked before-state prediction -> logs/before (from $BAKED_SRC)"
mkdir -p "$REPO_OUT/logs/before"
cp -a "$BAKED_SRC/logs/$BAKED_RUN/." "$REPO_OUT/logs/before/"
# clear stale scoped reports so optimality regenerates cleanly (also proves the binary)
rm -f "$REPO_OUT/logs/before/reports/optimality_scoped_"*.json 2>/dev/null || true

echo "== 5. copy the API package (docs, examples, canonical shim) -> vibesim-api/"
mkdir -p "$OUT/vibesim-api"
cp -a "$API_PKG/." "$OUT/vibesim-api/"

echo "== 6. app: container shim (canonical + prebaked simulate) + server + entrypoint"
cp "$API_PKG/vibesim_api.py" "$OUT/app/vibesim_api.py"
python3 "$HERE/patch_prebaked.py" "$OUT/app/vibesim_api.py"
cp "$HERE/app/server.py"     "$OUT/app/server.py"
cp "$HERE/app/entrypoint.sh" "$OUT/app/entrypoint.sh"
cp "$HERE/app/uv"            "$OUT/app/uv"   # offline uv shim for the R6/R7 labeler
cp -a "$HERE/skill"          "$OUT/skill"

echo "== 7. verify NO after/answer-key FILES remain (trust boundary = files, not name refs)"
# The answer is the after RECIPE / WORK-MAP / CONFIG / RUN and the plan/progress keys.
# A dangling `pub mod qwen3_dense_vllm_after;` name-reference in shared registry source
# (mod.rs/config.rs/unified.rs) is NOT the answer and is unreachable over the HTTP API
# (the agent never reads source; workspace-info globs *.rs stems, and after.rs is gone).
leak=$(find "$REPO_OUT" \
        \( -name '*_after*' -o -name '*after*.json' -o -name '*_plan.md' -o -name '*_progress.md' \) \
        -not -path '*/logs/before/*' 2>/dev/null)
if [ -n "$leak" ]; then
  echo "!! LEAK: after/answer-key file remains:" >&2; echo "$leak" >&2; exit 1
fi
# after recipe file specifically must be gone
if [ -f "$REPO_OUT/simulator/src/arch/qwen3_dense_vllm_after.rs" ]; then
  echo "!! LEAK: after arch recipe still present" >&2; exit 1
fi
# workspace-info must not surface an 'after' arch recipe name
if ls "$REPO_OUT/simulator/src/arch/"*after* >/dev/null 2>&1; then
  echo "!! LEAK: an *after* arch recipe file is listable" >&2; exit 1
fi
# the baked prediction MUST be genuine before-state (redundant copies present)
if ! grep -qrE 'q_input_copy|k_input_copy' "$REPO_OUT/logs/before/raw/cost_manifest/" 2>/dev/null; then
  echo "!! WRONG STATE: baked prediction has no *_input_copy leaves -- it is not the before state" >&2
  exit 1
fi
if [ ! -f "$REPO_OUT/logs/before/raw/params.json" ]; then
  echo "!! baked prediction lacks raw/params.json -> R6/necessary_share will be null" >&2
  exit 1
fi
echo "== OK. context ready at $OUT"
du -sh "$OUT" "$REPO_OUT/target/release/analyze" "$REPO_OUT/profiling/profile.db" 2>/dev/null || true
