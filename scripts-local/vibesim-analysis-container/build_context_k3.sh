#!/usr/bin/env bash
# Stage the Docker build context for the Kimi-K3 VibeSim analysis oracle (vibesim-analysis:k3).
#
# Same shape as build_context.sh (fixed prediction oracle: 5 agent-facing verbs, prebuilt
# `analyze` binary, no Rust/PyO3 at runtime) but sourced from the kimi-k3 branch worktree:
#   * repo assets at HEAD of the worktree (gpu, model incl. K3 label + location maps,
#     simulator source for workspace-info arch stems, eval-configs, K3 presets)
#   * the BRANCH profile.db (K3 B200 rows; the shared profiling/profile.db is untouched)
#   * the prebuilt analyze binary from the worktree
#   * baked predictions: logs/k3_kda (rank-1 KDA layer, B=1/32/128@8k) and logs/k3_mla
#     (rank-1 MLA layer); the container serves ONE of them per VIBESIM_PREBAKED_RUN, so
#     the KDA and MLA cases run two containers from the same image (ports 8801 / 8802).
# Open-ended case: there is no "after" answer key to purge; the agent's own edits never
# reach this container (it only re-serves the baked before-state prediction).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC_REPO="${SRC_REPO:-/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust}"
API_PKG="${API_PKG:-/raid/yilegu/roofline_guided_agent/vibesim-api-package}"
PROFILE_DB="${PROFILE_DB:-$HERE/kimi_single_layer/k3_branch_profile.db}"
OUT="${OUT:-$HERE/context_k3}"
REPO_OUT="$OUT/repo"
declare -A BAKE=( [k3_kda]=predict_kimi_k3_b200_rank1_layer_kda [k3_mla]=predict_kimi_k3_b200_rank1_layer_mla
                  [k3_kda_b512]=predict_kimi_k3_b200_rank1_layer_kda_b512 [k3_mla_b512]=predict_kimi_k3_b200_rank1_layer_mla_b512
                  [k3_kda_prefill]=predict_kimi_k3_b200_rank1_layer_kda_prefill [k3_mla_prefill]=predict_kimi_k3_b200_rank1_layer_mla_prefill )

echo "== staging into $OUT from $SRC_REPO @ $(git -C "$SRC_REPO" rev-parse --short HEAD) ($(git -C "$SRC_REPO" branch --show-current))"
rm -rf "$OUT"; mkdir -p "$REPO_OUT" "$OUT/app"

echo "== 1. tracked assets at HEAD"
# (upstream has no eval-configs/; the server's `config` default only matters for a live
#  simulate, which the prebaked shim short-circuits -- point it at the K3 preset anyway)
git -C "$SRC_REPO" archive HEAD -- gpu model simulator presets | tar -x -C "$REPO_OUT"
mkdir -p "$REPO_OUT/eval-configs"
cp "$REPO_OUT/presets/predict_kimi_k3_b200_rank1_layer_kda.json" "$REPO_OUT/eval-configs/before.json"
find "$REPO_OUT" -maxdepth 2 -type f \( -name '*_plan.md' -o -name '*_progress.md' \) -delete

echo "== 2. branch profile.db + analyze binary"
mkdir -p "$REPO_OUT/profiling" "$REPO_OUT/target/release"
cp "$PROFILE_DB" "$REPO_OUT/profiling/profile.db"
cp "$SRC_REPO/target/release/analyze" "$REPO_OUT/target/release/analyze"

echo "== 3. baked K3 predictions"
mkdir -p "$REPO_OUT/logs"
baked=0
for run in "${!BAKE[@]}"; do
  src="$SRC_REPO/logs/${BAKE[$run]}"
  if [ -f "$src/prediction.meta.json" ] && [ -f "$src/raw/params.json" ]; then
    cp -a "$src/." "$REPO_OUT/logs/$run/"
    rm -f "$REPO_OUT/logs/$run/reports/optimality_scoped_"*.json 2>/dev/null || true
    echo "   baked $run <- ${BAKE[$run]} ($(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('prediction_id'))" "$src/prediction.meta.json"))"
    baked=$((baked+1))
  else
    echo "   SKIP $run: $src has no complete prediction yet"
  fi
done
[ "$baked" -gt 0 ] || { echo "!! no baked prediction available"; exit 1; }

echo "== 4. API package + app (prebaked simulate) + skill"
mkdir -p "$OUT/vibesim-api"; cp -a "$API_PKG/." "$OUT/vibesim-api/"
cp "$API_PKG/vibesim_api.py" "$OUT/app/vibesim_api.py"
python3 "$HERE/patch_prebaked.py" "$OUT/app/vibesim_api.py"
cp "$HERE/app/server.py" "$OUT/app/server.py"
cp "$HERE/app/entrypoint.sh" "$OUT/app/entrypoint.sh"
cp "$HERE/app/uv" "$OUT/app/uv"
cp -a "$HERE/skill" "$OUT/skill"

echo "== OK. context ready at $OUT"
du -sh "$OUT" "$REPO_OUT/profiling/profile.db" "$REPO_OUT/target/release/analyze" 2>/dev/null || true
echo "build:  sed 's#COPY context/#COPY context_k3/#' $HERE/Dockerfile | docker build -t vibesim-analysis:k3 -f - $HERE"
echo "run:    docker run -d --name vibesim_oracle_k3_kda -e VIBESIM_PREBAKED_RUN=k3_kda -e VIBESIM_API_PORT=8801 -p 172.17.0.1:8801:8801 vibesim-analysis:k3"
echo "        docker run -d --name vibesim_oracle_k3_mla -e VIBESIM_PREBAKED_RUN=k3_mla -e VIBESIM_API_PORT=8802 -p 172.17.0.1:8802:8802 vibesim-analysis:k3"
echo "        large-batch cases: k3_kda_b512 -> 8803, k3_mla_b512 -> 8804 (same image, VIBESIM_PREBAKED_RUN=k3_{kda,mla}_b512)"
