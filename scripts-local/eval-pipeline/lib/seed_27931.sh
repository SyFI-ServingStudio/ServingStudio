#!/usr/bin/env bash
# Seed the eval-27931 base workspace from the roofline vllm-27931 clone,
# stripping spoilers that would leak the PR #27931 answer (vectorized RMSNorm).
# Cache-only: keeps warm profile.db + prebuilt target/ + .venv.
set -euo pipefail

WS=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
SRC="$WS/agent-workspaces/w_befc5365ebdc/repo"
DST="$WS/agent-workspaces/w_1a73838d022f/repo"

echo "[seed] $(date -Is) start  SRC=$SRC  DST=$DST"

rsync -a --delete \
  --exclude='logs/' \
  --exclude='reports/' \
  --exclude='tmp/' \
  --exclude='.git/' \
  --exclude='.pytest_cache/' \
  --exclude='__pycache__/' \
  --exclude='fdada40c0ad7_*' \
  --exclude='vllm_27931_*' \
  --exclude='HOST-NOTES.md' \
  --exclude='reference/' \
  "$SRC/" "$DST/"

echo "[seed] $(date -Is) rsync done; sanitizing PR-number comments"
# comment-only edits, no behavior change
sed -i 's/because PR #27931/because the kernel implementation/' "$DST/simulator/src/timing/kernels/rms_norm_vllm.rs"
sed -i 's/matching PR #27931 before its/matching the reference recipe before its/; s/matching PR #27931 after its/matching the reference recipe after its/' "$DST/simulator/src/arch/build.rs"

echo "[seed] placing clean before-state config (llama3 4096-prefill)"
mkdir -p "$DST/eval-configs" "$DST/logs"
python3 - "$SRC" "$DST" <<'PY'
import json, pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:3])
cfg = json.loads((src/"logs/20260824_20_llama3_4096_prefill_timing_predict_before/before.json").read_text())
cfg["log_dir"] = "logs/eval_before_run"
cfg["cases_file"] = "cases.json"
(dst/"eval-configs/before.json").write_text(json.dumps(cfg, indent=2))
cases = (src/"logs/20260824_20_llama3_4096_prefill_timing_predict_before/cases.json").read_text()
(dst/"eval-configs/cases.json").write_text(cases)
print("[seed] wrote eval-configs/before.json + cases.json")
PY

# vendor req-frontend if the bind-mount point is empty (host builds need it)
if [ ! -f "$DST/alignment/load_generator/req-frontend/Cargo.toml" ]; then
  rmdir "$DST/alignment/load_generator/req-frontend" 2>/dev/null || true
  cp -r "$WS/main/alignment/load_generator/req-frontend" "$DST/alignment/load_generator/req-frontend"
  echo "[seed] vendored req-frontend from main/"
fi

echo "[seed] residual 27931 mentions (should be none):"
grep -rli "27931" "$DST" --exclude-dir=logs 2>/dev/null || echo "  (clean)"

cd "$DST"
rm -rf .git
git init -q
git checkout -qB main
git config user.name "eval-seed"
git config user.email "eval-seed@example.invalid"
git add -A
git commit -qm "eval base: 27931 before-state (spoilers stripped)"
echo "[seed] $(date -Is) sealed commit: $(git rev-parse --short HEAD)"
