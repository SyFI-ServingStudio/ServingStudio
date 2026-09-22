#!/usr/bin/env bash
# Seed the eval-17973 base workspace from the roofline vllm-17973 clone,
# stripping spoilers (PR #17973 = optimized Qwen2.5-VL mRoPE preparation).
set -euo pipefail
WS=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
SRC="$WS/agent-workspaces/w_e3c7eebe2924/repo"
DST="$WS/agent-workspaces/w_7eda59064b88/repo"
echo "[seed] $(date -Is) start  SRC=$SRC  DST=$DST"
rsync -a --delete \
  --exclude='logs/' --exclude='reports/' --exclude='tmp/' --exclude='.git/' \
  --exclude='.pytest_cache/' --exclude='__pycache__/' \
  --exclude='5f2c129cd4cb_*' \
  --exclude='cases/vllm-17973/' \
  --exclude='reference/' \
  "$SRC/" "$DST/"
# .gitignore line mentioning the issue: neutralize just that line
sed -i '/17973/d' "$DST/.gitignore" 2>/dev/null || true
echo "[seed] rsync done; placing pinned before config (v19 pair)"
mkdir -p "$DST/eval-configs" "$DST/logs"
python3 - "$SRC" "$DST" <<'PY'
import json, pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:3])
cfg = json.loads((src/"logs/vllm-17973-predict-cataloged-v19/before/qwen2_5_vl_mrope_before_v13_postdemotion.json").read_text())
cfg["log_dir"] = "logs/eval_before_run"
cfg["cases_file"] = "cases.json"
(dst/"eval-configs/before.json").write_text(json.dumps(cfg, indent=2))
cases = (src/"logs/vllm-17973-predict-cataloged-v19/before/prediction.cases.json").read_text()
(dst/"eval-configs/cases.json").write_text(cases)
print("[seed] wrote eval-configs/before.json + cases.json")
PY
if [ ! -f "$DST/alignment/load_generator/req-frontend/Cargo.toml" ]; then
  rmdir "$DST/alignment/load_generator/req-frontend" 2>/dev/null || true
  cp -r "$WS/main/alignment/load_generator/req-frontend" "$DST/alignment/load_generator/req-frontend"
  echo "[seed] vendored req-frontend"
fi
echo "[seed] residual 17973 mentions (should be none):"
grep -rli "17973" "$DST" --exclude-dir=logs 2>/dev/null || echo "  (clean)"
cd "$DST"
rm -rf .git; git init -q; git checkout -qB main
git config user.name "eval-seed"; git config user.email "eval-seed@example.invalid"
git add -A; git commit -qm "eval base: 17973 before-state (spoilers stripped)"
echo "[seed] $(date -Is) sealed commit: $(git rev-parse --short HEAD)"
