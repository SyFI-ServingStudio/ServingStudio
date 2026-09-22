#!/usr/bin/env bash
# Seed the eval-30947 base workspace from the roofline sglang-30947 clone,
# stripping spoilers that would leak the PR #30947 answer (fused triton
# draft_topk1). Cache-only: keeps warm profile.db + prebuilt target/ + .venv.
set -euo pipefail

WS=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace
SRC="$WS/agent-workspaces/w_68f6afe40d6d/repo"
DST="$WS/agent-workspaces/w_f2c002bbcd89/repo"

echo "[seed] $(date -Is) start  SRC=$SRC  DST=$DST"

# Spoilers excluded:
#   logs/ reports/ tmp/ .git/          -> after runs, analyses, history
#   f0e6ea0f97ce_*.md                  -> original agent's plan/progress = solution
#   sglang-30947-evidence-manifest.md  -> issue evidence writeup
#   cases/sglang-30947/                -> contains the *after* fallback config
#   reference/                         -> pr-repro scripts + findings for 30947
#   tests/test_sglang_30947_configs.py -> issue-named config test
# Kept on purpose (consistent 28103 softness): the *_after arch sources and the
# profiling runner (part of sim infra; discovery still requires the roofline).
rsync -a --delete \
  --exclude='logs/' \
  --exclude='reports/' \
  --exclude='tmp/' \
  --exclude='.git/' \
  --exclude='f0e6ea0f97ce_*' \
  --exclude='sglang-30947-evidence-manifest.md' \
  --exclude='cases/sglang-30947/' \
  --exclude='reference/' \
  --exclude='tests/test_sglang_30947_configs.py' \
  "$SRC/" "$DST/"

echo "[seed] $(date -Is) rsync done; placing clean before-state config"

mkdir -p "$DST/eval-configs" "$DST/logs"
python3 - "$SRC" "$DST" <<'PY'
import json, pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:3])
cfg = json.loads((src/"logs/20260824_sglang30947_predict_before/before-config.json").read_text())
cfg["log_dir"] = "logs/eval_before_run"
cfg["cases_file"] = "eval-configs/cases.json"
(dst/"eval-configs/before.json").write_text(json.dumps(cfg, indent=2))
cases = (src/"logs/20260824_sglang30947_predict_before/matching-cases.json").read_text()
(dst/"eval-configs/cases.json").write_text(cases)
print("[seed] wrote eval-configs/before.json + cases.json")
PY

# belt-and-braces: no 30947 mentions left outside the kept infra files
echo "[seed] residual 30947 mentions (should be none):"
grep -rli "30947" "$DST" --exclude-dir=logs 2>/dev/null || echo "  (clean)"

cd "$DST"
rm -rf .git
git init -q
git checkout -qB main
git config user.name "eval-seed"
git config user.email "eval-seed@example.invalid"
git add -A
git commit -qm "eval base: 30947 before-state (spoilers stripped)"
echo "[seed] $(date -Is) sealed commit: $(git rev-parse --short HEAD)"
