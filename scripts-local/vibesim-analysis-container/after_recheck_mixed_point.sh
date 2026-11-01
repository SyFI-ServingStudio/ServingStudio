#!/usr/bin/env bash
# After the multi-seed recheck finishes: judge the MLA best_tree on the NEW mixed-length point
# (issue_k3_mla.json now has [16, 65536, "mix"]). Expected: round 25's hardcoded is_var_seq=False
# fails CHECK there. If so, retire that tree (best_tree.r25_overfit), restore the round-24 tree
# (best_tree.prev) as best_tree, and judge it too. Verdicts: iter_opt_eval_k3_mla/best_tree_mixed_*.json
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="${K3_GPU_INDEX:-7}"
export DOCKER_GPU_ARG="\"device=$GPU_IDX\""
OUT="$HERE/iter_opt_eval_k3_mla"
while pgrep -f "[r]echeck_best_trees_seeds" >/dev/null; do sleep 60; done
judge () {  # $1 = tree dir, $2 = out json
  python3 "$HERE/judge_k3.py" --agent-container mixed_point_check --config "$HERE/issue_k3_mla.json" \
    --golden-dir /raid/yilegu/eval_goldens --out "$2" --tree-dir "$1" --pristine-dir "$OUT/pristine_tree" \
    --min-improvement 0 > "${2%.json}.log" 2>&1
  python3 - "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); print(d.get("verdict"), (d.get("reason") or "")[:160])
for k, p in (d.get("points") or {}).items():
    print(f"  {k}: pristine {p['before_us']:.1f} -> tree {p['after_us']:.1f} us ({p['improvement']*100:+.1f}%), "
          f"CHECK {p['correctness'].get('pass')} rel {p['correctness'].get('max_rel_err')}")
PY
}
echo "==== $(date +%F_%T) judging current MLA best_tree (round-25 tree) on the mixed-length point"
judge "$OUT/best_tree" "$OUT/best_tree_mixed_verdict.json"
if python3 -c "import json,sys; d=json.load(open('$OUT/best_tree_mixed_verdict.json')); p=(d.get('points') or {}).get('16,65536mix'); sys.exit(0 if (p and not p['correctness'].get('pass')) or d.get('verdict')=='FAIL' and 'correctness' in (d.get('reason') or '') else 1)"; then
  echo "==== mixed-length CHECK FAILED on the round-25 tree -> retiring it, restoring the round-24 tree"
  rm -rf "$OUT/best_tree.r25_overfit"; mv "$OUT/best_tree" "$OUT/best_tree.r25_overfit"
  cp -a "$OUT/best_tree.prev" "$OUT/best_tree"
  echo "==== $(date +%F_%T) judging the restored round-24 tree on the mixed-length point"
  judge "$OUT/best_tree" "$OUT/best_tree_mixed_verdict_v2.json"
else
  echo "==== round-25 tree passed the mixed-length point (unexpected) -> best_tree kept"
fi
echo "==== $(date +%F_%T) mixed-point check done"
