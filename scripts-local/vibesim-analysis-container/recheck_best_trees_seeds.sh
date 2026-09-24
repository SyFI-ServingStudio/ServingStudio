#!/usr/bin/env bash
# Accuracy hardening: re-judge the final best trees against FRESH pristine goldens under extra
# seeds (new weights/state/inputs). The continuous loop only ever checked seed 0. Verdict files:
#   iter_opt_eval_k3_<case>/best_tree_seed<k>_verdict.json  (improvement is vs the PRISTINE tree)
# Usage: recheck_best_trees_seeds.sh <gpu> <seed> [<seed> ...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="$1"; shift
export DOCKER_GPU_ARG="\"device=$GPU_IDX\""
for seed in "$@"; do
  for case_name in mla kda; do
    OUT="$HERE/iter_opt_eval_k3_${case_name}"
    CFG="$OUT/issue_k3_${case_name}_seed${seed}.json"
    python3 - "$HERE/issue_k3_${case_name}.json" "$CFG" "$seed" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["seed"] = int(sys.argv[3]); c["min_improvement"] = 0.0
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
    echo "==== $(date +%F_%T) recheck $case_name best_tree, seed $seed"
    python3 "$HERE/judge_k3.py" --agent-container "recheck_${case_name}_seed${seed}" --config "$CFG" \
      --golden-dir /raid/yilegu/eval_goldens --out "$OUT/best_tree_seed${seed}_verdict.json" \
      --tree-dir "$OUT/best_tree" --pristine-dir "$OUT/pristine_tree" > "$OUT/best_tree_seed${seed}_judge.log" 2>&1
    python3 - "$OUT/best_tree_seed${seed}_verdict.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("verdict"), (d.get("reason") or "")[:120])
for k, p in (d.get("points") or {}).items():
    print(f"  {k}: pristine {p['before_us']:.1f} -> best {p['after_us']:.1f} us ({p['improvement']*100:+.1f}%), "
          f"CHECK {p['correctness'].get('pass')} rel {p['correctness'].get('max_rel_err')}")
PY
  done
done
echo "==== $(date +%F_%T) recheck done"
