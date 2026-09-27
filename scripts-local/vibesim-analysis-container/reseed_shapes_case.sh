#!/usr/bin/env bash
# Re-run the seed transfer check of a shape-matched case after an infrastructure fix (2026-09-26: the judge's
# read-only FlashInfer cache mount made the MLA 1x1M autotune write fail -> false CHECK failures). Same
# candidate chain as run_k3_shapes_slurm.sh; on PASS installs best_tree (picked up by the next round).
# Usage: reseed_shapes_case.sh <kda|mla>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; c="$1"
OUT="$HERE/iter_opt_eval_k3_${c}_shapes_claude"; GOLDEN_DIR=/raid/yilegu/eval_goldens
K3_SLURM_PARTITION="${K3_SLURM_PARTITION:-main}"
[ -d "$OUT/best_tree" ] && { echo "best_tree already present in $OUT"; exit 0; }
for cand in "$HERE/iter_opt_eval_k3_${c}_b512_claude/best_tree" "$HERE/iter_opt_eval_k3_${c}/best_tree"; do
  [ -d "$cand" ] || continue
  tag="$(basename "$(dirname "$cand")")"
  echo "==== $(date +%F_%T) ${c}_shapes_claude: transfer check (retry) of $tag/best_tree"
  rm -rf "$OUT/seed_candidate" "$OUT/seed_candidate_judged"; cp -a "$cand" "$OUT/seed_candidate"
  outf="$OUT/seed_transfer_${tag}.retry.slurm.out"
  job="$(sbatch --parsable --partition="$K3_SLURM_PARTITION" --job-name="k3t_${c}_shapes" --output="$outf" \
        "$HERE/slurm_gpu.sh" "$HERE/k3_slurm_step.sh" python3 "$HERE/judge_k3.py" \
        --agent-container transfer_check --config "$HERE/issue_k3_${c}_shapes_claude.json" \
        --golden-dir "$GOLDEN_DIR" --out "$OUT/seed_transfer_${tag}_verdict.json" \
        --tree-dir "$OUT/seed_candidate" --pristine-dir "$OUT/pristine_tree" --min-improvement 0 | cut -d';' -f1)"
  echo "$(date +%T) slurm job $job -> $outf"
  while squeue -h -j "$job" -o %T 2>/dev/null | grep -q .; do sleep 20; done; sleep 2
  ok="$(python3 - "$OUT/seed_transfer_${tag}_verdict.json" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: print("no"); sys.exit()
pts = d.get("points") or {}
for k, p in pts.items():
    c = p["correctness"]
    print(f"  {k}: pristine {p['before_us']:.1f} -> seed {p['after_us']:.1f} us ({p['improvement']*100:+.1f}%) CHECK {c.get('pass')} rel {c.get('max_rel_err')}", file=sys.stderr)
print("yes" if pts and all(p["correctness"].get("pass") for p in pts.values()) else "no")
PY
)"
  if [ "$ok" = yes ]; then
    mv "$OUT/seed_candidate" "$OUT/best_tree"; rm -rf "$OUT/seed_candidate_judged"
    echo "==== $(date +%F_%T) ${c}_shapes_claude seeded from $tag/best_tree"; exit 0
  fi
  mv "$OUT/seed_candidate" "$OUT/seed_candidate_failed_${tag}.retry"; rm -rf "$OUT/seed_candidate_judged"
  echo "!! $tag/best_tree fails CHECK on the new points (see $outf)"
done
echo "!! no candidate passed; rounds start from PRISTINE"; exit 1
