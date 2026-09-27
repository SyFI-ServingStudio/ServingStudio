#!/usr/bin/env bash
# Shape-matched Claude campaign (user 2026-09-26): cases k3_{kda,mla}_shapes_claude judge BOTH layers on the
# same five decode shapes (1x1M primary; 16x64k[,mix for MLA]; 128x8k; 32x8k; 1x8k), agent = Claude Opus 5.5,
# GPU work ONLY as slurm `main` jobs (agent container has no GPU; see run_iter_opt_eval.sh K3_GPU_MODE=slurm).
# Per case:
#   1. seed best_tree: try the most complete existing tree first (b512 Claude best = decode best + Claude
#      large-batch levers), then the decode best tree; each candidate is TRANSFER-CHECKED on the new points
#      by the judge (one sbatch job: pristine goldens/baselines for the new case key + candidate replay);
#      CHECK fail -> next candidate; none -> start from PRISTINE.
#   2. alternate continuous rounds kda_shapes_claude / mla_shapes_claude (run_k3_alternate.sh ... slurm).
# Usage: run_k3_shapes_slurm.sh <first_round_k> <n_rounds_per_case>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
K0="$1"; N="$2"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}" K3_SLURM_PARTITION="${K3_SLURM_PARTITION:-main}"
GOLDEN_DIR=/raid/yilegu/eval_goldens

slurm_judge () { # $1 name $2 out $3... judge args
  local name="$1" outf="$2"; shift 2
  local job; job="$(sbatch --parsable --partition="$K3_SLURM_PARTITION" --job-name="$name" --output="$outf" \
                 "$HERE/slurm_gpu.sh" "$HERE/k3_slurm_step.sh" python3 "$HERE/judge_k3.py" "$@" | cut -d';' -f1)"
  [ -n "$job" ] || { echo "!! sbatch failed ($name)"; return 1; }
  echo "$(date +%T) slurm job $job ($name) -> $outf"
  while squeue -h -j "$job" -o %T 2>/dev/null | grep -q .; do sleep 20; done; sleep 2
  local rc; rc="$(grep -a '^\[step\] EXIT_RC=' "$outf" 2>/dev/null | tail -1 | cut -d= -f2)"
  echo "$(date +%T) job $job done rc=${rc:-?}"; return "${rc:-1}"
}

for c in kda mla; do
  OUT="$HERE/iter_opt_eval_k3_${c}_shapes_claude"; mkdir -p "$OUT"
  [ -d "$OUT/pristine_tree" ] || cp -a "$HERE/iter_opt_eval_k3_${c}/pristine_tree" "$OUT/pristine_tree"
  [ -d "$OUT/best_tree" ] && { echo "==== ${c}_shapes_claude: best_tree exists, skipping seed"; continue; }
  for cand in "$HERE/iter_opt_eval_k3_${c}_b512_claude/best_tree" "$HERE/iter_opt_eval_k3_${c}/best_tree"; do
    [ -d "$cand" ] || continue
    tag="$(basename "$(dirname "$cand")")"
    echo "==== $(date +%F_%T) ${c}_shapes_claude: transfer check of $tag/best_tree on the shape-matched points"
    rm -rf "$OUT/seed_candidate" "$OUT/seed_candidate_judged"; cp -a "$cand" "$OUT/seed_candidate"
    slurm_judge "k3t_${c}_shapes" "$OUT/seed_transfer_${tag}.slurm.out" \
      --agent-container transfer_check --config "$HERE/issue_k3_${c}_shapes_claude.json" \
      --golden-dir "$GOLDEN_DIR" --out "$OUT/seed_transfer_${tag}_verdict.json" \
      --tree-dir "$OUT/seed_candidate" --pristine-dir "$OUT/pristine_tree" --min-improvement 0
    ok="$(python3 - "$OUT/seed_transfer_${tag}_verdict.json" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("no"); sys.exit()
pts = d.get("points") or {}
for k, p in pts.items():
    c = p["correctness"]
    print(f"  {k}: pristine {p['before_us']:.1f} -> seed {p['after_us']:.1f} us ({p['improvement']*100:+.1f}%) CHECK {c.get('pass')} rel {c.get('max_rel_err')}", file=sys.stderr)
print("yes" if pts and all(p["correctness"].get("pass") for p in pts.values()) else "no")
PY
)"
    if [ "$ok" = yes ]; then
      mv "$OUT/seed_candidate" "$OUT/best_tree"; rm -rf "$OUT/seed_candidate_judged"
      echo "==== ${c}_shapes_claude seeded from $tag/best_tree"; break
    else
      mv "$OUT/seed_candidate" "$OUT/seed_candidate_failed_${tag}"; rm -rf "$OUT/seed_candidate_judged"
      echo "!! ${c}_shapes_claude: $tag/best_tree fails CHECK (or no result) on the new points -> next candidate"
    fi
  done
  [ -d "$OUT/best_tree" ] || echo "!! ${c}_shapes_claude: no candidate passed -> rounds start from PRISTINE"
done
echo "==== $(date +%F_%T) starting alternate slurm rounds $K0..$((K0+N-1)) for kda_shapes_claude mla_shapes_claude"
CASES="kda_shapes_claude mla_shapes_claude" exec "$HERE/run_k3_alternate.sh" slurm "$K0" "$N"
