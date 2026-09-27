#!/usr/bin/env bash
# Generic slurm-mode campaign for a KDA/MLA case pair `k3_{kda,mla}_<suffix>` (generalizes run_k3_shapes_slurm.sh):
#   1. per layer, seed best_tree from the first CANDIDATE case whose best_tree passes a TRANSFER CHECK on the new
#      points (one sbatch judge job each; pristine goldens/baselines for the new case key are created on the way);
#      none -> start from PRISTINE;
#   2. alternate continuous Claude rounds kda_<suffix> / mla_<suffix> (agent container without GPU, every
#      measurement and the judge as sbatch jobs on partition main).
# Usage: CANDIDATES="prefill_claude shapes_claude" run_k3_pair_slurm.sh <suffix> <first_round_k> <n_rounds>
#   CANDIDATES = case suffixes (of iter_opt_eval_k3_{kda,mla}_<cand>) tried in order; "" = the decode case itself.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUFFIX="$1"; K0="$2"; N="$3"; CANDS="${CANDIDATES:-}"
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
  OUT="$HERE/iter_opt_eval_k3_${c}_${SUFFIX}"; mkdir -p "$OUT"
  [ -d "$OUT/pristine_tree" ] || cp -a "$HERE/iter_opt_eval_k3_${c}/pristine_tree" "$OUT/pristine_tree"
  [ -d "$OUT/best_tree" ] && { echo "==== ${c}_${SUFFIX}: best_tree exists, skipping seed"; continue; }
  for cs in $CANDS; do
    cand="$HERE/iter_opt_eval_k3_${c}_${cs}/best_tree"; [ "$cs" = "-" ] && cand="$HERE/iter_opt_eval_k3_${c}/best_tree"
    [ -d "$cand" ] || { echo "(no $cand)"; continue; }
    tag="$(basename "$(dirname "$cand")")"
    echo "==== $(date +%F_%T) ${c}_${SUFFIX}: transfer check of $tag/best_tree on the new points"
    rm -rf "$OUT/seed_candidate" "$OUT/seed_candidate_judged"; cp -a "$cand" "$OUT/seed_candidate"
    slurm_judge "k3t_${c}_${SUFFIX}" "$OUT/seed_transfer_${tag}.slurm.out" \
      --agent-container transfer_check --config "$HERE/issue_k3_${c}_${SUFFIX}.json" \
      --golden-dir "$GOLDEN_DIR" --out "$OUT/seed_transfer_${tag}_verdict.json" \
      --tree-dir "$OUT/seed_candidate" --pristine-dir "$OUT/pristine_tree" --min-improvement 0
    ok="$(python3 - "$OUT/seed_transfer_${tag}_verdict.json" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: print("no"); sys.exit()
pts = d.get("points") or {}
for k, p in pts.items():
    c = p["correctness"]
    print(f"  {k}: pristine {p['before_us']:.1f} -> seed {p['after_us']:.1f} us ({p['improvement']*100:+.1f}%) CHECK {c.get('pass')} rel {c.get('max_rel_err')}", file=sys.stderr)
if not pts: print((d.get("reason") or "no points")[:300], file=sys.stderr)
print("yes" if pts and all(p["correctness"].get("pass") for p in pts.values()) else "no")
PY
)"
    if [ "$ok" = yes ]; then
      mv "$OUT/seed_candidate" "$OUT/best_tree"; rm -rf "$OUT/seed_candidate_judged"
      echo "==== ${c}_${SUFFIX} seeded from $tag/best_tree"; break
    fi
    mv "$OUT/seed_candidate" "$OUT/seed_candidate_failed_${tag}"; rm -rf "$OUT/seed_candidate_judged"
    echo "!! ${c}_${SUFFIX}: $tag/best_tree fails CHECK (or no result) on the new points -> next candidate"
  done
  [ -d "$OUT/best_tree" ] || echo "!! ${c}_${SUFFIX}: no candidate passed -> rounds start from PRISTINE"
done
echo "==== $(date +%F_%T) starting alternate slurm rounds $K0..$((K0+N-1)) for kda_${SUFFIX} mla_${SUFFIX}"
CASES="kda_${SUFFIX} mla_${SUFFIX}" exec "$HERE/run_k3_alternate.sh" slurm "$K0" "$N"
