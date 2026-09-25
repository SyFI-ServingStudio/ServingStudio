#!/usr/bin/env bash
# Claude (Opus 5.5) large-batch rounds on TWO free GPUs in parallel: KDA on one, MLA on the other
# (different devices -> no measurement interference). Usage: launch_claude_b512_parallel.sh <k0> <n>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; K0="$1"; N="$2"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}"
G1="$("$HERE/pick_free_gpu.sh")"; G2="$("$HERE/pick_free_gpu.sh" "$G1")"
echo "==== $(date +%F_%T) GPUs: kda_b512_claude -> $G1, mla_b512_claude -> $G2"
for c in kda mla; do
  SRC="$HERE/iter_opt_eval_k3_${c}"; OUT="$HERE/iter_opt_eval_k3_${c}_b512_claude"; mkdir -p "$OUT"
  [ -d "$OUT/pristine_tree" ] || cp -a "$SRC/pristine_tree" "$OUT/pristine_tree"
  [ -d "$OUT/best_tree" ] || { cp -a "$SRC/best_tree" "$OUT/best_tree"; echo "==== seeded ${c}_b512_claude best_tree from $SRC/best_tree"; }
done
"$HERE/run_k3_continuous.sh" "$G1" kda_b512_claude "$K0" "$N" > "$HERE/k3_claude_kda_b512_rounds.log" 2>&1 &
P1=$!
"$HERE/run_k3_continuous.sh" "$G2" mla_b512_claude "$K0" "$N" > "$HERE/k3_claude_mla_b512_rounds.log" 2>&1 &
P2=$!
wait $P1 $P2
echo "==== $(date +%F_%T) both Claude b512 loops done"
