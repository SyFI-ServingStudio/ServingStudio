#!/usr/bin/env bash
# Large-batch rounds with the CLAUDE agent (Opus 5.5 via Bedrock; cases k3_{kda,mla}_b512_claude), warm-started
# from the decode best trees (already transfer-checked on the b512 points -> no new transfer check; goldens are
# shared with the Codex b512 cases via the same case key). GPU: 2 or 3 after IDLE_CHECKS consecutive idle
# readings (the user's neighbours cycle every ~10 min), GPU 7 only with >= 60 GB free.
# Usage: run_k3_claude_b512_when_free.sh <first_round_k> <n_rounds_per_case>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
K0="$1"; N="$2"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}"
IDLE_CHECKS="${IDLE_CHECKS:-30}"
declare -A IDLE_STREAK=([2]=0 [3]=0)
GPU_IDX=""
while [ -z "$GPU_IDX" ]; do
  for g in 2 3; do
    used="$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [ "${used:-99999}" -lt 1000 ]; then IDLE_STREAK[$g]=$((IDLE_STREAK[$g] + 1)); else IDLE_STREAK[$g]=0; fi
    if [ "${IDLE_STREAK[$g]}" -ge "$IDLE_CHECKS" ]; then GPU_IDX="$g"; break; fi
  done
  [ -n "$GPU_IDX" ] && break
  used7="$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  total7="$(nvidia-smi -i 7 --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
  if [ $((total7 - used7)) -ge 61440 ]; then GPU_IDX=7; export K3_SHARED_GPU=1; break; fi
  echo "$(date +%T) idle streaks 2=${IDLE_STREAK[2]} 3=${IDLE_STREAK[3]} (need $IDLE_CHECKS), GPU 7 free=$((total7 - used7)) MiB -> wait 120s"; sleep 120
done
echo "==== $(date +%F_%T) using GPU $GPU_IDX (shared=${K3_SHARED_GPU:-0})"
export DOCKER_GPU_ARG="\"device=$GPU_IDX\""
for c in kda mla; do
  SRC="$HERE/iter_opt_eval_k3_${c}"; OUT="$HERE/iter_opt_eval_k3_${c}_b512_claude"; mkdir -p "$OUT"
  [ -d "$OUT/pristine_tree" ] || cp -a "$SRC/pristine_tree" "$OUT/pristine_tree"
  [ -d "$OUT/best_tree" ] || { cp -a "$SRC/best_tree" "$OUT/best_tree"; echo "==== seeded ${c}_b512_claude best_tree from $SRC/best_tree"; }
done
echo "==== $(date +%F_%T) Claude rounds $K0..$((K0+N-1)) for kda_b512_claude mla_b512_claude on GPU $GPU_IDX"
CASES="kda_b512_claude mla_b512_claude" exec "$HERE/run_k3_alternate.sh" "$GPU_IDX" "$K0" "$N"
