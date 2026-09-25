#!/usr/bin/env bash
# Alternate MLA and KDA continuous rounds on one GPU so two trials never overlap (overlapping
# measurements on a shared device would inject noise into the judge's before/after medians).
# Usage: run_k3_alternate.sh <gpu> <first_round_k> <n_rounds_per_case>
#   K3_SHARED_GPU=1 AGENT_TIMEOUT=2700 ./run_k3_alternate.sh 7 20 3
#   CASES="mla_b512 kda_b512" ./run_k3_alternate.sh 7 1 3   (other case pairs; default "mla kda")
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="$1"; K0="$2"; N="$3"; CASES="${CASES:-mla kda}"
for i in $(seq 0 $((N - 1))); do
  k=$((K0 + i))
  for c in $CASES; do "$HERE/run_k3_continuous.sh" "$GPU_IDX" "$c" "$k" 1; done
done
echo "==== $(date +%F_%T) alternate loop done"
