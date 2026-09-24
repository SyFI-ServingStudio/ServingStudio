#!/usr/bin/env bash
# Alternate MLA and KDA continuous rounds on one GPU so two trials never overlap (overlapping
# measurements on a shared device would inject noise into the judge's before/after medians).
# Usage: run_k3_alternate.sh <gpu> <first_round_k> <n_rounds_per_case>
#   K3_SHARED_GPU=1 AGENT_TIMEOUT=2700 ./run_k3_alternate.sh 7 20 3
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="$1"; K0="$2"; N="$3"
for i in $(seq 0 $((N - 1))); do
  k=$((K0 + i))
  "$HERE/run_k3_continuous.sh" "$GPU_IDX" mla "$k" 1
  "$HERE/run_k3_continuous.sh" "$GPU_IDX" kda "$k" 1
done
echo "==== $(date +%F_%T) alternate loop done"
