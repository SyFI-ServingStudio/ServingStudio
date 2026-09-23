#!/usr/bin/env bash
# After the direct trial queue finishes on GPU 3: (1) re-judge KDA trial 7 with the judge that
# now carries the agent's JIT CUDA source edits (its v1 verdict dropped kda_fused_decode.cuh and
# crashed on the pristine kernel -> preserved as trial_7_verdict_v1.json), then (2) run the B8
# row fill. Both are GPU-3 work and must not overlap the trials' timing measurements.
#   setsid nohup ./after_trials_rejudge_and_fill.sh > codex_runs/after_trials_rejudge_and_fill.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_IDX="${K3_GPU_INDEX:-3}"
export DOCKER_GPU_ARG="\"device=$GPU_IDX\""
wait_idle () {
  while pgrep -f "[r]un_k3_trials_direct.sh" >/dev/null; do
    echo "$(date +%T) trials still running -> wait 120s"; sleep 120
  done
  while :; do
    used="$(nvidia-smi -i "$GPU_IDX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    [ "${used:-99999}" -lt 1000 ] && return 0
    echo "$(date +%T) GPU $GPU_IDX busy (${used} MiB) -> wait 60s"; sleep 60
  done
}

wait_idle
OUT="$HERE/iter_opt_eval_k3_kda"
[ -f "$OUT/trial_7_verdict_v1.json" ] || cp "$OUT/trial_7_verdict.json" "$OUT/trial_7_verdict_v1.json"
echo "==== $(date +%F_%T) re-judging KDA trial 7 (source diff incl. .cuh) on GPU $GPU_IDX"
python3 "$HERE/judge_k3.py" --agent-container iteropt_k3_kda_run_7_rejudge --config "$HERE/issue_k3_kda.json" \
  --golden-dir /raid/yilegu/eval_goldens --out "$OUT/trial_7_verdict.json" \
  --tree-dir "$OUT/trial_7_tree" --pristine-dir "$OUT/pristine_tree" > "$OUT/trial_7_rejudge.log" 2>&1
echo "==== $(date +%F_%T) re-judge exit=$? -> $(python3 -c "import json;d=json.load(open('$OUT/trial_7_verdict.json'));print(d.get('verdict'),(d.get('reason') or '')[:200])")"
python3 - "$OUT/trial_7_verdict.json" <<'PY'
import json,sys
f=sys.argv[1]; d=json.load(open(f)); d["trial"]=7; d["rejudged"]="v2: source diff incl. JIT CUDA (.cuh)"
json.dump(d,open(f,"w"),indent=2)
PY

wait_idle
echo "==== $(date +%F_%T) JIT fill (repo main-k3-rust @ $(git -C "$HERE/../../main-k3-rust" rev-parse --short HEAD))"
K3_GPU_INDEX="$GPU_IDX" "$HERE/run_k3_jit_fill.sh" \
  presets/predict_kimi_k3_b200_rank1_layer_kda.json \
  presets/predict_kimi_k3_b200_rank1_layer_mla.json
echo "==== $(date +%F_%T) fill done exit=$?"
