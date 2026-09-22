#!/usr/bin/env bash
# Real-repro eval loop for case vllm-28103.
# Per trial: fresh before+codex container on GPU 3 -> Codex (gpt-5.6-luna, max effort)
# reads the VibeSim diagnosis oracle, edits REAL vLLM, recompiles -> driver deterministically
# rebuilds+installs the agent's tree and runs judge_28103.py (latency-recovered + correctness).
# The agent owns the fix; the driver owns build+install+score so a trial can't be gamed by
# skipping the install step.
#
# Usage: run_repro_eval.sh [N_TRIALS=3] [EFFORT=max]
set -Eeuo pipefail
N="${1:-3}"; EFFORT="${2:-max}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMG=rga-local/vllm-pr-28103:b200-codex
CODEX_HOME_HOST=/raid/yilegu/codex_home_eval
TASK="$HERE/agent_task_28103_repro.md"
PROFILER="$HERE/profile_qk_norm.py"
SNAP=/raid/yilegu/models/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
FICACHE=/raid/yilegu/flashinfer_cache
SITE=/opt/venv/lib/python3.12/site-packages/vllm
OUT="$HERE/repro_eval_28103"; mkdir -p "$OUT" "$FICACHE"
GPU='"device=3"'
AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}"   # 45 min per agent turn

run_trial () {
  local k="$1" cname="eval28103_run_${k}" log="$OUT/trial_${k}"
  echo "==================== TRIAL $k ===================="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run -d --name "$cname" --gpus "$GPU" \
    -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -e CODEX_HOME=/root/.codex-eval \
    -v "$CODEX_HOME_HOST":/root/.codex-eval \
    -v /raid/yilegu/models:/raid/yilegu/models:ro \
    -v "$FICACHE":/root/.cache/flashinfer \
    --workdir /workspace/vllm \
    "$IMG" sleep infinity >/dev/null
  docker cp "$PROFILER" "$cname:/tmp/profile_qk_norm.py"

  echo "== [$k] launching Codex agent (gpt-5.6-luna effort=$EFFORT, timeout ${AGENT_TIMEOUT}s)"
  set +e
  timeout "$AGENT_TIMEOUT" docker exec -i --workdir /workspace/vllm "$cname" \
    codex exec -m gpt-5.6-luna -c model_reasoning_effort="$EFFORT" \
      --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
      "$(cat "$TASK")" > "${log}_agent.log" 2>&1
  local acode=$?
  set -e
  echo "== [$k] agent exit=$acode (transcript: ${log}_agent.log)"

  echo "== [$k] deterministic rebuild of the agent's tree"
  if ! docker exec --workdir /workspace/vllm "$cname" \
        bash -lc 'python3 setup.py build_ext --inplace' > "${log}_build.log" 2>&1; then
    echo "!! [$k] BUILD FAILED -> FAIL"; tail -15 "${log}_build.log"
    echo '{"trial":'"$k"',"verdict":"FAIL","reason":"agent left source non-compiling"}' > "${log}_verdict.json"
    docker rm -f "$cname" >/dev/null 2>&1 || true; return 0
  fi
  echo "== [$k] installing built tree into the wheel (/opt/venv)"
  docker exec "$cname" bash -lc "cp -a /workspace/vllm/vllm/. $SITE/"

  echo "== [$k] judging"
  set +e
  python3 "$HERE/judge_28103.py" --container "$cname" --out "${log}_verdict.json"
  set -e
  # tag the verdict with the trial number + agent exit
  python3 - "$k" "$acode" "${log}_verdict.json" <<'PY'
import json,sys
k,ac,f=sys.argv[1:4]
d=json.load(open(f)); d["trial"]=int(k); d["agent_exit"]=int(ac)
json.dump(d,open(f,"w"),indent=2)
PY
  docker rm -f "$cname" >/dev/null 2>&1 || true
  echo "== [$k] done -> ${log}_verdict.json"
}

for k in $(seq 1 "$N"); do run_trial "$k"; done

echo "==================== SUMMARY ===================="
python3 - "$OUT" "$N" <<'PY'
import json,glob,os,sys
out,N=sys.argv[1],int(sys.argv[2])
rows=[]
for f in sorted(glob.glob(os.path.join(out,"trial_*_verdict.json"))):
    d=json.load(open(f)); rows.append(d)
p=sum(1 for d in rows if d.get("verdict")=="PASS")
for d in rows:
    print(f"  trial {d.get('trial')}: {d.get('verdict'):4s} "
          f"recovered={d.get('recovered_frac')} lat={d.get('latency_us')}us "
          f"correct={d.get('correctness',{}).get('pass')} reason={d.get('reason')}")
print(f"  PASS {p}/{len(rows)}")
PY
