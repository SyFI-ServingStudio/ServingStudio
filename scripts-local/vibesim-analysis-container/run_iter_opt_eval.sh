#!/usr/bin/env bash
# Iterative optimizer eval (de-leaked, config-driven, multi-case).
#
# The agent is told ONLY "optimize metric X (e.g. per-step decode latency) for this model on
# this workload" -- no operator name, no target files, no target number, and NO hint about
# "necessary/unnecessary/removable work". It must DERIVE what to change from the VibeSim
# roofline analysis, map it to the real source, edit + recompile, and prove the layer got
# faster without changing its numerical output. It organizes intermediate results by iteration
# in /workspace/opt_run/iter_NN/.
#
# Per trial: fresh codex container on GPU 3 -> Codex (gpt-5.6-luna, max) drives the loop against
# the VibeSim oracle -> DRIVER deterministically rebuilds+installs the agent's tree and runs
# judge_general.py (holds the private target answer-key; captures its own golden from a pristine
# before-image). The agent's iteration workspace + transcript are saved for reward-leak audit.
#
# Usage: run_iter_opt_eval.sh [CONFIG=issue_28103.json] [N_TRIALS=3] [EFFORT=max]
#   START=<k>       resume from trial k (preserves earlier trials' evidence)
#   AGENT_TIMEOUT=<s>  per-agent-turn budget (default 3600)
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$HERE/issue_28103.json}"; [ -f "$CONFIG" ] || CONFIG="$HERE/$1"
N="${2:-3}"; EFFORT="${3:-max}"
CODEX_HOME_HOST=/raid/yilegu/codex_home_eval
TASK_TMPL="$HERE/agent_task_iter_opt.md"
EXTRACTOR="$HERE/extract_and_profile.py"
FICACHE=/raid/yilegu/flashinfer_cache
GOLDEN_DIR=/raid/yilegu/eval_goldens
AGENT_TIMEOUT="${AGENT_TIMEOUT:-3600}"

# ---- load config fields (single python read -> shell vars) --------------------------------
eval "$(python3 - "$CONFIG" <<'PY'
import json,sys,shlex
c=json.load(open(sys.argv[1])); r=c.get("render",{})
def e(k,v): print(f'{k}={shlex.quote(str(v))}')
e("IMG", c["codex_image"]); e("SNAP", c["model_snap"]); e("GPU_DEV", c.get("gpu","3"))
e("CASE", c.get("case", __import__("os").path.splitext(__import__("os").path.basename(sys.argv[1]))[0]))
e("R_METRIC", r.get("METRIC","per-step decode latency"))
e("R_FRAMEWORK", r.get("FRAMEWORK","vllm"))
e("R_CHECKOUT", r.get("CHECKOUT","/workspace/vllm"))
e("R_MODEL_NAME", r.get("MODEL_NAME","the model"))
e("R_WORKLOAD", r.get("WORKLOAD","short-prompt decode"))
e("R_TOKENS", r.get("TOKENS", c.get("tokens",8)))
e("R_REBUILD", r.get("REBUILD_CMD","python3 setup.py build_ext --inplace"))
e("SITE_FW", r.get("FRAMEWORK","vllm"))
PY
)"
# Under slurm_gpu.sh the allocation exports DOCKER_GPU_ARG (cgroup-scoped UUID); use it
# so the trial container binds the slurm-assigned GPU. Standalone falls back to config gpu.
GPU="${DOCKER_GPU_ARG:-\"device=$GPU_DEV\"}"
SITE="/opt/venv/lib/python3.12/site-packages/$SITE_FW"
OUT="$HERE/iter_opt_eval_${CASE}"; mkdir -p "$OUT" "$FICACHE" "$GOLDEN_DIR"

# ---- render the task (module path stays a placeholder the agent must derive) --------------
TASK_RUNTIME="$OUT/agent_task_rendered.md"
sed -e "s#\$MODEL_SNAP#$SNAP#g" \
    -e "s#\$METRIC#$R_METRIC#g" \
    -e "s#\$FRAMEWORK#$R_FRAMEWORK#g" \
    -e "s#\$CHECKOUT#$R_CHECKOUT#g" \
    -e "s#\$MODEL_NAME#$R_MODEL_NAME#g" \
    -e "s#\$WORKLOAD#$R_WORKLOAD#g" \
    -e "s#\$TOKENS#$R_TOKENS#g" \
    -e "s#\$REBUILD_CMD#$R_REBUILD#g" \
    "$TASK_TMPL" > "$TASK_RUNTIME"

run_trial () {
  local k="$1" cname="iteropt_${CASE}_run_${k}" log="$OUT/trial_${k}"
  echo "==================== $CASE TRIAL $k ===================="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run -d --name "$cname" --gpus "$GPU" \
    -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -e CODEX_HOME=/root/.codex-eval \
    -v "$CODEX_HOME_HOST":/root/.codex-eval \
    -v /raid/yilegu/models:/raid/yilegu/models:ro \
    -v "$FICACHE":/root/.cache/flashinfer \
    --workdir "$R_CHECKOUT" \
    "$IMG" sleep infinity >/dev/null
  docker cp "$EXTRACTOR" "$cname:/tmp/extract_and_profile.py"

  echo "== [$k] launching Codex agent (gpt-5.6-luna effort=$EFFORT, timeout ${AGENT_TIMEOUT}s)"
  set +e
  timeout "$AGENT_TIMEOUT" docker exec -i --workdir "$R_CHECKOUT" "$cname" \
    codex exec -m gpt-5.6-luna -c model_reasoning_effort="$EFFORT" \
      --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
      "$(cat "$TASK_RUNTIME")" > "${log}_agent.log" 2>&1
  local acode=$?
  set -e
  echo "== [$k] agent exit=$acode (transcript: ${log}_agent.log)"

  docker cp "$cname:/workspace/opt_run" "${log}_opt_run" >/dev/null 2>&1 \
    && echo "== [$k] saved iteration workspace -> ${log}_opt_run" \
    || echo "== [$k] (no /workspace/opt_run produced)"

  echo "== [$k] deterministic rebuild of the agent's tree"
  if ! docker exec --workdir "$R_CHECKOUT" "$cname" \
        bash -lc "$R_REBUILD" > "${log}_build.log" 2>&1; then
    echo "!! [$k] BUILD FAILED -> FAIL"; tail -15 "${log}_build.log"
    echo '{"trial":'"$k"',"verdict":"FAIL","reason":"agent left source non-compiling"}' > "${log}_verdict.json"
    docker rm -f "$cname" >/dev/null 2>&1 || true; return 0
  fi
  echo "== [$k] installing built tree into the wheel ($SITE)"
  docker exec "$cname" bash -lc "cp -a $R_CHECKOUT/$SITE_FW/. $SITE/"

  echo "== [$k] judging (general judge; private target answer-key)"
  set +e
  python3 "$HERE/judge_general.py" --agent-container "$cname" \
    --config "$CONFIG" --golden-dir "$GOLDEN_DIR" --out "${log}_verdict.json"
  set -e
  python3 - "$k" "$acode" "${log}_verdict.json" <<'PY'
import json,sys
k,ac,f=sys.argv[1:4]
d=json.load(open(f)); d["trial"]=int(k); d["agent_exit"]=int(ac)
json.dump(d,open(f,"w"),indent=2)
PY
  docker rm -f "$cname" >/dev/null 2>&1 || true
  echo "== [$k] done -> ${log}_verdict.json"
}

for k in $(seq "${START:-1}" "$N"); do run_trial "$k"; done

echo "==================== SUMMARY ($CASE) ===================="
python3 - "$OUT" <<'PY'
import json,glob,os,sys
out=sys.argv[1]; rows=[]
for f in sorted(glob.glob(os.path.join(out,"trial_*_verdict.json"))):
    rows.append(json.load(open(f)))
p=sum(1 for d in rows if d.get("verdict")=="PASS")
for d in rows:
    print(f"  trial {d.get('trial')}: {d.get('verdict'):4s} "
          f"recovered={d.get('recovered_frac')} lat={d.get('latency_us')}us "
          f"correct={d.get('correctness',{}).get('pass')} reason={d.get('reason')}")
print(f"  PASS {p}/{len(rows)}")
PY
