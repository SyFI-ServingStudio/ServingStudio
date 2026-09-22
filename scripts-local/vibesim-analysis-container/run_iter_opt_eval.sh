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
# Per trial: fresh codex container on the slurm GPU -> Codex drives the loop against the
# VibeSim oracle -> DRIVER deterministically rebuilds+installs the agent's tree and runs the
# case's judge (judge_general.py: PR answer-key cases; judge_k3.py: open-ended Kimi-K3 case).
# The agent's iteration workspace + transcript are saved for reward-leak audit.
#
# Usage: run_iter_opt_eval.sh [CONFIG=issue_28103.json] [N_TRIALS=3] [EFFORT=max]
#   START=<k>          resume from trial k (preserves earlier trials' evidence)
#   AGENT_TIMEOUT=<s>  per-agent-turn budget (default 3600)
#   CODEX_MODEL=<m>    override the codex model (default: config codex_model or gpt-5.6-luna)
#   TASK_TMPL=<file>   override the prompt template (default: config task_template or
#                      agent_task_iter_opt.md)
#
# Config keys beyond the judge's (all optional unless noted):
#   codex_image (req), model_snap, gpu, case, judge (script in this dir), driver (file copied
#   to /tmp in the agent container; default extract_and_profile.py), mounts ["host:cont[:ro]"],
#   shm_size, edit_tree {container_path}: bind-mount a host copy of that in-image directory so
#   the judge can diff the agent's edits against the pristine copy; render.* keys are
#   substituted as $KEY in the template (INSTALL_CMD replaces the default wheel copy step).
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$HERE/issue_28103.json}"; [ -f "$CONFIG" ] || CONFIG="$HERE/$1"
N="${2:-3}"; EFFORT="${3:-max}"
CODEX_HOME_HOST=/raid/yilegu/codex_home_eval
FICACHE=/raid/yilegu/flashinfer_cache
GOLDEN_DIR=/raid/yilegu/eval_goldens
AGENT_TIMEOUT="${AGENT_TIMEOUT:-3600}"

# ---- load config fields (single python read -> shell vars) --------------------------------
eval "$(python3 - "$CONFIG" <<'PY'
import json,sys,shlex,os
c=json.load(open(sys.argv[1])); r=c.get("render",{})
def e(k,v): print(f'{k}={shlex.quote(str(v))}')
e("IMG", c["codex_image"]); e("SNAP", c.get("model_snap","")); e("GPU_DEV", c.get("gpu","3"))
e("CASE", c.get("case", os.path.splitext(os.path.basename(sys.argv[1]))[0]))
e("R_FRAMEWORK", r.get("FRAMEWORK","vllm"))
e("R_CHECKOUT", r.get("CHECKOUT","/workspace/vllm"))
e("R_REBUILD", r.get("REBUILD_CMD","python3 setup.py build_ext --inplace"))
e("R_INSTALL", r.get("INSTALL_CMD",""))
e("SITE_FW", r.get("FRAMEWORK","vllm"))
e("JUDGE", c.get("judge","judge_general.py"))
e("DRIVER", c.get("driver","extract_and_profile.py"))
e("CFG_TMPL", c.get("task_template","agent_task_iter_opt.md"))
e("CFG_MODEL", c.get("codex_model","gpt-5.6-luna"))
e("SHM", c.get("shm_size",""))
e("EDIT_TREE_PATH", (c.get("edit_tree") or {}).get("container_path",""))
print("MOUNTS=(" + " ".join(shlex.quote(m) for m in c.get("mounts",[])) + ")")
PY
)"
CODEX_MODEL="${CODEX_MODEL:-$CFG_MODEL}"
TASK_TMPL="${TASK_TMPL:-$HERE/$CFG_TMPL}"; [ -f "$TASK_TMPL" ] || TASK_TMPL="$HERE/$CFG_TMPL"
# Under slurm_gpu.sh the allocation exports DOCKER_GPU_ARG (cgroup-scoped UUID); use it
# so the trial container binds the slurm-assigned GPU. Standalone falls back to config gpu.
GPU="${DOCKER_GPU_ARG:-\"device=$GPU_DEV\"}"
SITE="/opt/venv/lib/python3.12/site-packages/$SITE_FW"
OUT="$HERE/iter_opt_eval_${CASE}"; mkdir -p "$OUT" "$FICACHE" "$GOLDEN_DIR"

# ---- render the task (python, not sed: REBUILD_CMDs contain '&&' which sed treats specially)
TASK_RUNTIME="$OUT/agent_task_rendered.md"
python3 - "$CONFIG" "$TASK_TMPL" "$TASK_RUNTIME" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); r=dict(c.get("render",{}))
r.setdefault("METRIC","per-step decode latency"); r.setdefault("FRAMEWORK","vllm")
r.setdefault("CHECKOUT","/workspace/vllm"); r.setdefault("MODEL_NAME","the model")
r.setdefault("WORKLOAD","short-prompt decode"); r.setdefault("TOKENS", c.get("tokens",8))
r.setdefault("REBUILD_CMD","python3 setup.py build_ext --inplace")
r["MODEL_SNAP"]=c.get("model_snap","")
t=open(sys.argv[2]).read()
for k in sorted(r, key=len, reverse=True):   # longest first: $MODEL_SNAP before $MODEL
    t=t.replace("$"+k, str(r[k]))
open(sys.argv[3],"w").write(t)
left=[l for l in t.splitlines() if "$" in l and any(tok.isupper() for tok in l.split("$")[1:2])]
print(f"rendered {sys.argv[3]} ({len(t.splitlines())} lines)" + (f"; unresolved: {left[:3]}" if left else ""))
PY

# ---- pristine copy of the editable in-image tree (once per case) ---------------------------
PRISTINE=""
if [ -n "$EDIT_TREE_PATH" ]; then
  PRISTINE="$OUT/pristine_tree"
  if [ ! -d "$PRISTINE" ]; then
    echo "== extracting pristine $EDIT_TREE_PATH from $IMG -> $PRISTINE"
    tmpc="pristine_${CASE}_$$"; docker rm -f "$tmpc" >/dev/null 2>&1 || true
    docker create --name "$tmpc" "$IMG" true >/dev/null
    docker cp "$tmpc:$EDIT_TREE_PATH" "$PRISTINE"
    docker rm -f "$tmpc" >/dev/null
    find "$PRISTINE" -name __pycache__ -prune -exec rm -rf {} +
  fi
fi

run_trial () {
  local k="$1" cname="iteropt_${CASE}_run_${k}" log="$OUT/trial_${k}"
  echo "==================== $CASE TRIAL $k ===================="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  local -a extra=()
  [ -n "$SHM" ] && extra+=(--shm-size "$SHM")
  for m in "${MOUNTS[@]:-}"; do [ -n "$m" ] && extra+=(-v "$m"); done
  local tree=""
  if [ -n "$PRISTINE" ]; then
    tree="${log}_tree"; rm -rf "$tree"; cp -a "$PRISTINE" "$tree"
    extra+=(-v "$tree:$EDIT_TREE_PATH")
  fi
  docker run -d --name "$cname" --gpus "$GPU" \
    -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -e SGLANG_OPT_FUSED_KDA_VERIFY=0 -e TOKENIZERS_PARALLELISM=false \
    -e CODEX_HOME=/root/.codex-eval \
    -v "$CODEX_HOME_HOST":/root/.codex-eval \
    -v /raid/yilegu/models:/raid/yilegu/models:ro \
    -v "$FICACHE":/root/.cache/flashinfer \
    "${extra[@]}" \
    --workdir "$R_CHECKOUT" \
    "$IMG" sleep infinity >/dev/null
  docker cp "$HERE/$DRIVER" "$cname:/tmp/$(basename "$DRIVER")"
  [ "$DRIVER" != "extract_and_profile.py" ] || true
  docker exec "$cname" mkdir -p /workspace/opt_run

  echo "== [$k] launching Codex agent ($CODEX_MODEL effort=$EFFORT, timeout ${AGENT_TIMEOUT}s)"
  set +e
  timeout "$AGENT_TIMEOUT" docker exec -i --workdir "$R_CHECKOUT" "$cname" \
    codex exec -m "$CODEX_MODEL" -c model_reasoning_effort="$EFFORT" \
      --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
      "$(cat "$TASK_RUNTIME")" > "${log}_agent.log" 2>&1
  local acode=$?
  set -e
  echo "== [$k] agent exit=$acode (transcript: ${log}_agent.log)"

  docker cp "$cname:/workspace/opt_run" "${log}_opt_run" >/dev/null 2>&1 \
    && echo "== [$k] saved iteration workspace -> ${log}_opt_run" \
    || echo "== [$k] (no /workspace/opt_run produced)"

  # Judge-integrity controls (test-only): PLANT_PATCH=<patch> applies a known edit to the
  # agent's tree after the agent step so a planted slowdown / numerics change can be shown to
  # FAIL, and the pristine tree (AGENT_TIMEOUT=1, no patch) to score ~0 improvement.
  if [ -n "${PLANT_PATCH:-}" ] && [ -n "$tree" ]; then
    echo "== [$k] CONTROL: applying planted patch $PLANT_PATCH to the agent tree"
    patch -d "$tree" -p1 < "$PLANT_PATCH"
  fi

  echo "== [$k] deterministic rebuild of the agent's tree"
  if ! docker exec --workdir "$R_CHECKOUT" "$cname" \
        bash -lc "$R_REBUILD" > "${log}_build.log" 2>&1; then
    echo "!! [$k] BUILD FAILED -> FAIL"; tail -15 "${log}_build.log"
    echo '{"trial":'"$k"',"verdict":"FAIL","reason":"agent left source non-compiling"}' > "${log}_verdict.json"
    docker rm -f "$cname" >/dev/null 2>&1 || true; return 0
  fi
  if [ -n "$R_INSTALL" ]; then
    echo "== [$k] install step: $R_INSTALL"
    docker exec "$cname" bash -lc "$R_INSTALL"
  else
    echo "== [$k] installing built tree into the wheel ($SITE)"
    docker exec "$cname" bash -lc "cp -a $R_CHECKOUT/$SITE_FW/. $SITE/"
  fi

  echo "== [$k] judging ($JUDGE)"
  set +e
  if [ -n "$tree" ]; then
    python3 "$HERE/$JUDGE" --agent-container "$cname" --config "$CONFIG" \
      --golden-dir "$GOLDEN_DIR" --out "${log}_verdict.json" \
      --tree-dir "$tree" --pristine-dir "$PRISTINE"
  else
    python3 "$HERE/$JUDGE" --agent-container "$cname" \
      --config "$CONFIG" --golden-dir "$GOLDEN_DIR" --out "${log}_verdict.json"
  fi
  set -e
  [ -f "${log}_verdict.json" ] || echo '{"verdict":"FAIL","reason":"judge produced no verdict"}' > "${log}_verdict.json"
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
          f"improvement/recovered={d.get('recovered_frac')} lat={d.get('latency_us')}us "
          f"correct={(d.get('correctness') or {}).get('pass')} reason={d.get('reason')}")
print(f"  PASS {p}/{len(rows)}")
PY
