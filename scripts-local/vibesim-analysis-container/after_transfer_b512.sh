#!/usr/bin/env bash
# Continuation of run_k3_b512_when_free.sh after its parent was stopped (17:40, MLA-b512 oracle found
# to mispredict the attention leaf at B=256/512 -> Codex B12a fixes it; KDA-b512 prediction looked sane).
#   1. wait for the running mla_b512 transfer check (judge PID given) and print its verdict,
#   2. kda_b512: seed best_tree from the decode best tree + transfer check,
#   3. MLA prefill smoke with the fixed prefix-chunk pre-step hook (+ KDA regression) on GPU 7,
#   4. prefill profiles (evidence for VibeSim B12),
#   5. KDA-b512 continuous rounds only (MLA-b512 rounds start once 8804 is re-baked).
# Usage: after_transfer_b512.sh <judge_pid> <first_round_k> <n_rounds>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JPID="$1"; K0="$2"; N="$3"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}" K3_SHARED_GPU=1 DOCKER_GPU_ARG="\"device=7\""
while kill -0 "$JPID" 2>/dev/null; do sleep 30; done
echo "==== $(date +%F_%T) mla_b512 transfer check finished"
show () {
python3 - "$1" "$2" <<'PY'
import json, sys, shutil, os
d = json.load(open(sys.argv[1])); out = sys.argv[2]
print("transfer:", d.get("verdict"), (d.get("reason") or "")[:120])
ok = True
for k, p in (d.get("points") or {}).items():
    c = p["correctness"]; ok &= bool(c.get("pass"))
    print(f"  {k}: pristine {p['before_us']:.1f} -> seed {p['after_us']:.1f} us ({p['improvement']*100:+.1f}%) CHECK {c.get('pass')} rel {c.get('max_rel_err')}")
if not ok or not d.get("points"):
    print("!! seed tree fails CHECK (or no result) on the new points -> starting this case from PRISTINE")
    shutil.move(os.path.join(out, "best_tree"), os.path.join(out, "best_tree.transfer_failed"))
PY
}
show "$HERE/iter_opt_eval_k3_mla_b512/seed_transfer_verdict.json" "$HERE/iter_opt_eval_k3_mla_b512"

c=kda; SRC="$HERE/iter_opt_eval_k3_${c}"; OUT="$HERE/iter_opt_eval_k3_${c}_b512"; mkdir -p "$OUT"
[ -d "$OUT/pristine_tree" ] || cp -a "$SRC/pristine_tree" "$OUT/pristine_tree"
if [ ! -d "$OUT/best_tree" ]; then
  cp -a "$SRC/best_tree" "$OUT/best_tree"; echo "==== seeded ${c}_b512 best_tree from $SRC/best_tree"
  echo "==== $(date +%F_%T) transfer check: ${c}_b512 seed tree vs pristine on the new points"
  python3 "$HERE/judge_k3.py" --agent-container transfer_check --config "$HERE/issue_k3_${c}_b512.json" \
    --golden-dir /raid/yilegu/eval_goldens --out "$OUT/seed_transfer_verdict.json" \
    --tree-dir "$OUT/best_tree" --pristine-dir "$OUT/pristine_tree" --min-improvement 0 \
    > "$OUT/seed_transfer_judge.log" 2>&1
  show "$OUT/seed_transfer_verdict.json" "$OUT"
fi

echo "==== $(date +%F_%T) MLA prefill smoke (fixed prefix-chunk hook) + KDA regression"
COMMON="--moe-backend flashinfer_mxfp4 --attn-heads 12 --experts 112 --ep 8 --local-topk 2 --hidden-scale 1.0 --cuda-graph --iters 5 --warmup 2"
smoke () {
  docker run --rm --name "k3_prefill_smoke2_$1" --gpus '"device=7"' -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -e SGLANG_OPT_FUSED_KDA_VERIFY=0 -e TOKENIZERS_PARALLELISM=false --shm-size 32g \
    -v /raid/hf/hub:/raid/hf/hub:ro -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer \
    -v "$HERE/kimi_single_layer_decode.py:/tmp/kimi_single_layer_decode.py:ro" --workdir /tmp lmsysorg/sglang:v0.5.20 \
    bash -c "python3 /tmp/kimi_single_layer_decode.py --point '$3' $2 $COMMON --capture /tmp/g.pt 2>&1 | grep -E 'JSON|saved golden|FAILED|Error|Traceback|\[time\]|\[mla-prefill\]'; echo '==== replay'; python3 /tmp/kimi_single_layer_decode.py --point '$3' $2 $COMMON --replay /tmp/g.pt 2>&1 | grep -E 'CHECK|FAILED|Error|Traceback'"
}
smoke mla "--attn-type mla --attention-backend cutedsl_mla --kv-cache-dtype fp8_e4m3" "1,2048,pf2048;2,1024,pf1024;1,8192"
smoke kda "--attn-type kda" "1,2048,pf2048"
echo "==== $(date +%F_%T) prefill profiles (B12 evidence)"
"$HERE/k3_prefill_profiles.sh" > "$HERE/k3_prefill_profiles.log" 2>&1; tail -12 "$HERE/k3_prefill_profiles.log"
echo "==== $(date +%F_%T) KDA-b512 rounds $K0..$((K0+N-1)) on GPU 7"
CASES="kda_b512" exec "$HERE/run_k3_alternate.sh" 7 "$K0" "$N"
