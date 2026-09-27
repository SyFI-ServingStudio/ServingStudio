#!/usr/bin/env bash
# Follow-up #1 (user 2026-09-27): give the long-context prefill agents a VibeSim prediction of THEIR points.
#   1. JIT-fill + timing-predict the new presets predict_kimi_k3_b200_rank1_layer_{kda,mla}_lcprefill (16k/32k chunks
#      at 131072/229376/245760 prefixes) as ONE slurm job (real profiling -> slurm main);
#   2. commit presets + prediction logs on the worktree branch kimi-k3-arch, fast-forward main/ kimi-k3;
#   3. re-bake vibesim-analysis:k3 (build_context_k3.sh has the k3_{kda,mla}_lcprefill BAKE entries) and start
#      vibesim_oracle_k3_kda_lcprefill (8807) / vibesim_oracle_k3_mla_lcprefill (8808); existing oracles keep running;
#   4. point issue_k3_{kda,mla}_lcprefill_claude.json at 8807/8808 (render-only change: goldens keep their key);
#   5. launch MLA lcprefill rounds 4..6 (slurm mode). KDA rounds are follow-up #3 (longer agent budget).
# Usage: bake_lcprefill_oracles.sh [first_round_k=4] [n_rounds=3]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; K0="${1:-4}"; N="${2:-3}"
WT=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust
MAIN=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main
PRESETS=(presets/predict_kimi_k3_b200_rank1_layer_kda_lcprefill.json presets/predict_kimi_k3_b200_rank1_layer_mla_lcprefill.json)

echo "==== $(date +%F_%T) 1. JIT fill + predict (slurm main)"
job="$(sbatch --parsable --partition=main --job-name=k3_lcprefill_fill --output="$HERE/k3_lcprefill_fill.slurm.out" \
       "$HERE/slurm_gpu.sh" "$HERE/k3_slurm_step.sh" "$HERE/run_k3_jit_fill.sh" "${PRESETS[@]}" | cut -d';' -f1)"
echo "slurm job $job -> $HERE/k3_lcprefill_fill.slurm.out"
while squeue -h -j "$job" -o %T 2>/dev/null | grep -q .; do sleep 60; done; sleep 2
grep -a "^RESULT" "$HERE/k3_lcprefill_fill.slurm.out"
for c in kda mla; do
  m="$WT/logs/predict_kimi_k3_b200_rank1_layer_${c}_lcprefill/prediction.meta.json"
  [ -f "$m" ] || { echo "!! no prediction for $c (see k3_lcprefill_fill.slurm.out)"; exit 1; }
  echo "$c prediction: $(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('prediction_id'))" "$m")"
done

echo "==== $(date +%F_%T) 2. commit on kimi-k3-arch, ff main/ kimi-k3"
cd "$WT" && git add presets/predict_kimi_k3_b200_rank1_layer_*_lcprefill*.json \
  logs/predict_kimi_k3_b200_rank1_layer_kda_lcprefill logs/predict_kimi_k3_b200_rank1_layer_mla_lcprefill 2>/dev/null
git -C "$WT" commit -q -m "kimi_k3: long-context chunked-prefill presets + baked predictions (16k/32k chunks at 128k-262k prefixes) for the lcprefill oracles

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git -C "$WT" log --oneline -1
(cd "$MAIN" && git merge --ff-only kimi-k3-arch 2>&1 | tail -1)

echo "==== $(date +%F_%T) 3. bake image + start 8807/8808"
"$HERE/build_context_k3.sh" > "$HERE/k3_lcprefill_bake.log" 2>&1 \
  && sed 's#COPY context/#COPY context_k3/#' "$HERE/Dockerfile" | docker build -q -t vibesim-analysis:k3 -f - "$HERE" >> "$HERE/k3_lcprefill_bake.log" 2>&1 \
  || { echo "!! bake failed (k3_lcprefill_bake.log)"; exit 1; }
for spec in k3_kda_lcprefill:8807 k3_mla_lcprefill:8808; do
  run="${spec%%:*}"; port="${spec##*:}"
  docker rm -f "vibesim_oracle_${run}" >/dev/null 2>&1 || true
  docker run -d --name "vibesim_oracle_${run}" -e VIBESIM_PREBAKED_RUN="$run" -e VIBESIM_API_PORT="$port" \
    -p "172.17.0.1:${port}:${port}" vibesim-analysis:k3 >/dev/null
done
sleep 25
for port in 8807 8808; do printf "oracle %s: " "$port"; curl -s -m 10 "http://172.17.0.1:$port/api/v1/analyze?level=run_summary" | head -c 160; echo; done

echo "==== $(date +%F_%T) 4. point the lcprefill cases at the new oracles"
python3 - "$HERE" <<'PY'
import json, sys, pathlib
here = pathlib.Path(sys.argv[1])
for c, port in (("kda", 8807), ("mla", 8808)):
    p = here / f"issue_k3_{c}_lcprefill_claude.json"; d = json.loads(p.read_text())
    d["render"]["ORACLE_URL"] = f"http://172.17.0.1:{port}"
    d["_note"] += f" 2026-09-27: oracle re-baked for the long-context points (port {port}); rounds 1-3 used the <=48k-prefix prefill oracle."
    p.write_text(json.dumps(d, indent=2) + "\n"); print(p.name, "->", port)
PY
echo "==== $(date +%F_%T) 5. MLA lcprefill rounds $K0..$((K0+N-1)) (slurm mode)"
exec "$HERE/run_k3_continuous.sh" slurm mla_lcprefill_claude "$K0" "$N"
