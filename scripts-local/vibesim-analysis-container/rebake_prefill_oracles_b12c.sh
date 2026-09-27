#!/usr/bin/env bash
# Re-bake vibesim-analysis:k3 with the B12c-corrected prefill predictions and restart the prefill oracles.
# $@ = ports to restart now (default 8805 8806 8807); 8808 is restarted separately once the running MLA agent is done.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PORTS=("${@:-8805 8806 8807}"); [ $# -gt 0 ] || PORTS=(8805 8806 8807)
declare -A RUN=([8805]=k3_kda_prefill [8806]=k3_mla_prefill [8807]=k3_kda_lcprefill [8808]=k3_mla_lcprefill)
echo "==== $(date +%F_%T) bake with $(git -C /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust log --oneline -1)"
"$HERE/build_context_k3.sh" > "$HERE/k3_b12c_rebake.log" 2>&1 \
  && sed 's#COPY context/#COPY context_k3/#' "$HERE/Dockerfile" | docker build -q -t vibesim-analysis:k3 -f - "$HERE" >> "$HERE/k3_b12c_rebake.log" 2>&1 \
  || { echo "!! bake failed (k3_b12c_rebake.log)"; exit 1; }
for port in "${PORTS[@]}"; do
  run="${RUN[$port]}"; docker rm -f "vibesim_oracle_${run}" >/dev/null 2>&1 || true
  docker run -d --name "vibesim_oracle_${run}" -e VIBESIM_PREBAKED_RUN="$run" -e VIBESIM_API_PORT="$port" -p "172.17.0.1:${port}:${port}" vibesim-analysis:k3 >/dev/null
done
sleep 25
for port in "${PORTS[@]}"; do printf "oracle %s: " "$port"; curl -s -m 10 "http://172.17.0.1:$port/api/v1/analyze?level=iteration" | python3 -c "
import json,sys
try: t=json.load(sys.stdin)['tree_text']; print([l.strip()[:90] for l in t.splitlines() if l.startswith('total') or 'attention_prefix' in l][:3])
except Exception as e: print('ERR', e)"; done
echo "==== $(date +%F_%T) done"
