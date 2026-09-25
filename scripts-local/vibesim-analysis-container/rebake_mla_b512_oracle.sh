#!/usr/bin/env bash
# Re-bake the oracle image with the B12a-fixed MLA-b512 prediction and restart ONLY the 8804 container
# (8801/8802/8803 keep serving; the running KDA-b512 trial talks to 8803).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# (no codex wait: "codex exec" also matches the optimizer-loop trial agents inside their containers;
#  the worktree B12a run had already exited when this was launched)
echo "==== $(date +%F_%T) rebake with $(git -C /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust log --oneline -1)"
"$HERE/build_context_k3.sh" > "$HERE/k3_mla_b512_rebake.log" 2>&1 \
  && sed 's#COPY context/#COPY context_k3/#' "$HERE/Dockerfile" | docker build -q -t vibesim-analysis:k3 -f - "$HERE" >> "$HERE/k3_mla_b512_rebake.log" 2>&1 \
  || { echo "!! bake failed (k3_mla_b512_rebake.log)"; exit 1; }
docker rm -f vibesim_oracle_k3_mla_b512 >/dev/null 2>&1 || true
docker run -d --name vibesim_oracle_k3_mla_b512 -e VIBESIM_PREBAKED_RUN=k3_mla_b512 -e VIBESIM_API_PORT=8804 \
  -p 172.17.0.1:8804:8804 vibesim-analysis:k3 >/dev/null
sleep 25; curl -s "http://172.17.0.1:8804/api/v1/analyze?level=iteration" | python3 -c "
import json,sys,re; t=json.load(sys.stdin)['tree_text']; print([l for l in t.splitlines() if l.startswith('total') or 'mla_decode_attention (' in l][:6])"
echo "==== $(date +%F_%T) 8804 restarted"
