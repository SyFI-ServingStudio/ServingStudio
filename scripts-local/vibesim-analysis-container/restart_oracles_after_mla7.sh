#!/usr/bin/env bash
# Swap both K3 oracle containers to the freshly built vibesim-analysis:k3 image (post-B8
# predictions) once MLA trial 7 -- whose agent talks to 8802 live -- has finished.
set -uo pipefail
while pgrep -f "[q]ueue_mla7_after_fill.sh" >/dev/null; do sleep 120; done
echo "==== $(date +%F_%T) MLA trial 7 done -> restarting oracles on the post-B8 image"
for spec in k3_kda:8801 k3_mla:8802; do
  run="${spec%%:*}"; port="${spec##*:}"
  docker rm -f "vibesim_oracle_${run}" >/dev/null 2>&1 || true
  docker run -d --name "vibesim_oracle_${run}" -e VIBESIM_PREBAKED_RUN="$run" -e VIBESIM_API_PORT="$port" \
    -p "172.17.0.1:${port}:${port}" vibesim-analysis:k3 >/dev/null
done
sleep 20
for port in 8801 8802; do
  curl -s "http://172.17.0.1:$port/api/v1/analyze?level=run_summary" | head -c 300; echo
done
echo "==== $(date +%F_%T) oracles restarted"
