#!/usr/bin/env bash
# Measure the before/after B200 baseline delta for case vllm-28103.
# Serves each image on GPU 3, runs `vllm bench serve` REPS times, saves JSON.
# The eval judge scores an agent's fix against the (after - before) delta this prints.
#
# Usage: measure_28103_baseline.sh [REPS=3]
set -Eeuo pipefail
REPS="${1:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SNAP=/raid/yilegu/models/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
OUT="$HERE/baseline_28103_b200"
FICACHE=/raid/yilegu/flashinfer_cache            # persist cubins across ephemeral containers
mkdir -p "$OUT" "$FICACHE"
GPU='"device=3"'

serve_and_bench () {
  local variant="$1" img="$2" cname="vllm28103_meas_$1"
  echo "== [$variant] launching serve ($img)"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run -d --name "$cname" --gpus "$GPU" --workdir /tmp \
    -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -v /raid/yilegu/models:/raid/yilegu/models:ro \
    -v "$FICACHE":/root/.cache/flashinfer \
    -v "$OUT":/results \
    -p 127.0.0.1:8003:8000 \
    "$img" \
    vllm serve "$SNAP" --served-model-name Qwen/Qwen3-0.6B \
      --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 2048 \
      --host 0.0.0.0 --port 8000 >/dev/null

  echo "== [$variant] waiting for /health (up to 300s)"
  local ok=0
  for i in $(seq 1 150); do
    if curl -s --max-time 3 http://127.0.0.1:8003/health -o /dev/null; then ok=1; break; fi
    sleep 2
  done
  if [ "$ok" != 1 ]; then
    echo "!! [$variant] server never became healthy; last logs:" >&2
    docker logs "$cname" 2>&1 | tail -20 >&2
    docker rm -f "$cname" >/dev/null 2>&1 || true
    return 1
  fi

  for r in $(seq 1 "$REPS"); do
    echo "== [$variant] bench rep $r/$REPS"
    docker exec --workdir /tmp "$cname" \
      vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
        --endpoint /v1/completions --model Qwen/Qwen3-0.6B --tokenizer "$SNAP" \
        --dataset-name random --random-input-len 1024 --random-output-len 1024 \
        --random-range-ratio 0 --num-prompts 32 --max-concurrency 8 \
        --request-rate inf --seed 0 --ignore-eos --disable-tqdm \
        --save-result --result-dir /results --result-filename "${variant}_rep${r}.json" \
      2>&1 | grep -iE 'throughput|TPOT|TTFT|Successful|Mean|Median' | sed "s/^/   [$variant r$r] /"
  done
  docker rm -f "$cname" >/dev/null 2>&1 || true
  echo "== [$variant] done, container removed"
}

serve_and_bench before rga-local/vllm-pr-28103:b200-before
serve_and_bench after  rga-local/vllm-pr-28103:b200-after
echo "== ALL DONE. results in $OUT"
