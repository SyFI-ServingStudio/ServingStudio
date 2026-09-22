#!/usr/bin/env bash
# B5 alignment PROBE capture: run the Kimi-K3 single-layer driver under Nsight Systems inside
# the sglang v0.5.20 container with --nvtx-align, so VibeSim's alignment pipeline
# (alignment/nsys/parse.py -> label -> analyze) can compare measured per-kernel time of the
# real layer against the kimi_k3_sglang single-rank prediction. Produces, per case:
#   kimi_single_layer/align/<name>/{profile.nsys-rep, profile.sqlite, records/server.log}
# Usage: sbatch slurm_gpu.sh ./run_k3_nvtx_capture.sh [kda|mla|both]
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WHICH="${1:-both}"
IMG="${SGLANG_IMG:-lmsysorg/sglang:v0.5.20}"
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
OUT="$HERE/kimi_single_layer/align"; mkdir -p "$OUT"
KDA_ARGS="--attn-type kda --moe-backend flashinfer_mxfp4 --attn-heads 12 --mamba-ssm-dtype bfloat16 --experts 112 --ep 8 --cuda-graph --iters 20 --warmup 5"
MLA_ARGS="--attn-type mla --moe-backend flashinfer_mxfp4 --attention-backend cutedsl_mla --attn-heads 12 --kv-cache-dtype fp8_e4m3 --experts 112 --ep 8 --cuda-graph --iters 20 --warmup 5"

capture () {  # capture <name> <point> <driver args...>
  local name="$1" point="$2"; shift 2
  local d="$OUT/$name"; rm -rf "$d"; mkdir -p "$d/records"
  echo "==== capture $name point=$point ===="
  docker run --rm --name "k3nsys_$$_$name" --gpus "$GPU" --shm-size 32g \
    -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
    -e SGLANG_OPT_FUSED_KDA_VERIFY=0 \
    -v /raid/hf/hub:/raid/hf/hub:ro -v "$HERE":/work --workdir /tmp \
    --entrypoint bash "$IMG" -c '
      NSYS=$(ls /opt/nvidia/nsight-systems-cli/*/bin/nsys 2>/dev/null | head -1)
      [ -n "$NSYS" ] || NSYS=$(which nsys)
      echo "nsys=$NSYS"; "$NSYS" --version | head -1
      "$NSYS" profile --force-overwrite=true -t cuda,nvtx --cuda-graph-trace=node \
        --capture-range=none -o /work/kimi_single_layer/align/'"$name"'/profile \
        python3 /work/kimi_single_layer_decode.py --point '"$point"' '"$*"' \
          --nvtx-align /work/kimi_single_layer/align/'"$name"'/records --nvtx-align-steps 20 \
          --json-out /work/kimi_single_layer/align/'"$name"'/run.json 2>&1 | tail -25
      "$NSYS" export --type sqlite --force-overwrite=true \
        -o /work/kimi_single_layer/align/'"$name"'/profile.sqlite \
        /work/kimi_single_layer/align/'"$name"'/profile.nsys-rep 2>&1 | tail -2
      ls -la /work/kimi_single_layer/align/'"$name"'/
    '
}
case "$WHICH" in
  kda)  capture kda_b128_l8192 128,8192 $KDA_ARGS ;;
  mla)  capture mla_b128_l8192 128,8192 $MLA_ARGS ;;
  both) capture kda_b128_l8192 128,8192 $KDA_ARGS; capture mla_b128_l8192 128,8192 $MLA_ARGS ;;
esac
echo "==== done ===="
