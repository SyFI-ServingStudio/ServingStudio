#!/usr/bin/env bash
# Measure the pristine tree at the chunked-prefill case points on GPU 7 (shared, user-authorized) and
# dump per-kernel tables: evidence for VibeSim B12 (prefill kinds/worklets/presets). Not a judge run.
#   ./k3_prefill_profiles.sh            -> kimi_single_layer/profiles/prefill_{kda,mla}_B*_L*pf*.json
#                                          kimi_single_layer/result_prefill_{kda,mla}.json
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU="${K3_GPU_INDEX:-7}"
OUTD="$HERE/kimi_single_layer"; mkdir -p "$OUTD/profiles"
COMMON="--moe-backend flashinfer_mxfp4 --attn-heads 12 --experts 112 --ep 8 --local-topk 2 --hidden-scale 1.0 --cuda-graph --iters 20 --warmup 5"
run () { # $1 case, $2 attn args, $3 points
  docker run --rm --name "k3_prefill_prof_$1" --gpus "\"device=$GPU\"" -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -e SGLANG_OPT_FUSED_KDA_VERIFY=0 -e TOKENIZERS_PARALLELISM=false --shm-size 32g \
    -v /raid/hf/hub:/raid/hf/hub:ro -v /raid/yilegu/flashinfer_cache:/root/.cache/flashinfer \
    -v "$HERE/kimi_single_layer_decode.py:/tmp/kimi_single_layer_decode.py:ro" -v "$OUTD:/out" \
    --workdir /tmp lmsysorg/sglang:v0.5.20 \
    python3 /tmp/kimi_single_layer_decode.py --point "$3" $2 $COMMON \
      --profile-kernels "/out/profiles/prefill_$1.json" --json-out "/out/result_prefill_$1.json" 2>&1 \
    | grep -E "JSON|\[time\]|\[profile\]|FAILED|Traceback|Error"
}
echo "==== $(date +%F_%T) KDA prefill profiles"
run kda "--attn-type kda" "1,16384,pf;1,16384,pf49152;4,4096,pf"
echo "==== $(date +%F_%T) MLA prefill profiles"
run mla "--attn-type mla --attention-backend cutedsl_mla --kv-cache-dtype fp8_e4m3" "1,16384,pf49152;1,16384,pf;4,4096,pf"
echo "==== $(date +%F_%T) done"; ls -la "$OUTD"/profiles/prefill_* "$OUTD"/result_prefill_* 2>/dev/null
