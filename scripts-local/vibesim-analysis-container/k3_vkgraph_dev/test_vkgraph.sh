#!/usr/bin/env bash
# GPU smoke of the --verify-graph driver change (2026-09-28): pristine sglang v0.5.20 image, both layers.
# For each layer: (1) eager-vs-graph verify at the campaign's primary point + a small point, (2) capture a golden
# in graph mode and replay it (self-consistency: CHECK pass with rel err ~0), (3) eager-mode replay against the
# graph golden (does graph replay give the same numerics as the eager verify step?).
# Run as:  sbatch --partition=main --job-name=k3_vkgraph_test --output=<dev>/out/test.slurm.out \
#            slurm_gpu.sh k3_slurm_step.sh --docker <docker args below> bash /work/test_vkgraph.sh
set -uo pipefail
cd /work
D=/work/kimi_single_layer_decode.py
COMMON="--hidden-scale 1.0 --cuda-graph --iters 40 --warmup 10 --bf16-gemm-init --flashinfer-autotune"
KDA="--attn-type kda --moe-backend flashinfer_mxfp4 --attn-heads 12 --mamba-ssm-dtype bfloat16 --experts 112 --ep 8 --local-topk 2 $COMMON"
MLA="--attn-type mla --moe-backend flashinfer_mxfp4 --attention-backend cutedsl_mla --attn-heads 12 --kv-cache-dtype fp8_e4m3 --experts 112 --ep 8 --local-topk 2 $COMMON"
rc=0
for L in kda mla; do
  if [ $L = kda ]; then A="$KDA"; else A="$MLA"; fi
  P="64,8192,vk3;4,8192,vk3"
  echo "===== $L: graph-mode capture + replay ($P)"
  python3 $D --point "$P" $A --verify-graph --capture /work/out/${L}_vkgraph.pt --json-out /work/out/${L}_cap.json 2>&1 | grep -E "^JSON|^CHECK|saved golden|graph|Traceback|Error|error" | cut -c1-400 || rc=1
  python3 $D --point "$P" $A --verify-graph --replay /work/out/${L}_vkgraph.pt --json-out /work/out/${L}_rep.json 2>&1 | grep -E "^JSON|^CHECK|graph|Traceback|Error|error" | cut -c1-400 || rc=1
  echo "===== $L: EAGER replay against the graph golden (numerics graph vs eager)"
  python3 $D --point "$P" $A --replay /work/out/${L}_vkgraph.pt --json-out /work/out/${L}_rep_eager.json 2>&1 | grep -E "^JSON|^CHECK|Traceback|Error|error" | cut -c1-400 || rc=1
  echo "===== $L: 5 graph-mode reps at the primary point (sigma)"
  for i in 1 2 3 4 5; do
    python3 $D --point "64,8192,vk3" $A --verify-graph 2>&1 | grep -E "^JSON" | python3 -c "import sys,json; [print('rep', json.loads(l[5:])['latency_us'], json.loads(l[5:])['latency_mode'], json.loads(l[5:]).get('us_step')) for l in sys.stdin if l.startswith('JSON')]"
  done
done
echo "TEST_RC=$rc"
exit $rc
