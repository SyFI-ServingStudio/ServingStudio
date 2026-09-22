#!/usr/bin/env bash
set -Eeuo pipefail
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
docker run --rm --gpus "$GPU" -e CUDA_VISIBLE_DEVICES=0 --entrypoint python3 \
  rga-local/sglang-pr-20469:b200-after -c "
import torch; torch.zeros(1,device='cuda')
import sglang.srt.layers.attention.mamba.causal_conv1d as m
print('_HAS_SGL_KERNEL =', getattr(m,'_HAS_SGL_KERNEL','MISSING'))
import sgl_kernel
print('has causal_conv1d_fwd:', hasattr(sgl_kernel,'causal_conv1d_fwd'))
print('conv ops:', [o for o in dir(torch.ops.sgl_kernel) if 'conv' in o.lower()])
" 2>&1 | grep -E "_HAS_SGL_KERNEL|causal_conv1d_fwd|conv ops|Error|Traceback"
