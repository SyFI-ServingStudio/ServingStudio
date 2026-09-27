#!/usr/bin/env bash
# Run an arbitrary command on the slurm-allocated B200 (this agent container has no GPU): the
# command executes in a fresh sglang container with your edited tree at
# /sgl-workspace/sglang/python/sglang and /workspace/opt_run shared. cwd is kept if it is under
# /sgl-workspace or /workspace/opt_run, else /workspace/opt_run. Example:
#   /tmp/gpu_run.sh python3 -c 'import torch; print(torch.cuda.get_device_name())'
#   /tmp/gpu_run.sh nsys profile -o /workspace/opt_run/iter_03/trace python3 /tmp/kimi_single_layer_decode.py ...
exec python3 /tmp/k3_gpu_shim.py --exec -- "$@"
