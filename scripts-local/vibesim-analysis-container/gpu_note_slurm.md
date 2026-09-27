**GPU access in this container (read before measuring).** This container has **no GPU**
(`nvidia-smi`, `torch.cuda` and `ncu` will not work here). Every `python3 /tmp/kimi_single_layer_decode.py …`
call is transparently forwarded to a **slurm-allocated B200**: the real driver (read-only source:
`/tmp/k3_driver_src/kimi_single_layer_decode.py`) runs in a fresh sglang container with **your edited
tree mounted at the same path** and `/workspace/opt_run` shared, and its output streams back with the
driver's exit code. Consequences:
- Each call costs slurm queue + container start (typically 1–3 min before the first driver line):
  **batch all points into one call**, do not poll with many small runs.
- Output paths: write under `/workspace/opt_run/…`. A `/tmp/<x>` path you pass is rewritten to
  `/workspace/opt_run/tmp/<x>` and symlinked back to `/tmp/<x>` here, so `--capture /tmp/golden.pt`
  still works. Environment variables you export here do NOT reach the GPU job (only source edits
  carry over — exactly as with the judge).
- For any other GPU command (micro-benchmarks, `nsys`, a quick `python3 -c` on CUDA):
  `/tmp/gpu_run.sh <command…>` runs it in the same kind of measurement container.
- Edited JIT-CUDA sources compile inside the GPU job; a compile error shows up in that output.
