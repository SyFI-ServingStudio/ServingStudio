# Task: eliminate the qk_norm inefficiency in real vLLM (case 28103)

You are optimizing a **real vLLM checkout** at `/workspace/vllm` on an NVIDIA **B200**
(`CUDA_VISIBLE_DEVICES=0`). The serving workload is **Qwen3-0.6B** (per-head q_norm/k_norm
RMSNorm). There is a known inefficiency in the attention normalization path. Your job is to
locate it, fix it in the real source, recompile, and prove the fix with the profiler.

## Diagnosis oracle (read-only, no GPU) — use this FIRST
A VibeSim roofline analysis API is reachable over HTTP at
`http://172.17.0.1:8799/api/v1/<verb>` (args as query params). Verbs:

- `workspace-info` — action space (configs, GPU, profile.db coverage). **Call first.**
- `simulate` — the baked before-state prediction (`prediction=.:before`).
- `analyze?level=operator|run_summary|iteration` — time breakdown.
- `optimality?prediction=.:before&scope=iter` — **the bottleneck finder**: an R0/R5/R6/R7
  ladder per node, ranked by headroom. A node with `necessary_share ≈ 0` is doing
  **unnecessary work** — that is your target.
- `kernels` — per-leaf drill-down + `has_cached_alternative`.

Reading rule: `R6 ≈ R5` → necessary work (batching/launch bound, kernel swaps won't help).
A large gap with `necessary_share ≈ 0` → removable work. Find that node, map it to the
vLLM source that produces it.

## Measurement tool — self-check every change
`/tmp/profile_qk_norm.py` reproduces the **exact strided tensor** Qwen3's q_norm/k_norm
receive (from the fused-qkv `.split()` view) and CUDA-times RMSNorm on it. Run it against
your rebuilt tree:

```bash
cd /tmp && PYTHONPATH=/workspace/vllm python3 /tmp/profile_qk_norm.py \
    --granularity module --tokens 8 --iters 3000 --check
```

- `T=8 module ... us/call` is your latency signal (before ≈ 13 µs).
- `CHECK {... "pass": true}` MUST stay true: the normalized output must still match the
  fp32 reference (abs-err < 0.05, no NaN). A "fix" that drops work but returns wrong or NaN
  values FAILS. Do not sacrifice correctness for speed.

## Recompile (incremental, ~2 min)
Edit under `/workspace/vllm`. The cmake cache + ccache are warm, so:
```bash
cd /workspace/vllm && python3 setup.py build_ext --inplace
```
rebuilds only what you touched and relinks `_C`. If you edit a CUDA kernel you must also
update any header it depends on, or the compile fails.

## Success
- Profiler `T=8 module` latency **substantially lower** than the ~13 µs baseline (the
  redundant work fully removed), AND
- `CHECK ... "pass": true` still holds.

Work autonomously: query the oracle, form a hypothesis, edit the real source, recompile,
re-profile, iterate until both conditions hold. Report the node you fixed, the files you
changed, and your before/after profiler numbers.
