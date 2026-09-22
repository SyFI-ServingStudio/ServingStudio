# Task: `mla_cache_append:sglang_cuda` must cover Kimi-K3's paged fp8 latent-KV append (branch `kimi-k3-arch`)

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust`, branch `kimi-k3-arch` (do not cd to
`.../main`). CPU only; `uv run --no-sync ...`; `cargo` at `~/.cargo/bin` (only if a Rust twin needs touching —
prefer not). Touch ONLY `profiling/runners/attention/mla_cache_append.py` (+ `mla_cache_append_reference.py` if
needed), `profiling/kernels/mla_cache_append.py`, and their tests; `git add` only those; commit with trailer
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Another process runs the launcher in this worktree
concurrently — keep edits atomic and do not run `cargo build`.

## Failure (GPU JIT-fill of presets/predict_kimi_k3_b200_rank1_layer_mla.json, slurm 905)
`mla_cache_append:sglang_cuda profiled 0/70 specs; 70 failed (torch mla_cache_append requires cache_format='plain',
got 'page_planar_fp8')`. The K3 MLA worklet (`simulator/src/worklet/kimi_k3_mla_local.rs:167`) requests
`cache_format="page_planar_fp8"`, `kv_dtype=fp8_e4m3`, `block_size=64`, `kv_lora_rank=512`, `rope_dim=64`,
input bf16 — the cookbook B200 recipe (`--kv-cache-dtype fp8_e4m3`, cutedsl_mla with page 64).

## What sglang v0.5.20 actually launches for that path (measured with the real layer on B200)
One kernel per decode step: `sglang::set_mla_kv_concat_q_fp8_kernel<8, true, long>` (17.5 us at B=128) — the
fp8 branch of `MLATokenToKVPool.set_mla_kv_buffer` / the cute-dsl backend's KV write in
`sglang/srt/layers/attention/cutedsl_mla_backend.py` (`forward_decode`, look for `set_mla_kv_buffer` /
`set_mla_kv_concat_q_fp8`). Read the exact call inside the image:
`docker run --rm --entrypoint bash lmsysorg/sglang:v0.5.20 -c 'grep -rn "set_mla_kv_concat_q_fp8\|def set_mla_kv_buffer" /sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/cutedsl_mla_backend.py /sgl-workspace/sglang/python/sglang/kernels/ops -r | head -20'`
and the wrapper's signature (`sglang/kernels/ops/...` or `sgl_kernel`). The pool buffer for page_size 64 is
`(num_pages*64 + 64, 1, 576)` fp8 (K3: kv_lora_rank 512 + rope 64), written at `out_cache_loc` per token with
the bf16 latent+rope input quantized to fp8 (scale 1.0 unless the call passes one).

## Change
Add `cache_format="page_planar_fp8"` support to the `sglang_cuda` backend runner: build the fp8 paged pool
operands (block_size 64 pages, scattered slots like the plain path), call the real sglang kernel exactly as the
backend does, time it with the existing Timer/Energy helpers, and report bytes = tokens*(576 input bf16 read +
576 fp8 written). Keep `plain` working. Extend the kind's validation/`BackendSupport` so `sglang_cuda` advertises
both formats on B200 and `torch`/`vllm_cuda` keep their current contract. Update tests
(`tests/test_mla_cache_append*.py` or wherever this kind is tested). Report: the exact kernel call you wrap
(module path + signature) and the test result.
