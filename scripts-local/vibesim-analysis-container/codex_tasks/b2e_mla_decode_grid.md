# Task: `mla_decode_attention` sweep grid + cute-dsl constraints (branch `kimi-k3-arch`)

Worktree `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust`, branch `kimi-k3-arch` (do not cd to
`.../main`). `uv run --no-sync ...`, `cargo` at `~/.cargo/bin` (retry on build-dir lock; another agent edits
`model/work/**` and `simulator/src/arch/kimi_k3_sglang.rs` concurrently — do NOT touch those). Touch only
`simulator/src/timing/kernels/mla_decode_attention.rs`, `profiling/runners/attention/mla_decode_attention.py`,
`profiling/kernels/mla_decode_attention.py` and their tests. Commit with the trailer
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Failure (GPU JIT-fill of presets/predict_kimi_k3_b200_rank1_layer_mla.json, slurm 910)
`mla_decode_attention:sglang_cutedsl_mla profiled 134/153 specs; 19 failed`:
  * 9x `Expected block_num % (128 / block_size) == 0, got block_num=1 and block_size=64` — the sweep asks
    kv_len=64 (one 64-token page); the cute-dsl MLA decode kernel needs an even page count at page 64, i.e.
    kv_len a multiple of 128. sglang itself never launches it with a single page (check how
    `cutedsl_mla_backend.py` sizes/pads the page table for short sequences and mirror that: pad kv_len up to
    the next multiple of 128 and document it in the runner, or make the Rust sweep grid start at 128 — do
    BOTH so the cache key set the arch asks for is always measurable).
  * 1x CUDA OOM (72 GiB KV) at the grid corner (largest batch × 1,048,576 kv) — cap the sweep grid so the
    fp8/bf16 KV footprint stays under ~48 GiB (batch × kv_len × 576 × elem_bytes) and make the runner raise
    `ProfilerNotImplemented` (not a crash) for shapes beyond the pool it can allocate; the K3 workload points
    that MUST stay measurable are (B=1,kv=1,048,576), (B=16,kv=65,536), (B=128,kv=8,192).
Then the arch lookup `missing profile entry for mla_decode_attention:sglang_cutedsl_mla spec {batch_size:1,
kv_len:64,...}` must no longer occur: whatever grid the Rust `SweepCoords` enumerates must be exactly what the
runner can measure (read `simulator/src/timing/kernels/mla_decode_attention.rs` `sweep_grid`/`enumerate` and
the base kernel's interpolation contract in `simulator/src/timing/` so a kv_len below the first grid point is
still resolvable — e.g. clamp or include 128 as the floor).
Apply the same grid sanity to the `sglang_trtllm_mla` and `sglang_triton` backends (trtllm also uses page 64).

## Verify
`uv run --no-sync pytest tests/test_kimi_k3_profiling.py -q`; `cargo test -p simulator --lib mla_decode` green;
paste the new kv_len/batch grids. Report files changed + the grids.
