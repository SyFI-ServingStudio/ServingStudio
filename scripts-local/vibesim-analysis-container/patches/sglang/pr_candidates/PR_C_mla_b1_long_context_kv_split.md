also 

# PR C — MLA decode at batch 1, long context: split the KV ×8 and merge by LSE

**Status: worth a PR, but small on its own.** It gives a real −6.3% decode time per token at 1M context end to end,
and it is bit-exact. Under 10% alone, it is the largest piece of the merged tree's 1M-decode gain (−10.0%).

|                                                                         |                                                                                                                                                                       |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| patch (vs stock sglang v0.5.20, `patch -p1` inside `python/sglang`) | `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang/pr_candidates/PR_C_mla_b1_long_context_kv_split.patch` |
| git                                                                     | `/raid/yilegu/k3_merge`, branch `prC_b1_kvsplit`, commit `7f1379c` (parent = branch `pristine`)                                                               |
| exported tree used in the benchmarks                                    | `/raid/yilegu/k3_e2e_trees/prC_b1_kvsplit` (instrumented copy: `prC_b1_kvsplit_debug`)                                                                            |
| size                                                                    | 4 files, +345 / −34 (175 lines are the new Triton module)                                                                                                            |
| origin                                                                  | hand-port of the loop's `mla_shapes_01_r1` (T4) plus the LSE-merge pieces of `mla_b512_02_r2`, onto stock                                                         |
| upstream status                                                         | not upstream. Related: open #42535 (opt-in split-KV for the *cute-dsl* MLA decode at batch 48)                                                                      |

## 1. The problem

At batch size 1 the TRT-LLM (trtllm-gen) MLA decode launcher takes its DeepSeek **min-latency** branch. That branch
halves `headDimPerCtaV`, so two CTAs stream every KV tile. The profile shows the kernel as
`fmhaSm100f…HVPerCta256…MultiCtasKv…`.

With Kimi-K3's 12 heads per rank at TP8, the result:

- **Every KV tile is loaded twice.** The torch-profiler trace of the real server (job 6951) shows the grid is
  `[74, 2, 1]` × 512 threads, at 1 CTA per SM:
  - 74 KV chunks × 2 V-halves = 148 CTAs, one per SM, so the GPU is already full.
  - The two CTAs of a KV chunk each read the same K/V tile and produce 256 of the 512 V columns.
- **The kernel is slow, and occupancy is not the reason.** Each of K3's 12 MLA layers spends **130.8 µs** in this
  kernel per decode step, about **1.57 ms of a 6.1 ms step**.
  - That is 604 MB of fp8 latent cache in 130.8 µs, about 4.6 TB/s.
  - A plain `torch.sum` over the same 604 MB takes 102 µs (5.9 TB/s), the practical HBM read floor.
  - So stock runs about 28% above the floor, with every SM busy.
- **There are two costs** (measured in §5.1):
  1. **The duplicated read.** Both V-halves load every K/V tile.
  2. **The cross-chunk reduction.** In `MultiCtasKv` mode, each KV chunk writes a partial output and LSE. The last
     CTA to finish for a (sequence, V-slice) then reduces all partials. With one sequence, that reduction runs in
     very few CTAs, and its cost grows with the number of KV chunks per sequence.

## 2. The change, by file

### `kernels/ops/attention/mla_decode_kv_split.py` (new): split tables + LSE merge

```python
# Chunks per request; measured on B200 at 1M context (k = 2: 118 us, 4: 108, 16: 104 vs 131 us unsplit).
KV_SPLIT_FACTOR = 8
# Below this the replicate/merge/LSE overhead outweighs the bandwidth gain.
KV_SPLIT_MIN_SEQ_LEN = 65536

def kv_split_factor(batch_size, max_seq_len):
    if batch_size == 1 and max_seq_len >= KV_SPLIT_MIN_SEQ_LEN:
        return KV_SPLIT_FACTOR
    return 0
```

**Idea.** Present the single request to the kernel as a **pseudo-batch of 8 rows**, each covering one page-aligned
eighth of the KV. With bs = 8 the launcher no longer takes the min-latency branch, so it uses the full-V kernel.

- **The grid** (job 6960 trace) becomes `[18, 1, 8]` = 144 CTAs: 18 KV chunks per row × 8 rows, at 1 CTA per SM.
  That is about the same CTA count as stock (148). Two things change:
  - **Each KV byte is read once.** Every CTA computes all 512 V columns of its tile.
  - **The reduction is spread out.** Each of the 8 rows reduces only 18 partials in-kernel, in parallel, and PR C's
    merge kernel combines the 8 rows.
- **Result:** the kernel drops to 100.5 µs per layer (1.206 ms / 12), which is at the 102 µs `torch.sum` read floor.

The split does not add CTAs. It escapes the min-latency branch, and it shortens the per-sequence reduction.

**Corrections (2026-10-15/16):**
- An earlier version of this doc said stock used "too few CTAs" and that the split "puts 8× more CTAs on the read".
  The profiler grids show both CTA counts are ≈ 148, so that was wrong.
- The next version said the duplicated read was the whole cost. The launcher A/B in §5.1 shows it isn't. Removing
  only the duplicated read (`[148, 1, 1]`) is *slower* than stock at 128k–512k and only 7% faster at 1M.

### `kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh` + `kernels/ops/attention/set_mla_kv_concat_q.py`

Stock's fused kernel already writes the KV cache and the fp8 query in one launch. It gains a `q_replicas` /
`q_replica_stride` parameter, so the same launch writes the 8 query replicas the pseudo-batch needs. That avoids an
extra copy kernel.

```cpp
for (uint32_t r = 0; r < params.q_replicas; ++r) {
  uint8_t* dst = o_row + r * params.q_replica_stride;
  reinterpret_cast<int4*>(dst)[lane_id] = nope;
  reinterpret_cast<uint16_t*>(dst + kFp8NopeDim)[lane_id] = rope;
}
```

### `srt/layers/attention/trtllm_mla_backend.py`: gate, metadata, dispatch

- **Backend-level gate, decided once:**

  ```python
  self._kv_split_supported = (
      self.backend == "trtllm-gen"
      and type(self)._run_decode_kernel is TRTLLMMLABackend._run_decode_kernel   # not cute-dsl / tokenspeed subclasses
      and self.data_type == torch.float8_e4m3fn
      and not get_parallel().dcp_enabled
      and self.max_context_len >= KV_SPLIT_MIN_SEQ_LEN
  )
  ```

  The workspace grows 150 → 256 MB, but only when this gate is true.
- **New metadata fields:** `kv_split_factor`, `kv_split_block_kv_indices` and `kv_split_seq_lens_k`.
- **Where the split tables are built:**

  - in eager `init_forward_metadata`;
  - in the decode CUDA-graph capture and replay hooks. The graph bakes in the split for bs = 1, decided from the
    capture bound `max_context_len`.
- **Decode dispatch:** run the kernel on the pseudo-batch with `return_lse=True`, then merge:

  ```python
  raw_out, lse = self._run_decode_kernel(query=..., block_tables=metadata.kv_split_block_kv_indices,
                                         seq_lens=metadata.kv_split_seq_lens_k, ..., return_lse=True)
  out = merge_kv_split(raw_out, lse, metadata.kv_split_seq_lens_k, bs)
  ```

## 3. What was dropped from the loop's version

- **The large-batch tail split** (the b512 r2 `tail_split_rows` machinery).
- **`is_var_seq=False` / `_fixed_len_decode`:** K3-harness constants.
- **The `K3_KVSPLIT` env knob:** the factor is now a constant, 8.
- **The `# EXP` gate-stream hunk in `kimi_k3.py`.**
- **The always-on workspace growth:** it is now gated as described above.

## 4. Correctness and per-layer speed (single-layer judge vs stock: seed 1, fresh goldens)

| case                            | backend         | point                                      | stock     | PR C      | Δ                            | max rel error             |
| ------------------------------- | --------------- | ------------------------------------------ | --------- | --------- | ----------------------------- | ------------------------- |
| MLA, serving backend (job 6941) | `trtllm_mla`  | 1 × 1M decode                             | 302.6 µs | 253.4 µs | **−16.2%**             | **0.0 (bit-exact)** |
|                                 |                 | 1 × 64k decode                            | 155.2 µs | 155.2 µs | 0.0%                          | 0.0                       |
|                                 |                 | 32 × 8k decode                            | 282.2 µs | 284.2 µs | +0.7%                         | 0.0089                    |
| MLA decode shapes (job 6858)    | `cutedsl_mla` | 1×1M / 16×64k / 128×8k / 32×8k / 1×8k |           |           | −0.7 / +0.4 / 0 / −0.7 / 0% | 0.0                       |

The `cutedsl_mla` row is a check that the gate is correctly **off**: the cute-dsl backend overrides
`_run_decode_kernel`.

## 5. Kernel-level evidence from the real server

How the profile was taken:

- **Model and server:** 48-layer K3, 8 × B200, TP8/EP8, `trtllm_mla`.
- **Capture:** torch profiler, 5 steady-state decode steps on rank 0 of a 1,040,000-token request.
- **Jobs:** 6951 stock, 6960 PR C.

| per decode step              | stock                                                        | PR C                                                | Δ               |
| ---------------------------- | ------------------------------------------------------------ | --------------------------------------------------- | ---------------- |
| MLA decode kernel, 12 layers | 1.570 ms (`…HVPerCta256…MultiCtasKv…`, 130.8 µs/layer) | 1.206 ms (full-V variant on the 8-row pseudo-batch) | −0.364 ms       |
| split merge + LSE kernels    | —                                                           | 0.047 ms                                            | +0.047 ms        |
| **decode step period** | **6.08–6.10 ms**                                      | **5.77–5.79 ms**                             | **−5.2%** |

A debug build (job 6905) confirmed the gate is on in the server and the split path is captured into the bs = 1 CUDA
graph: `supported=True backend=trtllm-gen dtype=fp8 max_context_len=1048576`, then `decode path taken: k=8`.

### 5.1 Why not fix the launcher instead? (flashinfer A/B, job 7535, 2026-10-16)

The V split comes from flashinfer's launcher (`fmhaKernels.cuh`, `computeCtaAndClusterConfig`). In min-latency mode
(`corrFactor = 1`), its check `totalNumCtas * corrFactor <= numSMs` fires even when the KV split alone has already
filled all 148 SMs.

**What was compared:** the batch-1 MLA decode kernel on its own, with the K3 TP8 shapes (12 heads, fp8, page 64).
- Timing: device time from the profiler, averaged over 12 rotating KV caches × 50 iterations.
- Arms:
  - stock, flashinfer's prebuilt (AOT) launcher;
  - the same launcher rebuilt from source (JIT), as a control;
  - the launcher with `<=` changed to `<` (`lt`);
  - the launcher with the V split disabled (`nosplit`);
  - PR C's 8-row pseudo-batch.

Script and data: `k3_e2e/fmha_launcher_ab/{bench_b1_mla.py, run_ab.sbatch, out_7535/}`.

| context | stock `[74,2,1]` µs | launcher `<` fix | V split off | PR C ×8 (kernel only) | PR C + merge (~4 µs, see note) |
|---|---|---|---|---|---|
| 8k | 11.1 `[16,4,1]` | 11.1 (same grid) | 11.8 `[16,1,1]` | (gate off) | |
| 32k | 19.7 `[37,4,1]` | 19.7 (same grid) | 20.7 `[64,1,1]` | (gate off) | |
| 64k | 25.6 | 25.5 (same grid) | 31.4 `[128,1,1]` | 17.3 `[18,1,8]` | ≈ 21 |
| 128k | 32.7 | **40.8** `[148,1,1]` | 40.8 | 23.7 | ≈ 28 |
| 256k | 48.0 | **54.9** | 54.9 | 37.0 | ≈ 41 |
| 512k | 75.5 | 77.6 | 77.7 | 59.9 | ≈ 64 |
| 1M | 131.4 | **122.4** (−6.9%) | 122.4 | 103.7 | ≈ 108 (−18%) |

Note on the last column: the benchmark's own merge is unfused torch, about 20 µs. The ≈ column instead adds the
4 µs per layer that PR C's fused Triton merge measured in the real server (§5).

**Accuracy:** the rebuilt control is bit-identical to the prebuilt launcher. Every full-V arm differs from stock by
the same ≤ 0.0067 max rel error, a reduction-order effect.

**Reading:**
- **The one-character launcher fix is not a substitute for PR C.** With the duplicated read removed and a 148-way
  KV split of one sequence, the kernel is slower than stock up to 512k. The extra cost tracks the number of KV
  chunks per sequence. That fits a reduction tail: the last CTA reduces 148 × 12 heads × 512 fp32 partials, ~3.6 MB.
- **PR C's ×8 pseudo-batch wins at every length ≥ 64k.** It removes the duplicated read *and* keeps the
  per-sequence reduction at 18 partials. At 64k its gain (≈ −18%) is larger than the 1M one, which suggests
  lowering the gate. PR C's own in-graph sweep chose 64k from end-to-end replay numbers, so re-check in the server
  before changing it.
- **An upstream flashinfer fix** would have to change both the V-split rule and how many KV chunks one sequence
  gets: cap `numCtasPerSeqKv`, or reduce hierarchically. That is a design change, not a one-line PR.
- **Caveat:** the reduction-tail explanation is inferred from timings. A ncu profile of the reduction phase has not
  been run.

## 6. End to end (48-layer K3, dummy weights, 8 × B200; clean protocol)

Setup: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/k3_e2e/RECIPE_K3_48L_8xB200.md`.

| workload                                                                       | metric                        | stock          | PR C           | Δ                                    | merged tree Δ |
| ------------------------------------------------------------------------------ | ----------------------------- | -------------- | -------------- | ------------------------------------- | -------------- |
| W1l: 1 × 1,040,000 → 512 (stable decode, 511 steps; jobs 7123 / 7125 / 7124) | TPOT                          | 6.152 ms       | 5.765 ms       | **−6.3%**                      | −10.0%        |
| W1: 1 × 1,040,000 → 32 (2 runs each)                                         | TPOT (31-step average, noisy) | 6.15 / 6.60 ms | 6.12 / 5.75 ms | −6.9%                                | −11.5%        |
|                                                                                | TTFT (1M prefill)             | 55.2 / 56.1 s  | 55.2 / 55.1 s  | −0.9% (untouched, as expected)       | −19.0%        |
| W2: 128 × 8k; W5: 512 × 8k; W6: 16 × 64k; W4: 16k prefill                   | throughput                    |                |                | within ±1.6% (not batch 1; gate off) |                |

- **Where it acts:** only on batch-1 decode with ≥ 64k context. Every other workload is unchanged within noise.
- **Share of the merged tree's gain:** about two thirds of the merged tree's 1M-context decode improvement.

## 7. Before submitting upstream

- **Short-context regression check.** Under CUDA graphs, a server with `context_length ≥ 64k` bakes the split into
  every bs = 1 graph replay, even for short contexts. Measure bs = 1 TPOT at 8k and 32k, and if it regresses, gate the
  replay on the actual sequence length.
- **Make the split factor a function of SMs, heads and sequence length,** instead of the constant 8.
- **Other head counts.** Check DeepSeek-V3 / R1 at TP8 (16 heads/rank) and TP16: does the min-latency branch fire
  there too, and does the split help?
- **Coordinate with #42535** (cute-dsl split-KV) and with the upstream LSE-base discussion (#40927). This merge uses
  base-2 LSE consistently, end to end.
- **Unit test:** split vs unsplit decode at 64k / 256k / 1M, bit-exact or within bf16 tolerance.

## 8. How the agent found it (case `mla_shapes_claude`, round 1, Claude Opus 5.5)

Evidence: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/iter_opt_eval_k3_mla_shapes_claude/trial_1_agent.log`
(transcript) and `trial_1_opt_run/{log.md, iter_00/analysis.md, iter_00/vs_optimality.json, iter_01/hypothesis.md, iter_02/analysis.md}`.

1. **Warm start.** The agent read the knowledge base (`k3_opt_history/TECHNIQUES.md`, `INDEX.md`) and the seed tree's
   diff. The seed already contained the cute-dsl → trtllm-gen redirect (T1) and the b512 tail-split / LSE-merge
   plumbing. The KB said nothing about batch-1 attention; the previous round had concluded "closed cubin, no knob left".
2. **Baseline profile at all 5 points.** At 1 × 1M the kernel table showed `fmhaSm100fKernel_…HVPerCta256…MultiCtasKv`
   taking 130 µs of a 241 µs step. The agent's note: *"604 MB of fp8 latent in 130 µs ≈ 4.6 TB/s, well below B200 HBM
   peak."* **It picked the target here, from the profiler, before querying VibeSim.** The `HVPerCta256` token in the
   kernel name was the clue.
3. **Read the FlashInfer/TRT-LLM launcher source** (`fmhaKernels.cuh`, `computeCtaAndClusterConfig`) and found the
   root cause:
   `isDsv3MinLatencyMode = mBatchSize == 1 && mHeadDimQk == 576 && mHeadDimV == 512`. That branch sets
   headDimPerCtaV = 256, so two CTAs re-read every KV tile.
4. **Microbenchmarks.**
   - It measured an HBM read floor (`torch.sum` over 604 MB, 102.2 µs).
   - It swept the split factor in isolation: 131.6 µs unsplit vs 112 / 108 / 106 / 104 µs at k = 2 / 4 / 8 / 16.
     Odd page counts fail the kernel's `block_num % (128 / page_size)` check.
   - Other shapes (16 × 64k, 32 × 8k, 128 × 8k) didn't benefit, so it gated the split on bs = 1.
5. **Integration.**
   - It reused the seed's LSE-merge plumbing.
   - It fixed a FlashInfer-autotune workspace overflow (202 MB needed vs 150 MB) by reading the autotuner source.
   - It added the ≥ 64k gate after 1 × 8k regressed (130.5 → 134.6 µs).
   - An in-graph sweep chose k = 8: 1 × 1M at 222.7 µs.
6. **iter_02.**
   - Rejected: a stride-0 expanded query, which gave wrong output.
   - Adopted: writing the 8 query replicas from the fused fp8 KV-concat kernel (216.5 µs).
7. **Correctness beyond the harness.**
   - The agent deliberately broke the merge, keeping only chunk 0. The harness CHECK still passed (rel 0.0) at 1M
     context.
   - So it wrote its own test (`scratch/test_split.py`, L ∈ {64k, 70k, 300k, 1M−37, 1M}): split vs unsplit ≤ 0.0068.
8. **Judge:** 241.1 → 216.6 µs at 1 × 1M (−10.15%), accepted.

**What drove the decision:**

- the kernel name in the profile;
- a hand-measured HBM floor;
- reading the closed-source launcher's open dispatcher;
- split-factor microbenchmarks;
- plumbing from an earlier round.

**VibeSim usage** (oracle at `172.17.0.1:8802`):

- **Calls:** `workspace-info`, `simulate`, `analyze` (run_summary / operator) and `optimality`. Several calls failed on
  the prediction-handle format, and `kernels` was unusable.
- **What the successful calls returned:**
  - The top node by headroom was `mxfp4_fused_moe` (r0/r5 = 44). That is a known dead end: the MoE is a closed cubin.
  - `mla_decode_attention` came second (r0/r5 1.58, 131 µs headroom). Its numbers were aggregated over the baked cases
    and internally inconsistent (r6 > r0).
  - Nothing pointed at the min-latency branch or its duplicated KV read.
- **The prediction is static.** It is a baked "before" state, so the agent's iter_01 / iter_02 "re-queries" returned
  byte-identical results.
- **The citations were written last.** `hypothesis.md` cites the VibeSim headroom, but the agent wrote
  `iter_00/analysis.md` at the very end, after searching its own prompt for the "measure, simulate, THEN analyze" rule.

**Verdict on VibeSim: marginal, and partly misleading.**

- It agreed that attention had headroom, which the profile already showed.
- It ranked an unfixable MoE kernel first.
- It had no model of the mechanism and could not evaluate the patched tree.

Without the oracle, this trajectory would very likely have come out the same. (`vibesim_usage_audit.py` also
undercounts these calls, because the agent used a `$B/<verb>` URL form.)
