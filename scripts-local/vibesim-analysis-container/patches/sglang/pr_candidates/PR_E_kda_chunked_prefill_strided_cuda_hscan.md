# PR E — KDA chunked prefill: strided q/k/v views, no host sync, CUDA h-scan

**Status: candidate. It clears the 3% end-to-end bar on every clean-protocol workload (+3.4…+7.8%, 2 runs).** One open
question decides whether to send it upstream: on sglang `main`, K3's prefill now defaults to the FlashInfer KDA
kernel on SM100 (#43246, 2026-10-08), not the Triton path this PR speeds up. See §7.

| | |
|---|---|
| patch (vs stock sglang v0.5.20, `patch -p1` inside `python/sglang`) | `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang/pr_candidates/PR_E_kda_chunked_prefill_strided_cuda_hscan.patch` |
| git | `/raid/yilegu/k3_merge`, branch `prE2_no_conv`, commit `8a38c86` (parent = branch `pristine`) |
| exported tree used in the benchmarks | `/raid/yilegu/k3_e2e_trees/prE2_no_conv` |
| size | 9 files, +1011 / −24 (630 lines are the new CUDA kernel `kda_chunk_h.cuh`) |
| origin | the KDA chunked-prefill lineage (`kda_prefill_claude` r1, `kda_lcprefill_claude` r1/r2/r4). Dropped: the fused conv1d + SiLU + L2-norm kernel (follow-up, §6), the approximate SiTU activation (must not ship) and the agent's `K3_PTX… # TMP_AB` A/B hooks |
| upstream status (main `0d0be8fc98`, 2026-10-11) | no CUDA h-scan and no strided l2norm on main. Main reads `logical_num_tokens` from forward metadata when set (a partial fix of the host sync). K3 prefill on SM100 defaults to FlashInfer since #43246 |

## 1. The problem

**What KDA chunked prefill runs.** It is the Triton `chunk_kda` pipeline:
- conv1d over the packed q/k/v projection, split into q/k/v;
- L2 norm of q and k, then the intra-chunk solve;
- the inter-chunk state scan (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, the "h-scan");
- the output.

**Where the time goes in stock** (16k chunks, 12 KDA heads per rank at TP8):
1. **Copies.** q/k/v (and the output gate) are strided views into the packed conv output. Stock makes them
   contiguous with `.contiguous()` before `l2norm_fwd`, `recompute_w_u` and `fused_norm_gate`, which costs extra
   full passes over the activations.
2. **A device→host sync.** `int(query_start_loc[-1])` in `KDAAttnBackend` drains the launch queue on every prefill
   forward.
3. **A latency-bound scan.** With few sequences, the h-scan launches `cdiv(V, BV) · N · H` CTAs (for one 16k
   request, cdiv(128, 32) × 1 × 12 = 48 CTAs on 148 SMs). Each CTA walks all 256 chunks serially. Every chunk does four tiny dots
   through the tcgen05/TMEM path, and the issue → commit → wait round trip dominates.

## 2. The change, by file

### `kernels/ops/attention/fla/kda.py`, `fla/l2norm.py`, `fla/fused_norm_gate.py`: read the strided views in place

- **`l2norm_fwd_strided_heads` (new, l2norm.py).** It is the same tile and reduction as `l2norm_fwd`, but it
  addresses rows of a `[1, T, H, D]` view with any token stride. So q/k are normalized straight out of the packed
  conv output:

  ```python
  def _l2norm_heads(x):
      if _is_packed_heads_view(x) and x.shape[-1] <= 512 and not x.is_contiguous():
          return l2norm_fwd_strided_heads(x)      # no .contiguous() copy
      return l2norm_fwd(x.contiguous())
  ```

- **`recompute_w_u` (kda.py)** takes v's token stride (`stride_v`) instead of a contiguous copy. It reuses v's
  storage for the output when v is not a view into the packed qkv.
- **`fused_norm_gate`** accepts a head-packed gate with a token stride (`G_HEADS`, `stride_g`) for D ≤ 512.
- **`chunk_intra.py`** prunes the inter-solve autotune to BK = 32 on one warp. The autotune key omits T, so the
  first call's length used to pick the config for every length. One warp is fastest, or within 1 µs, everywhere
  from T = 1.5k to 64k.

### `srt/layers/attention/linear/kda_backend.py`: no host sync

```python
extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
if extend_seq_lens_cpu is not None and (
    forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
    and not forward_batch.forward_mode.is_split_prefill()
):
    logical_num_tokens = int(sum(extend_seq_lens_cpu))   # host copy, no D2H sync
else:
    logical_num_tokens = int(query_start_loc[-1])
```

### `kernels/jit/csrc/attention/kda_chunk_h.cuh` + `kernels/ops/attention/kda_chunk_h.py` (new): CUDA h-scan

**What it replaces.** It is a drop-in for the Triton h-scan on the per-channel-gate path used by `chunk_kda`:
gk in log2 space, K = 128, no scalar g, no state tracking. `covered()` checks those preconditions, and everything
else falls back to Triton.

**How it works:**
- **One CTA per (sequence, head, 16 V columns).** The state lives in `mma.sync` accumulator registers.
- **Warp-specialized:** 4 compute warps, plus 4 load warps that stream each chunk's h-independent operands
  (w, k, u, gk) through a 4-stage `cp.async` ring.
- **Bit-identical by construction.** Every dot keeps Triton's reduction order: one fp32 chain over K in k16 steps
  for `w @ hᵀ`, and a fresh chain over the 64 tokens for `kᵀ @ v`, added after the decay.

**Dispatch in `chunk_delta_h.py`.** It uses the CUDA scan only while its grid fits in one wave, since that is where
it was measured to win:

```python
if (k.is_cuda
    and triton.cdiv(V, 16) * N * H <= get_multiprocessor_count(k.device.index)
    and kda_chunk_h_covered(...)):
    return kda_chunk_h(k=k, w=w, u=u, gk=gk, ...)
```

**The Triton fallback is retuned too.** When the grid at BV = 32 is ≤ 2 waves, the Triton kernel itself switches to
`BV = 16`, `num_stages = 3`, which doubles the CTAs and prefetches the next chunk's operands. Unless an
`SGLANG_GDN_CHUNK_H_*` env knob pins the tile. This path is **shared with GDN models** (e.g. Qwen3-Next), so it
also changes their few-sequence prefill. It is bit-identical (V columns recur independently), but it needs a GDN
benchmark in the PR.

It also derives the chunk offsets on the device, so the `prepare_chunk_offsets` host round trip goes away. The
Triton kernel gains optional `final_state` / `final_state_indices`, so a segmented scan can carry the fp32 state
between segments.

### `kernels/ops/mamba/causal_conv1d_triton.py`

`BLOCK_M = 8 → 16`, `num_warps = 2` for the prefill conv1d (T27).

## 3. What was dropped from the loop's version

| dropped | why |
|---|---|
| fused conv1d + SiLU + q/k L2 norm (`kda_conv_l2norm.{cuh,py}`, ~410 lines) | worth ≤ 0.5 pt on prefill-only workloads (ablation, §4), but 4–6 pt per layer on mixed batches. Better as a follow-up PR with its own mixed-chunk numbers |
| approximate SiTU (`tanh.approx` in `situ_and_mul.cuh`) | changes numerics for every caller (`K3_OPTIMIZATION_REPORT.md` §5) |
| attn-res TMA kernel with the residual add fused | ≈ 0 end to end (tree `cand_E2_attnres_add`, job 7577: −0.5…+1.4%) |
| `K3_PTX`, `K3_PTX_SUFFIX`, `K3_PTX_ARCH` (`# TMP_AB`) | agent A/B switches, off by default |
| the `kimi_k3.py` beta = 1 prefix fold | needs the decode lineage's shared-expert overlap; single-rank only |

## 4. Correctness and per-layer speed (single-layer judge vs stock: seed 1, fresh goldens)

Jobs 7603 (full lineage, `prE_kda_prefill`), 7604 (ablations) and 7619 (this patch, `prE2_no_conv`). The max abs
error is **0.0 at every point**: the CUDA scan and the strided kernels are bit-identical to stock by construction.

| case | point | this patch | + fused conv (`prE_kda_prefill`) | − CUDA h-scan (ablation) |
|---|---|---|---|---|
| prefill | 1 × 16k | −5.1% | −6.4% | −4.7% |
| | 1 × 16k @ 48k prefix | −8.6% | −9.5% | −7.5% |
| | 4 × 4k | −5.1% | −6.3% | −6.7% |
| long-context prefill | 16k @ 245k | −4.8% | −5.3% | −3.6% |
| | 32k @ 229k | −4.6% | −5.1% | −4.2% |
| | 16k @ 131k | −6.7% | −7.4% | −5.5% |
| | 32k @ 131k | −6.0% | −6.3% | −4.8% |
| mixed (decode rows + prefill chunk) | 64 dec + 16k | −7.2% | −11.1% | |
| | 128 dec + 4k | −9.8% | −16.2% | |
| | 64 dec + 16k @ 48k | −5.7% | −9.5% | |
| decode shapes | 1×1M / 16×64k / 128×8k / 32×8k / 1×8k | (prefill-only change) | +3.1 / −1.0 / −0.8 / −0.8 / 0.0% | |

**Reading the decode-shapes row.** That row is a no-op check, and it is clean. The +3.1% at 1 × 1M also appears
when stock is judged against itself (job 7618, `pristine_ctrl`: +3.1 / 0.0 / −0.2 / 0.0 / 0.0%). So it is the
judge's stored baseline at that point, not a regression.

**Where the gain comes from:**
- **Strided views + no host sync:** ~4–5 pt per layer, the bulk.
- **CUDA h-scan:** ~1–2 pt per layer.
- **Fused conv:** ~1 pt on prefill and 4–6 pt on mixed batches. It is dropped here (§3).

## 5. End to end (48-layer K3, dummy weights, 8 × B200, TP8/EP8; clean protocol)

Setup: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/k3_e2e/RECIPE_K3_48L_8xB200.md`.
Workloads `smoke w1 w2 w4 w5 w6`, run in the same order with `--flush-cache`. Stock is 3 runs (jobs 6906, 6911,
7514).

| workload | metric | stock (mean of 3) | this patch (jobs 7620, 7633) | Δ (mean of 2) | full lineage `cand_E` (2 runs) | ALL |
|---|---|---|---|---|---|---|
| W6 16 × 64k → 64 | output tok/s | 41.63 | 44.97 / 44.77 | **+7.8%** | +8.3% | +12.3% |
| | TTFT | 13.18 s | 12.17 / 12.22 s | **−7.5%** | −7.9% | −12.6% |
| W4 16 × 16k prefill | input tok/s | 46,742 | 49,972 / 49,253 | **+6.1%** | +7.8% | +9.1% |
| W1 1 × 1,040,000 → 32 | TTFT (1M prefill) | 55.47 s | 52.78 / 52.94 s | **−4.7%** | −5.1% | −19.0% |
| W5 512 × 8k → 128 | output tok/s | 701.6 | 733.8 / 729.9 | **+4.3%** | +4.8% | +7.2% |
| W2 128 × 8k → 256 | output tok/s | 1,326 | 1,381 / 1,361 | **+3.4%** | +4.6% | +7.0% |

The two runs agree within 1.5% on every workload.

**Why decode-throughput workloads move.** The prefill change also speeds up W2 and W5 because those runs start with
128 or 512 prefills of 8k tokens, and TPOT includes the decode steps stalled behind them. That is about +4–5% of
ALL's +7%. So, of the merged tree's 8k-workload gain, most comes from this patch, not from the decode levers.

## 6. Follow-ups

- **PR E2: the fused conv1d + SiLU + L2 norm** (`kda_conv_l2norm`, in `prE_kda_prefill`). It needs its own
  `--enable-mixed-chunk` end-to-end numbers.
- **The mixed-batch guard (T38).** It lifts the one-wave guard, so the CUDA scan also serves N = 65/129-sequence
  mixed batches (kda_mixed r2, −3.4%). It is not in this patch.

## 7. Before submitting upstream

- **The FlashInfer question.** On main (since #43246), safe-gate KDA models such as K3 default to
  `--linear-attn-prefill-backend flashinfer` on SM100/SM103. This patch only speeds up the Triton path.
  - That FlashInfer kernel is not in v0.5.20: `--linear-attn-prefill-backend flashinfer` fails at startup (job 7626).
  - The v0.5.20 alternatives on the same workloads:
    - `cutedsl` (7635): ≈ stock Triton, ±0.9% everywhere.
    - `nvidia_kda` (7636): the 1M prefill is +70% slower.
    - `flashkda` (7627): falls back to Triton above 2048 tokens, so it is identical to stock here.
  - So this patch is the fastest KDA prefill in v0.5.20, by 3–8 points. Comparing with main's FlashInfer kernel needs
    a run on main.
  - If FlashInfer is faster, this PR matters only where Triton is still used: other GPUs, TBO, full prefill CUDA
    graphs, deterministic mode and the original Kimi Linear.
  - The PR text must make that case.
- **Rebase onto main.** `kda_backend.py` moved: main's `forward_metadata.logical_num_tokens` already covers part of
  the host-sync fix.
- **Kernel tests:** CUDA vs Triton h-scan bit-exactness over chunk counts, N, H, initial/final-state slots, and the
  fall-back conditions; strided vs contiguous l2norm / norm-gate.
- **Hardware:** the scan's `_arch_ok` admits SM80+. It was measured on B200 only, so check H100 before claiming it
  there.

## 8. How the agent found it (Claude Opus 5.5; 3 accepted rounds plus 2 failed precursors)

Trial number = round number (`rounds.jsonl`). `export_k3_patches.sh` numbers the lcprefill patches by accepted trial,
so `01_r1` = trial 1, `02_r2` = trial 2 and `03_r4` = trial 4; trial 3 FAILED.
`<base>` = `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container`.

**Where each PR E piece came from** (checked against the patches):

| PR E piece | Introduced in |
|---|---|
| Strided l2norm, gated-norm gate and `recompute_w_u` v (copies removed); host-sync removal; Triton h-kernel BV16/nw4/ns3 tile for small grids | `kda_prefill_claude` r1 (T25) |
| CUDA `mma.sync` h-scan `kda_chunk_h` plus the `final_state` carry plumbing; conv1d BLOCK_M=16 / 2 warps | `kda_lcprefill_claude` r1 (T26/T27) |
| Warp-specialized 4-stage h-scan; `_prune_inter_solve_configs` | `kda_lcprefill_claude` r4 (T34) |

Dropped from PR E:
- the attn-res fused add (kda_prefill r1);
- SiTU (lcprefill r2, T28);
- `kda_conv_l2norm` (lcprefill r4);
- inert `TMP_AB` kda_ptx toggles (lcprefill r1).

Both h-scan kernels were first written in rounds that failed the judge, and reached accepted rounds through the
warm-start history: `kda_prefill_claude` trial 2 wrote the first kernel, and `kda_lcprefill_claude` trial 3 added
warp specialization.

### 8.1 `kda_prefill_claude` round 1 (T25), 2026-09-25
Evidence: `<base>/iter_opt_eval_k3_kda_prefill_claude/trial_1_agent.log`, `trial_1_opt_run/{log.md, iter_00..04/*.md, micro/}`.

1. **Measure.**
   - Kernel table [19–20], then VibeSim [23–35], then its own trace/timeline tool [51–55].
   - The timeline showed `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` at 629.7 µs (48 CTAs), three
     `aten::copy_ [1,16384,12,128]` at ~63 µs each, a 59 µs gate copy, and a `Memcpy DtoH` under `aten::_local_scalar_dense`.
2. **Tile (iter_01).** It read `chunk_delta_h.py` [58–63] and swept tiles in `micro/bench_h.py` [76–85]:
   *"BV=16 / 4 warps / 3 stages is **bit-exact** and cuts the kernel from 609 to 395 µs (N=1). Changing warps alters
   numerics"* [79]. N=16 regressed (104 → 139 µs), so the tile is gated on grid size.
3. **Host sync (iter_02).** Process noise (±2–3%) masked the tile change [104–114]. A gap tool [117–122] then found
   `cudaStreamSynchronize` 701 µs inside `aten::item` and a 143 µs GPU gap. Source reading [125–141] showed
   `query_start_loc[-1] == sum(extend_seq_lens_cpu)`; the fix is at [145].
4. **Copies (iter_04).**
   - Read the l2norm, gated-norm, `chunk_intra` and `kda.py` sources [148–177].
   - Strided l2norm: bit-exact, 78 → 24 µs per tensor [287–290].
   - Strided gate load: 89 → 35 µs [314–317].
   - `stride_v` for `recompute_w_u`: the first attempt broke the autotuned caller, fixed at [340–347].
   - *"Every copy kernel is gone (41 → 34 launches)"* [357]; 8584 / 8693 / 8594 µs [353].
5. **Not kept:**
   - the iter_03 attn-res fused add, ported from `mla_b512_claude_3` in the KB [181–188];
   - a cuBLAS vs DeepGEMM swap for the front GEMM: equal [396];
   - conv1d BLOCK_M=16 / nw=2, 106 → 96 µs and exact [430], reverted [444]. It came back in lcprefill r1.
6. **Judge.** The first judge returned no verdict; the rejudge passed. Results: 9071.9 → 8564.0 µs (−5.6%),
   9169 → 8591 µs (−6.3%), 8932 → 8453 µs (−5.4%). Bit-exact.

### 8.2 Precursor: `kda_prefill_claude` trial 2 (first CUDA h-scan, FAIL)
Evidence: `.../iter_opt_eval_k3_kda_prefill_claude/trial_2_agent.log`, `trial_2_opt_run/iter_0{0,1}/*.md`, `patches/sglang/claude/UNACCEPTED_kda_prefill_r2_cuda_kda_chunk_h_hscan.patch`.

1. **GEMM detour.** VibeSim `kernels` said `has_cached_alternative: true` for the GEMMs [50–51]. Benchmarks of
   torch / DeepGEMM / cuDNN / CUTLASS / cuBLASLt found nothing faster [76–88].
2. **Root cause from the Triton cache.** The cached h-kernel [100–101] contains 4× `ttng.tc_gen5_mma`, TMEM
   alloc/load, and `tcgen05.commit/wait`: every tiny per-chunk dot makes a tcgen05 round trip.
3. **Failed Triton route.** It transposed the dots so Triton would fall back to `mma.sync` [114, 121–122]. Bit-exact,
   but slower: 406 µs vs 493–933 µs [119].
4. **CUDA JIT kernel** (`add-jit-kernel` skill [131]):
   - state kept in `mma.sync` registers, operands fed by a cp.async ring;
   - ncu showed it issue-bound [194], but trimming address math gave nothing (228 µs) [205];
   - standalone N=1: 398 → 223 µs; N=4 regressed (136 → 155), so it is gated to one wave;
   - in-layer: 516 → 281 µs [270].
5. **Judge: FAIL.** −1.07% at the primary point, below the 3σ floor of 1.33%.

### 8.3 `kda_lcprefill_claude` round 1 (T26 + T27), 2026-09-27
Evidence: `<base>/iter_opt_eval_k3_kda_lcprefill_claude/trial_1_agent.log`, `trial_1_opt_run/{log.md, iter_00, iter_01, micro/}`.

1. **Target chosen from the history, before any profile.** The agent read `kda_prefill_claude_2` [9] and called
   *"trial 2's h-scan kernel diff (the most promising transferable lever: chunk_kda h-scan 516→281 µs)"* [42]. It also
   read trial 3's segmented-pipeline diff [43–50].
2. **Measure.** Kernel table [63–65]. nsys: *"No host gaps … wins must come from kernel time"* [82].
3. **Port with toggles** [89–103].
   - Same-job A/B: 16k/245k 8660 → 8484 µs [146].
   - h-kernel time [153]: 454 µs on Triton, 299 µs on CUDA, 475 µs on the segmented path.
   - The segmented path was reverted [202–203]. Its `final_state` plumbing stayed in `chunk_delta_h.py` and is in PR E.
4. **conv1d.** It picked the conv kernel from the profile (113 µs [106]), read the source [120–132], and swept tiles
   [139–143]. BLOCK_M=16, BLOCK_N=256, nw=2 gave 103 → 94 µs at 16k and 197 → 173 µs at 32k. Exact.
5. **Dead ends:**
   - MoE tactics are already autotuned at the exact m [171–175];
   - folding add3 into the cuBLAS beta costs +65–95 µs per GEMM [183].
6. **Unverifiable.** The transcript stops at [344], mid-exploration of the vendored `kda_ptx_prefill` backend, with no
   final message. The inert `K3_PTX` `TMP_AB` toggles in the judged tree match that exploration; their intent is unverified.
7. **Judge: PASS.** −1.6% / −0.6% / −3.1% / −2.1%; at the primary point 8612.9 → 8471.6 µs. Bit-exact.

### 8.4 `kda_lcprefill_claude` round 2 (T28, not in PR E)
- iter_01: a g/bfa GEMM side-stream overlap was rejected (*"Essentially no effect … ~0.3%"* [143]).
- The accepted change was an approximate `situ_and_mul`: 200 → 143 µs, 0.07% of elements mismatched [216].
- There is no write-up for iter_02/03.

### 8.5 `kda_lcprefill_claude` round 4 (T34), 2026-09-27
Evidence: `<base>/iter_opt_eval_k3_kda_lcprefill_claude/trial_4_agent.log`, `trial_4_opt_run/{log.md, iter_00..03/*.md, bench/}`. Precursor: `trial_3_agent.log` (FAIL, −0.73% vs a 1.64% floor). Trial 3 built the warp-specialized kernel from ncu stalls: *"1830 cycles/step"* [152], *"Four stages is the best depth"* [193].

1. **Warm start.** The agent read lcprefill trials 1–3 [14–17] and checked whether trial 3's kernel was already in the
   tree [20–21], before querying VibeSim [29–41].
2. **Port and probe.**
   - Ported trial 3's kernel: standalone 250 → 169 µs [119].
   - Ruled out: split mma chains, other stage counts [137–165], and SM-carveout overlap (517–569 µs vs 250 + 70 serial) [102].
3. **conv+l2norm fusion** (iter_02, dropped from PR E).
   - Motivation: *"162 µs against a ~40 µs memory floor"* [189].
   - A Triton fusion was exact but slower (166 vs 136 µs) [233].
   - A CUDA kernel replicating Triton's PTX math [261–271] ran 127 → 76 µs [310].
   - It first never ran: the fp32 conv weights failed `covered()` [329–337].
4. **inter_solve pin** (iter_03, kept).
   - The new profile showed `inter_solve_fused` at 165 µs, up from 110 µs [387].
   - Its own change was ruled out [397–415]. The cause: the autotune key omits T, and no autotune cache persists
     between jobs [425].
   - BK=32 / nw=1 was best or within 1 µs for T = 1.5k–64k [429], so the agent pinned it: 110 µs [443].
5. **Judge: PASS.** −1.6% / −1.6% / −3.5% / −1.9%; at the primary point 8264.9 → 8133.9 µs. Bit-exact.

**What drove the decision:**
- the agent's own trace/gap tools and nsys;
- the Triton compile artifacts (tcgen05/TMEM);
- reading the FLA source;
- bit-exact microbenchmark sweeps;
- ncu;
- the warm-start history, which carried both h-scan kernels across failed rounds;
- profile-vs-profile diffs.

**VibeSim usage:**
- **Which oracle.** kda_prefill r1/r2 and lcprefill r1/r2 queried the ≤48k `k3_kda_prefill` oracle (port 8805), not a
  long-context one.
  - Ranked by headroom, `merged_front_prefill` came first.
  - `kda_chunk_prefill` had the largest r0/r5: 37.9, 2.89 ms → 76 µs, summed over cases.
  - Only lcprefill r4 used `k3_kda_lcprefill` (port 8807). It ranked `kda_chunk_prefill` first (8.42 → 0.15 ms, 55×).
- **Read before the edits, but the node is coarse.**
  - VibeSim was always read before the edits, and the ranked node does contain the fixes. But it is the whole chunk
    chain.
  - `kernels` returned only `sglang_triton`, with no alternative [lcprefill r1 267].
  - The `idle` and `fusion` buckets were 0, so it could not see the host sync or the copies.
  - Its GEMM `has_cached_alternative: true` sent trial 2 into a dead end.
- **Citations written afterwards.** In kda_prefill r1 and lcprefill r1, the `analysis.md` files citing VibeSim were
  written after the edits [226–238], [269–277]. Each time the agent first grepped its own transcript for its VibeSim
  calls [210–219], [249–259].

**Verdict on VibeSim: confirmatory at best, misleading once (GEMMs).** It pointed at the right coarse node. Every
mechanism and every fix came from profiling, reading compiled artifacts and source, microbenchmarks, and the
warm-start history.
