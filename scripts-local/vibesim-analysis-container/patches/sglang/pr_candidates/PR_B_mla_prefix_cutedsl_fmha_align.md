# PR B — MLA chunked prefill: prefix attention on the CuTe-DSL fp8 FMHA (+ 16-byte alignment fix + KV split)

**Status (2026-10-16): superseded by `PR_Bprime_mla_long_context_prefill_on_F.md`.** PR B rebuilt on PR F and extended with T24/T32/T33 adds −7.7 pt on the 1M prefill. This doc is kept for its T22/T23 detail and provenance.

**Original status: hold.** Large per-layer gains at long prefixes, but only −3.7% end to end on the 1M prefill, and ≈ 0 on every
other workload. It also adds a new kernel path that overlaps several open upstream PRs.

| | |
|---|---|
| patch (vs stock sglang v0.5.20, `patch -p1` inside `python/sglang`) | `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang/pr_candidates/PR_B_mla_prefix_cutedsl_fmha_align.patch` |
| built from (applied in order onto stock) | `…/patches/sglang/claude/mla_prefill_02_r2_prefix_attention_cutedsl_fmha.patch` (T22) + `…/patches/sglang/claude/mla_lcprefill_01_r1.patch` (T23) |
| exported tree used in the benchmarks | `/raid/yilegu/k3_e2e_trees/B_prefix_fmha` |
| size | 2 files, +290 / −0: new `kernels/ops/attention/cute_dsl_fmha_fp8.py` (273 lines) and the dispatch in `srt/layers/attention/trtllm_mla_backend.py` |
| upstream status | the CuTe-DSL fp8 prefill path does not exist on main. Overlaps: #41716 (fused prefix K/V pack + fp8 quant), #43272 (host q/kv lengths), #40927 / issue #40903 (LSE base in the chunk merge) |

## 1. The problem

**Where the cost is.** In MLA chunked prefill, each new chunk attends to (a) itself, causally, and (b) every earlier
prefix chunk, non-causally. Stock runs both passes on the TRT-LLM ragged fp8 FMHA cubin. At long prefixes the
non-causal prefix pass dominates the MLA layer: about 66% of the step at a 245k prefix.

**Why CuTe-DSL helps.** flashinfer ships a CuTe-DSL FMHA that is faster on these 12-head shapes. However, its
prebuilt artifacts use `enable_skip_correction`, which also drops the fp8 P pre-scale and doubles the row error.

## 2. The change, by file

### `kernels/ops/attention/cute_dsl_fmha_fp8.py` (new)

**(a) Compile flashinfer's Blackwell FMHA with the correction on**, which restores the 2^8 P pre-scale and
trtllm-gen-level accuracy:

```python
fmha = BlackwellFusedMultiHeadAttentionForward(
    qk_acc_dtype=cutlass.Float32, pv_acc_dtype=cutlass.Float32, mma_tiler=(128, 128),
    head_dim=(head_dim, head_dim_v), is_persistent=False,
    mask_type=WINDOW_MASK_INFERENCE if is_causal else RESIDUAL_MASK,
    enable_ex2_emulation=True,
    enable_skip_correction=False,        # <- the key flag
    use_tma_store=False)
```

**(b) T23, the 16-byte alignment fix.** The kernel is compiled for 16-byte-aligned `cum_seqlen` tensors. Rows of the
`[num_prefix_chunks, bs + 1]` int32 chunk table after row 0 are not aligned at bs = 1. Stock never hits this, because
stock never calls this kernel. But without the fix the new path silently falls back to the ragged cubin at every
prefix that is not a multiple of the chunk size. In the loop, that was the difference between −2.7% (48k prefix) and
−12.7% (245k prefix).

```python
def _align16(t):
    return t if t.data_ptr() % 16 == 0 else t.clone()
```

**(c) T23, the wave-aware KV split for one long non-causal request.** The grid is one 256-row Q tile per CTA. For a
16k chunk over a long prefix it runs ≤ 6 waves with a large partial last wave. The fix splits the KV into S = 4 (≤ 6
waves) or S = 2 (otherwise) batch entries over a replicated Q, then merges the partial outputs in fp32, base-2 LSE
(`_merge_kv_splits_kernel`).

```python
def _num_kv_splits(num_ctas, kv_len, num_sms):
    splits = 4 if num_ctas <= 6 * num_sms else 2
    while splits > 1 and kv_len < splits * _MIN_SPLIT_KV:   # 16k
        splits //= 2
    return splits
```

### `srt/layers/attention/trtllm_mla_backend.py`: dispatch

```python
def _use_cute_dsl_fp8_prefill(q, k, v, out, is_causal):
    return (_cute_dsl_fp8_prefill_ok and not is_causal          # prefix pass only
            and cute_dsl_fmha_fp8.is_supported()               # CC 10.0
            and q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
            and out is not None and out.dtype == torch.bfloat16
            and out.data_ptr() % 32 == 0)                      # 256-bit stores
```

- **Only the prefix pass.** The causal in-chunk pass stays on trtllm-gen: cute-dsl rounds differently, and on
  no-prefix batches that alone moves the layer output past the 1% mean-error budget.
- **Fallback.** Any exception permanently falls back to the cubin, with a warning (`_disable_cute_dsl_fp8_prefill`).

## 3. Per-layer results (single-layer judge, from the loop rounds)

These were measured inside the prefill-lineage trees, on top of the fused fp8 K/V pack (T21). They were not measured
as stand-alone B on stock.

| round | point | Δ layer step | numerics |
|---|---|---|---|
| T22 (`mla_prefill_02_r2`) | 1 × 16k chunk @ 48k prefix | −2.7% | rel 0.0156 |
| T23 (`mla_lcprefill_01_r1`) | 16k @ 245k / 32k @ 229k / 16k @ 131k / 32k @ 131k | **−12.7 / −7.5 / −7.4 / −4.4%** | bit-exact vs T22 |

## 4. End to end (48-layer K3, dummy weights, 8 × B200, TP8/EP8)

Setup: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/k3_e2e/RECIPE_K3_48L_8xB200.md`.

| workload | metric | stock | PR B | Δ | merged tree Δ |
|---|---|---|---|---|---|
| W1: 1 × 1,040,000 → 32 (TTFT = the 1M prefill; 5–6 runs pooled across both protocols) | TTFT | 54.4 / 54.6 / 54.5 / 55.2 / 56.1 s (mean 54.9) | 51.5 / 52.6 / 51.7 / 52.5 / 55.9 s (mean 52.8) | **−3.7%** | −19.0% |
| W4: 16 × 16k, no prefix (clean, 2 runs) | input tok/s | 46,753 / 46,715 | | −0.1% | +9.1% |
| W6: 16 × 64k (clean, 2 runs) | output tok/s | 42.0 / 40.7 | | +1.3% | +12.3% |
| W2: 128 × 8k; W5: 512 × 8k (clean) | output tok/s | | | +0.3% / −0.1% | +7.0% / +7.2% |

- **W4 is unaffected:** prefills without a prefix never take the prefix pass.
- **A retracted claim:** the earlier single-run "+37% on W6" and "+19% slower on W4" were workload-order artefacts.
  See `K3_E2E_BENCHMARK_REPORT.md` §12.

**Why −12.7% per layer becomes −3.7% end to end:**
- Only 12 of 48 layers are MLA.
- Only chunks after the first have a prefix.
- The 1M prefill is also MoE- and KDA-heavy.

The merged tree's −19% on the same prefill comes from these levers *plus* the fused fp8 K/V pack, the KDA CUDA chunk
scan and the other prefill levers acting together.

## 5. Why hold

- **The stand-alone gain is small (−3.7%)** against a new 273-line kernel module and a new compile-time dependency on
  flashinfer's CuTe-DSL internals.
- **It overlaps open PRs:** #41716 (prefix K/V pack), #43272 (host lengths) and #40927 (natural-log vs log2 LSE in the
  chunk merge). The KV-split merge here uses log2, so it has to match whatever #40927 settles on.
- **Better as part of a larger long-context prefill PR**, together with the fused fp8 K/V pack (T21/T32) and the
  causal-pass overlap (T33) once those are re-derived on main. Measured together, those levers deliver the merged
  tree's −19% on the 1M prefill.

## 6. How the agent found it (Claude Opus 5.5)

### 6.1 T22: CuTe-DSL prefix FMHA (`mla_prefill_claude`, round 2, 2026-09-25)

Evidence: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/iter_opt_eval_k3_mla_prefill_claude/trial_2_agent.log`
and `trial_2_opt_run/{log.md, iter_00/analysis.md, iter_01/hypothesis.md, iter_01/analysis.md}`.

1. **Measure.** In the `--profile-kernels` table at 16k @ 48k, `fmhaSm100aKernel_…DenseVarSeqQ256Kv128` was the top
   kernel at 3.52 ms, ahead of the merged front GEMM (2.29 ms). A private per-launch trace driver confirmed this:
   prefix FMHA 3.33 ms.
2. **Choose a replaceable target.** The merged front was larger in the model, but it is a closed cuBLAS kernel. The
   agent's note: *"the merged front is the largest node, but it is a closed cuBLAS nvjet kernel … no code-level
   lever"*. So it went for the prefix FMHA, which it judged *"the largest replaceable node"*.
3. **Find the alternative.** Reading flashinfer's `trtllm_ragged_attention_deepseek` source, it noticed the wrapper
   *"also exposes a cute-dsl (FA4-style) ragged-prefill backend"*. A microbenchmark on the exact shapes gave prefix
   3.47 → 2.86 ms and causal 0.72 → 0.55 ms.
4. **First attempt failed numerics** (CHECK mean error 0.021 on two points). The agent then:
   - benchmarked accuracy against an fp32 reference: cute-dsl had 2× trtllm-gen's error;
   - read `fmha.py` and found `enable_skip_correction` → `rescale_threshold = 8.0`, which drops the fp8 P pre-scale;
   - built its own JIT variant with the correction on, which matched trtllm-gen accuracy (0.0107 vs 0.0110).
5. **Speed vs accuracy.** It swept compile options. No accurate variant reached the prebuilt artifact's 2.86 ms (best:
   3.29 ms). It accepted the accurate one.
6. **Scope.** The causal pass still failed CHECK, so it restricted the new path to the non-causal prefix pass. It also
   rejected the inaccurate prebuilt artifact even there, because it noticed the driver's prefix V was all-zero, so
   CHECK could not see that accuracy loss.
7. **Validation.**
   - Interleaved 3 × 3 A/B: −2.1% (≈ 7σ).
   - `LAYER_SMOKE_OK`.
   - Judge: −2.7% (11,539 → 11,229 µs).

**VibeSim (oracle `k3_mla_prefill`, port 8806, the pre-correction bake): marginal.**
- **What it got right:** its prefix-attention row (3.53 ms) matched the measurement.
- **What it got wrong:**
  - It ranked `merged_front_prefill` first, inflated ~2.4×. A later VibeSim fix (B12c) confirmed the prefill model was
    mis-composed.
  - Its "attention subtree R0/R5 6.4" signal came from tiny leaves (`mla_cache_append` 64×), not from prefix attention.
  - `kernels` always errored (`kernels requires kernel_set`), so it never suggested an alternative kernel.
- **How the agent used it:** the agent's iter_01 VibeSim fetch came after the edit was validated, and returned
  byte-identical output (*"VibeSim models the design, not our profile"*).
- **What drove the decision:** profiling, the closed-cuBLAS argument, source reading and microbenchmarks.

### 6.2 T23: 16-byte alignment fix + KV split (`mla_lcprefill_claude`, round 1, 2026-09-27)

Evidence: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/iter_opt_eval_k3_mla_lcprefill_claude/trial_1_agent.log`
and `trial_1_opt_run/iter_0{0,1,2,2b,3}/`.

1. **Found a silent fallback.**
   - At 16k @ 245k the kernel table still showed the trtllm-gen `fmhaSm100aKernel_…Dense`, although the tree should
     have been running cute-dsl.
   - Grepping its own logs turned up the warning *"cute-dsl fp8 ragged prefill unavailable (Misaligned Tensor data on
     argument #6 … expected data alignment=16 bytes); using trtllm-gen"*.
   - Prefix FMHA was 16.7 of 25.1 ms (66% of the step).
2. **Root cause from source.**
   - `prefix_chunk_cu_seq_lens` is a `[num_chunks, bs + 1]` int32 tensor, so at bs = 1 row 1 sits at +8 bytes.
   - Above 128k the prefix splits into 2 chunks, chunk 1 fails the alignment guard, and
     `_disable_cute_dsl_fp8_prefill` turns the path off **for the whole process**. That is why even single-chunk points
     ran the old kernel.
3. **Fix:** `_align16()`. FMHA 16.7 → 15.1 ms, CHECK exact.
4. **KV split: the agent's own wave arithmetic.** *"64 q-tiles × 12 heads = 768 CTAs … 5.19 waves executed as 6 (last
   wave 28/148 SMs busy) … ~13.5% of the FMHA is tail"*. The idea came from the b512 tail-split precedent in the
   knowledge base.
5. **Iterations.**
   - The first merge used natural-log LSE, but the kernel emits log2, so CHECK failed and the agent fixed it.
   - A slow `expand().reshape` query copy was replaced with a memcpy.
   - A wave-cost model picked S = 5, but the 32k points regressed.
   - An S-sweep microbenchmark (median of 6 interleaved runs) gave the measured policy: S = 4 if ≤ 6 waves, else 2.
     It overrode the model.
6. **Judge:** −12.7% at 16k @ 245k (25,197 → 21,998 µs). Exact.

**VibeSim: not used in any causal sense, and misleading at the margin.**
- **Wrong oracle.** The agent queried the ≤ 48k, 3-case prefill oracle (port 8806). The long-context oracles and the
  corrected B12c bake didn't exist yet.
- **Ranking ignored.** It again ranked the inflated merged-front GEMM first. The agent had to scale the prefix row 5×
  by hand (*"at 245,760 prefix it is 5× the modeled 49k value"*) and then ignored the ranking.
- **Blind to the real bugs.** The model has no notion of the silent fallback, the alignment, or wave quantization.
- **Unsupported claims.** `iter_01/analysis.md` says "re-fetch identical", but the transcript contains no later
  VibeSim call. Treat the citations in iter_01–03 as unsupported.

**Verdict for PR B: VibeSim was marginal for T22 and not used for T23.** Both discoveries came from:
- reading kernel names in the profile;
- the agent's own log warnings;
- reading flashinfer and sglang source;
- accuracy and split-factor microbenchmarks.

(`vibesim_usage_audit.py` undercounts these calls: 3 and 2, vs ~10 and ~7 counted by hand. It only matches the literal
`api/v1/<verb>` form.)
