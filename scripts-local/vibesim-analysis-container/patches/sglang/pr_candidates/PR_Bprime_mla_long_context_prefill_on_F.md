# PR B′ — MLA long-context chunked prefill on `trtllm_mla` (stacked on PR F)

**Status: candidate. It replaces PR B.** PR B alone was −2.6…−3.7% on the 1M prefill. Rebuilt on top of PR F and
extended with the three later long-context rounds, this PR adds **−7.7 points**: F alone is −6.9%, F + B′ is −14.6%.

A kernel-level test confirms the accuracy that the judge cannot see at ≥ 131k prefixes. B′'s attention output is as accurate as stock's or better (§4.1). Its LSE is ~10× less precise (1e-3), probably from T24.

| | |
|---|---|
| patch (**on top of PR F**, `patch -p1` inside `python/sglang`) | `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang/pr_candidates/PR_Bprime_mla_long_context_prefill_on_F.patch` |
| git | `/raid/yilegu/k3_merge`, branch `prBp_on_F` (parent = branch `prF_kvpack`, i.e. PR F) |
| exported tree used in the benchmarks (PR F + B′) | `/raid/yilegu/k3_e2e_trees/prBp_on_F` |
| size | 6 files, +2213 / −45. 1130 lines are a CuTe DSL GEMM vendored from the CUTLASS examples (BSD-3), 633 are the CuTe-DSL FMHA module |
| origin | `mla_prefill_claude` r2 (T22), `mla_lcprefill_claude` r1 (T23), r2 (T24), r5 (T32), r6 (T33). The agent's `K3_EXP_CAUSAL_FIRST` experiment knob was removed |
| upstream status | no CuTe-DSL fp8 prefill path on main. Overlaps: #41716 (prefix K/V pack), #43272 (host q/kv lengths), #40927 / #40903 (LSE base in the chunk merge) |

## 1. The problem

At long prefixes, the chunked-prefill MLA layer is dominated by the **non-causal prefix pass**: each 16k chunk
attends to every earlier chunk. At a 245k prefix that is 66% of the layer step.

**Five costs, one lever each:**

| # | cost in stock (after PR F) | lever |
|---|---|---|
| T22 | the TRT-LLM ragged fp8 cubin is slower than flashinfer's Blackwell CuTe-DSL FMHA on these 12-head shapes; the prebuilt CuTe-DSL artifacts skip the softmax correction and lose the fp8 P pre-scale (2× error) | compile the CuTe-DSL FMHA with `enable_skip_correction=False` and use it for the prefix pass only |
| T23 | that path silently falls back at bs = 1 beyond the first chunk (the `[chunks, bs+1]` cu_seqlens rows are only 8-byte aligned); one long non-causal request leaves a partial last wave | `_align16()`; a wave-aware KV split (S = 4 at ≤ 6 waves, else 2) merged in fp32 log2-LSE |
| T24 | the fp8 prefill is MUFU/issue bound in the softmax | 40 of 128 exp2 per tile on the FMA pipe (degree-2 polynomial on the fraction, `_EX2_EMULATION_COUNT = 40`) |
| T32 | `kv_b_proj` writes a bf16 `[s, h, nope + v]` projection that is then re-read to pack fp8 K/V | `mla_kv_b_proj_pack_fp8`: the GEMM epilogue writes fp8 K-nope / V straight into the packed layout; a small kernel broadcasts `k_pe` |
| T33 | the causal in-chunk pass runs after the prefix passes | issue it on the alt stream behind the first prefix launch and join before `merge_state` |

## 2. The change, by file

- **`kernels/ops/attention/cute_dsl_fmha_fp8.py` (new, 633 lines).**
  - Compiles flashinfer's `BlackwellFusedMultiHeadAttentionForward` with the softmax correction on (T22). Background:
    `PR_B_mla_prefix_cutedsl_fmha_align.md` §2.
  - `_align16`, `_num_kv_splits`, `_split_indptrs` and `_merge_kv_splits_kernel` (T23).
  - A subclass that reschedules exp2 onto the FMA pipe (T24). It uses flashinfer CuTe-DSL **internals**, the
    softmax schedule, so it is version-sensitive.
- **`kernels/ops/attention/mla_kv_b_proj_pack_fp8.py` (new, 201 lines)** and
  **`kernels/ops/gemm/dense_gemm_sm100_fp8_via_bf16_epilogue.py` (new, 1130 lines, vendored CUTLASS example).**
  T32: a persistent SM100 GEMM, `(256, 256)` tile, `(2, 2)` cluster. Its epilogue rounds to bf16, then to e4m3, so
  it is bit-identical to `kv_b_proj` followed by the unit-scale pack. Gated on CC 10.0.
- **`srt/layers/attention/trtllm_mla_backend.py` (+103):**
  - dispatches the prefix pass to the CuTe-DSL kernel, with a permanent fallback on any exception (a logged warning);
  - exposes the kv_b-fused pack.
- **`srt/models/deepseek_common/attention_forward_methods/forward_mha.py` (+201 / −45):**
  - calls the fused kv_b pack for prefix chunks;
  - `_causal_attn_mha_on_alt_stream` (T33);
  - merges behind the attention launches, so the next chunk's gather is not queued behind the overlapped causal pass.
- **`srt/layers/attention/tokenspeed_mla_backend.py` (+1):** a hook stub.

## 3. Ladder: what each round adds end to end (1M prefill, W1 TTFT, vs stock)

| tree | levers | Δ | jobs |
|---|---|---|---|
| PR F | fp8 K/V pack | −6.9% | 7622 |
| F1 + T22 + T23 | + CuTe-DSL prefix FMHA, alignment, KV split (≈ old PR B on top of F) | −10.5% | 7609 |
| **PR F + B′** | + T24 exp2, T32 kv_b fp8 epilogue, T33 causal overlap | **−14.6%** | 7638 |
| full MLA prefill lineage `cand_F` | same plus the attn-res add | −14.9% | 7511, 7579 |

## 4. Correctness and per-layer speed (single-layer judge vs stock: seed 1, fresh goldens, job 7637; tree = F + B′)

| case | points | Δ layer step | max rel err | verdict |
|---|---|---|---|---|
| MLA prefill | 16k @ 48k / 16k / 4 × 4k | **−12.0 / −4.1 / −4.9%** | 0.0141 / 0 / 0 | PASS (limit 0.02) |
| MLA long-context prefill | 16k @ 245k / 32k @ 229k / 16k @ 131k / 32k @ 131k | **−26.3 / −17.1 / −18.9 / −14.0%** | 0.0 (see below) | PASS |
| MLA mixed | 64 dec + 16k / 128 dec + 4k / 64 dec + 16k @ 48k | −28.9 / −53.3 / −15.0% | ≤ 0.0093 | PASS (mostly PR A's target) |
| MLA decode shapes | 1×1M … 1×8k | −0.7 … +0.4% | 0.0 | PASS: decode untouched |

**What the 0.0 at ≥ 131k does not show.** The new prefix kernel changes the numerics: at 16k @ 48k the judge sees
rel 0.0141. At ≥ 131k prefixes, the judge's driver uses random prefix latents, so attention averages over 131k+
near-random keys and its contribution vanishes under the bf16 residual. That 0.0 therefore says nothing about
attention accuracy there.

### 4.1 Kernel-level accuracy (job 7660)

`k3_e2e/fmha_launcher_ab/acc_prefix_fmha.py` runs the prefix pass alone (16k queries, 12 heads, fp8), on inputs with
a shared low-rank q/k component so softmax rows are peaky. Each kernel is compared against an fp32 softmax over all
keys, for 256 random query rows × 12 heads. Log: `k3_e2e/fmha_launcher_ab/logs/acc_7660.log`.

| prefix (keys) | stock TRT-LLM: out max / p99 / mean rel err | B′ CuTe-DSL: out max / p99 / mean rel err | LSE max abs err, stock / B′ |
|---|---|---|---|
| 48k | 0.0277 / 0.0204 / 0.0018 | **0.0268 / 0.0134 / 0.0015** | 6e-5 / 1.0e-3 |
| 128k | 0.0330 / 0.0191 / 0.0020 | **0.0292 / 0.0112 / 0.0013** | 9e-5 / 1.1e-3 |
| 256k | 0.0298 / 0.0195 / 0.0019 | **0.0216 / 0.0085 / 0.0013** | 9e-5 / 1.0e-3 |

- **Output: B′ meets the acceptance criterion.** Its output error is lower than stock's at every prefix and
  percentile; both are dominated by fp8 input/P quantization. B′ differs from stock by ≤ 0.059 max and 0.0019 mean
  per row.
- **LSE: B′ is ~10× less precise than stock.** Its error is ~1e-3 in natural log, against ~1e-4 for stock.
  - The likely cause is the degree-2 exp2 emulation (T24) feeding the row sum. That is unverified; the T24-off
    ablation has not been run.
  - In the chunk merge, a 1e-3 LSE error moves the chunk weights by ~0.1%, which is below the fp8 output noise. It is
    acceptable, but it should be stated in the PR, and T24 should be the separate, optional piece.

## 5. End to end (48-layer K3, dummy weights, 8 × B200; clean protocol; stock = mean of 3 runs)

| workload | metric | stock | PR F | **PR F + B′** (job 7638) | B′ increment |
|---|---|---|---|---|---|
| W1 1 × 1,040,000 → 32 | TTFT (the 1M prefill) | 55.47 s | 51.66 s (−6.9%) | **47.38 s (−14.6%)** | **−7.7 pt** |
| W6 16 × 64k → 64 | output tok/s | 41.63 | +2.0% | **+4.1%** | +2.1 pt |
| W4 16k prefill (no prefix), W2, W5 | throughput | | ±0.4% | +0.8 / +0.9 / +1.3% | ≈ 0 |

1 run. W1's TPOT reads +4.6% in this run; that is the noisy 31-step metric, and decode is bit-exact and unchanged
in the judge.

## 6. Before submitting upstream

- **Accuracy (§4.1):** the output is fine. State the LSE precision (1e-3), and run the T24-off ablation to confirm where it comes from.
- **Split it.** One PR with 2,200 lines touching flashinfer CuTe-DSL internals will not get reviewed. Suggested order:
  1. T22 + T23 (prefix FMHA with the correction on, alignment, KV split): ~300 lines, the base.
  2. T33 causal-pass overlap: small, independent.
  3. T32 kv_b fp8 epilogue: needs a vendored CuTe DSL GEMM; ask maintainers whether a CUTLASS-example vendoring is
     acceptable, or upstream the epilogue to flashinfer.
  4. T24 exp2 emulation: subclasses flashinfer internals; better proposed to flashinfer itself.
- **Measure each split end to end.** The ladder gives T22+T23 ≈ −3.6 pt on top of F, and T24/T32/T33 together
  ≈ −4.1 pt, so each piece alone may be under 3%.
- **Agree on the LSE base with #40927** (the split merge uses log2).

## 7. How the agent found it

- **T22 and T23:** see `PR_B_mla_prefix_cutedsl_fmha_align.md` §6. Profile kernel names, a log warning, source
  reading and microbenchmarks; VibeSim marginal or not used.
- **T24, T32, T33** (`mla_lcprefill_claude` r2/r5/r6): these were covered by the 2026-10-15 audit of the six
  long-context prefill rounds (`K3_VIBESIM_ATTRIBUTION.md`). None was caused by VibeSim; the targets came from the
  per-kernel profile and nsys. A per-step write-up like PR B's §6 has not been done for these three rounds.
