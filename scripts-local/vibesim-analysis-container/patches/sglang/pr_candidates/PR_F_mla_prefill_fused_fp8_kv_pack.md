# PR F — MLA chunked prefill on `trtllm_mla`: pack and quantize K/V to fp8 in one launch

**Status: candidate. It is the smallest new PR: 5 files, +108/−18, and bit-exact.** It cuts the 1M-token prefill by
**6.9%** end to end. The problem is still present on sglang `main`.

| | |
|---|---|
| patch (vs stock sglang v0.5.20, `patch -p1` inside `python/sglang`) | `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang/pr_candidates/PR_F_mla_prefill_fused_fp8_kv_pack.patch` |
| git | `/raid/yilegu/k3_merge`, branch `prF_kvpack`, commit `d05179b` (parent = branch `pristine`) |
| exported tree used in the benchmarks | `/raid/yilegu/k3_e2e_trees/prF_kvpack` |
| size | 5 files, +108 / −18 |
| origin | `mla_prefill_claude` round 1 (T21), minus its attn-res fused add (≈ 0 end to end) |
| upstream status (main `0d0be8fc98`, 2026-10-11) | `trtllm_mla` on main still has no `pack_extend_kv` / `pack_prefix_chunk_kv` / `prepare_chunked_prefill_qkv`, so it still runs the bf16 concat + strided cast. Main already ships `mla_kv_pack_quantize_fp8`, but only the tokenspeed backend calls it. Check overlap with open #41716 (fused prefix K/V pack + fp8 quant) before opening |

## 1. The problem

In MLA chunked prefill, the non-absorbed (MHA) path builds K = `[k_nope | k_pe]` and V from `kv_b_proj`'s output,
for the current chunk and for every cached prefix chunk. Then it feeds them to the TRT-LLM ragged fp8 FMHA. With an
fp8 KV cache, stock `trtllm_mla` does this in several passes:

- **The concat.** `_concat_and_cast_mha_k` concatenates `k_nope` and `k_pe` in bf16. Its fast paths need a
  power-of-two head count. Kimi-K3 has 12 heads per rank at TP8, so it falls back to aten slice copies.
- **Two strided casts.** `_quantize_fp8_qkv` then casts K and V to fp8 with strided `.to(fp8)` copies.
- **Q is cast twice per layer,** once per ragged launch (the causal pass and each prefix pass).

**Cost.** At 1 × 16k on a 48k prefix, this is ~800 µs of `aten::copy_` / `ConvertToFloat8E4M3fn` kernels per MLA
layer, about 7% of the layer step.

**The tool already exists in the tree.** Stock already has `kernels/ops/attention/mla_kv_pack_quantize_fp8.py`, a
Triton kernel that builds fp8 K and V in one launch, and `forward_mha.py` already calls backend hooks for it. Only
the tokenspeed backend implements those hooks; `trtllm_mla` does not.

## 2. The change, by file

### `srt/layers/attention/trtllm_mla_backend.py`: implement the three hooks

```python
def _packs_fp8_prefill_kv(self, forward_batch) -> bool:
    return (self.data_type == torch.float8_e4m3fn
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and self.forward_prefill_metadata is not None
            and not self.forward_prefill_metadata.fallback_to_flashinfer_impl)

def pack_extend_kv(self, k_nope, k_pe, v, layer, forward_batch):      # current chunk
    if not self._packs_fp8_prefill_kv(forward_batch): return None
    if layer.k_scale_float not in (None, 1.0) or layer.v_scale_float not in (None, 1.0): return None
    return mla_kv_pack_quantize_fp8(k_nope, k_pe, v, enable_pdl=False)

def pack_prefix_chunk_kv(self, k_nope, k_pe, v):                     # every prefix chunk
    if self.data_type == torch.float8_e4m3fn:
        return mla_kv_pack_quantize_fp8(k_nope, k_pe, v, enable_pdl=False)
    ...                                                               # bf16: plain concat

def prepare_chunked_prefill_qkv(self, q, k, v, forward_batch):       # cast Q once per layer
    if self._packs_fp8_prefill_kv(forward_batch): q = q.to(torch.float8_e4m3fn)
    return q, k, v
```

- **Why it is bit-exact.** The pack writes `bf16 → fp32 → e4m3` round-to-nearest-even at unit scale, which is what
  the old `concat + .to(fp8)` produced.
- **Scale handling.** `_quantize_fp8_qkv` now leaves already-fp8 K/V alone (scale 1.0). Checkpoints with
  `k_scale ≠ 1` keep the old path.

### `srt/models/deepseek_common/attention_forward_methods/forward_mha.py`: call the hook for the current chunk

```python
pack_extend_kv = getattr(backend, "pack_extend_kv", None)
if pack_extend_kv is not None and self.qk_rope_head_dim > 0:
    packed = pack_extend_kv(k_nope, k_pe, v, self.attn_mha, forward_batch)
    if packed is not None:
        return q, packed[0], packed[1], forward_batch
k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)   # unchanged fallback
```

### Two small gather fixes on the prefix path

| file | change | measured (B200) |
|---|---|---|
| `kernels/ops/kvcache/mla_buffer.py`, `get_mla_kv_buffer_kernel` | 8 rows per Triton program with 4 warps, instead of one 576-byte row per program | 31 → 14 µs for 49,152 rows, bit-exact |
| `kernels/ops/kvcache/kv_indices.py` + `forward_batch_deepseek_mha_mixin.py`, `create_chunked_prefix_cache_kv_indices` | optional grid axis 1, so one request's index copy runs on up to 64 CTAs (sized on the host from `prefix_chunk_max_seq_lens`, no sync) | 52 → 3.2 µs, bit-exact |

## 3. What was dropped from the loop's version

- **The attn-res TMA kernel with the residual add fused** (same round). It is Kimi-specific and ≈ 0 end to end
  (`cand_E2_attnres_add`, job 7577; and the F1 with-add vs PR F without-add runs both gave −6.6…−6.9% on the 1M
  prefill).

## 4. Correctness and per-layer speed (single-layer judge vs stock: seed 1, fresh goldens, job 7621)

| case (backend) | points | Δ layer step | max abs err | verdict |
|---|---|---|---|---|
| MLA prefill (trtllm_mla) | 16k @ 48k / 16k / 4 × 4k | **−5.5 / −3.1 / −4.5%** | 0.0 | PASS |
| MLA long-context prefill | 16k @ 245k / 32k @ 229k / 16k @ 131k / 32k @ 131k | **−8.8 / −5.4 / −7.9 / −5.2%** | 0.0 | PASS |
| MLA mixed (decode rows + chunk) | 64 dec + 16k / 128 dec + 4k / 64 dec + 16k @ 48k | −21.7 / −33.5 / −12.6% | 0.0 | PASS |
| MLA decode shapes | 1×1M / 16×64k mix / 128×8k / 32×8k / 1×8k | −0.7 / +0.4 / 0.0 / −0.7 / 0.0% | 0.0 | PASS: decode untouched |

**About the mixed-batch numbers.** They are large only because stock runs every decode row of a mixed batch through
the chunked-prefix path, which is the bug PR A fixes. With PR A applied, most of that work disappears, so don't
count these gains on top of PR A's.

**About the 0.0 errors.** This patch is bit-exact by construction. Note, though, that the judge's prefill check
compares the layer's final hidden state, which may hide small attention differences behind the bf16 residual add.
For this PR that doesn't matter (exact), but a kernel-level test is still needed upstream (§6).

## 5. End to end (48-layer K3, dummy weights, 8 × B200, TP8/EP8; clean protocol)

Setup: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/k3_e2e/RECIPE_K3_48L_8xB200.md`.
Workloads `smoke w1 w2 w4 w5 w6`, same order, `--flush-cache`. Stock 3 runs (jobs 6906, 6911, 7514).

| workload | metric | stock (mean of 3) | PR F (job 7622) | Δ | F1 = PR F + attn-res add (2 runs) | MLA prefill lineage (`cand_F`, 2 runs) |
|---|---|---|---|---|---|---|
| W1 1 × 1,040,000 → 32 | TTFT (the 1M prefill) | 55.47 s | 51.66 s | **−6.9%** | −6.6% | −14.9% |
| W6 16 × 64k → 64 | output tok/s | 41.63 | 42.48 | +2.0% | +2.4% (1 run) | +3.7% |
| W4 16 × 16k prefill (no prefix) | input tok/s | 46,742 | 46,728 | 0.0% | +0.3% | +0.7% |
| W2 128 × 8k, W5 512 × 8k | output tok/s | | | +0.3 / +0.4% | | |

**Where it acts.** Only prefills with a cached prefix gain. At 1M that is every chunk after the first: 63 of 64
chunks, 12 MLA layers each. Prefix-free prefills (W4) and decode are unchanged, as expected.

**How it relates to PR B.** The MLA prefill lineage reaches −14.9% on the same workload:
- PR F: −6.9%.
- + T22/T23 (PR B's prefix FMHA + alignment + KV split): −10.5%.
- + T24 exp2, T32 kv_b fp8 epilogue, T33 causal overlap: −14.9%.

PR B should therefore be rebased on top of PR F and extended with T24/T32/T33. That gives a second PR worth about a
further 8 points.

## 6. Before submitting upstream

- **Coordinate with #41716**, an open PR for a fused prefix K/V pack + fp8 quant. If it lands first, PR F shrinks to
  the `trtllm_mla` hooks.
- **Rebase onto main.** `trtllm_mla_backend.py`, the mixin and `forward_mha.py` all moved, so expect a hand re-port;
  the logic is small.
- **Unit test:** pack vs `concat + .to(fp8)` bit-exactness over head counts (12, 16, 128), tokens and
  `qk_rope_head_dim`; plus the scaled-checkpoint fallback (`k_scale ≠ 1`).
- **Real model:** gsm8k with chunked prefill on DeepSeek-V2/V3 (any MLA model on `trtllm_mla` with an fp8 KV cache).
  Measure throughput on a long-prompt workload.

## 7. How the agent found it (`mla_prefill_claude`, round 1, Claude Opus 5.5, 2026-09-25)

Trial 1 = round 1 (`rounds.jsonl`). `export_k3_patches.sh` diffs `iter_opt_eval_k3_mla/best_tree` against
`trial_1_tree_judged` to produce `mla_prefill_01_r1_kv_pack_quantize_fp8_attnres_fused_add.patch`.

Evidence: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/iter_opt_eval_k3_mla_prefill_claude/trial_1_agent.log`
(transcript) and `trial_1_opt_run/{log.md, iter_00/{analysis.md,opt.json,an_*.json}, iter_01..03/{hypothesis,analysis}.md,
iter_04/diff_rejected.patch, null_ab/}`. Microbenchmarks: `/tmp/mb_pack.py`, `/tmp/mb_getkv.py` inside the agent
container; only their output is in the transcript.

Iterations and what reached PR F:

| Iteration | Change | In PR F? |
|---|---|---|
| iter_01 | fp8 K/V pack + quantize for trtllm_mla, plus a single Q cast | yes |
| iter_02 | attn-res fused add (T20) | no |
| iter_03 | 2D-grid prefix kv-index copy | yes |
| iter_05 | 8-rows-per-program `get_mla_kv_buffer` | yes |
| iter_04 | cute-dsl prefix FMHA, rejected here; it became T22 in round 2 (PR B) | no |

1. **Measure, then VibeSim, then the agent's own trace.**
   - The kernel table at pf49152 [16–17] put the prefix FMHA at 3502 µs and the front GEMM at 2289 µs. Strided
     `direct_copy_kernel`s appeared further down.
   - VibeSim next [19–29], with the usual handle failures before `.:k3_mla_prefill` resolved [21–27].
   - Then a shape-annotated trace tool [43–49]. It showed `aten::copy_` on `[16384,12,128]` and `[16384,12,64]`, plus
     `ConvertToFloat8E4M3fn` copies. `iter_00/analysis.md` totals these at *"~800 us of aten copy / fp8-convert
     elementwise kernels in the attention data path (… 172+68+61+167 us; chunk: 58+24+24+22+58+26 us)"*.
2. **Rule out the big nodes by argument.** The same note says:
   - *"GEMMs run at ~73% of the bf16 dense peak (compute-bound, closed nvjet kernels)"*;
   - *"the FMHA is bounded by MUFU exp (… ~2.1 ms floor)"*.
   So the copies were the target.
3. **Root cause from source** [51–76].
   - In `trtllm_mla_backend.py` and `forward_mha.py`, `forward_mha.py` already had backend hooks `pack_prefix_chunk_kv`
     and `prepare_chunked_prefill_qkv` [63–64]. The TRT-LLM backend did not implement them.
   - `_concat_and_cast_mha_k`'s fast paths require a power-of-two head count. With K3's 12 heads, K/V fell back to
     aten slice copies followed by a strided `.to(fp8)` [72–76].
   - A grep found the existing Triton kernel `kernels/ops/attention/mla_kv_pack_quantize_fp8.py` [71–74].
   - Q was cast to fp8 once per ragged launch, i.e. twice per layer.
4. **Microbenchmark before editing** [90–91]. The ref path was bit-identical to the pack kernel at every size:

   | Tokens | ref (µs) | pack (µs) |
   |---|---|---|
   | 49152 | 456 | 73 |
   | 16384 | 162 | 28 |
   | 4096 | 41 | 28 |

   The agent wrote `iter_00/analysis.md` and `iter_01/hypothesis.md` at this point [93], before the edit. The
   hypothesis records why the result is identical: bf16 → fp32 → e4m3 RNE at unit scale. Scaled checkpoints
   (k_scale ≠ 1) keep the old path.
5. **Implement and validate.**
   - Backend hooks `pack_extend_kv`, `pack_prefix_chunk_kv` and `prepare_chunked_prefill_qkv` [95–102].
   - Replay bit-exact [107].
   - 3-round A/B [113–118]: pf49152 12096 → 11609 µs (+4.0%), pf +2.5%, 4×4096 +1.7%.
   - Re-profile: the fp8 K/V pack takes 101 µs for 2 calls, *"(was ~800 us of aten copy/convert)"* (`iter_01/analysis.md`).
6. **kv-index copy (iter_03).**
   - The iter_01 profile listed `create_chunked_prefix_cache_kv_indices` at 50 µs. The agent called it *"the next
     small target"* [232].
   - Source reading showed `grid = (batch_size,)`: 1 CTA walking 96 × 512 indices serially [233–235].
   - `prefix_chunk_max_seq_lens` is already on the host, *"so the grid can be sized from it with no sync"* [281].
   - Result: 52 → 3.2 µs, bit-exact [288].
7. **`get_mla_kv_buffer` (iter_05).**
   - Its source had one program per 576-byte row [297–303].
   - A rows/warps sweep [305–312] found *"8 rows per program with 4 warps: 31 → 14 µs, bit-exact"* [314].
   - **Could not verify the implementation step.** The transcript ends at [388], at 05:37. That is before
     `mla_buffer.py` in `trial_1_tree_judged` was modified (05:38), and no `iter_05/` directory exists. Only the
     microbenchmark and the plan are evidenced.
8. **Noise control.** A null A/B on identical trees [315–341] gave pf49152 −0.15%, pf +1.99% and 4×4096 −0.93%. The
   agent concluded that secondary-point deltas under ~2% are not significant (log.md).
9. **Judge: PASS.**
   - 12128.5 → 11573.4 µs (−4.6%) at pf49152 (primary), 8573 → 8226 µs (−4.0%), 8058 → 7848 µs (−2.6%).
   - Bit-exact on all 3 points (the attn-res add was still included in this judged tree).

**What drove the decision:**
- shape-annotated profiler traces showing the copy and convert kernels;
- the compute-bound / MUFU-floor argument that set aside the GEMMs and the FMHA;
- reading the attention dispatch, which found the missing backend hooks and the power-of-two head-count fallback;
- reuse of a kernel already in the tree;
- bit-exactness microbenchmarks for the pack, the index copy and the row gather;
- a null A/B to calibrate noise.

**VibeSim usage** (oracle `k3_mla_prefill`, port 8806, fixed before-state, summed over 3 cases):
- **Read before the target was chosen** [19–29]. Ranked by headroom:

  | Node | R0 / R5 (ms) | R0/R5 |
  |---|---|---|
  | `merged_front_prefill` (first) | 16.8 / 5.0 | 3.4 |
  | attention + output-gate subtree | — | 6.4 |
  | `mxfp4_fused_moe_prefill` | — | 3.0 |
  | `mla_cache_append` | — | 63.7 |
  | `mla_prefix_gather` | — | 7.2 |

- **No node for the main fix.** The agent attached it to the subtree by analogy: *"maps to R0 >> R5 on the attention
  subtree / mla_cache_append-like memory ops"*. `mla_cache_append` is the KV-cache write, not the FMHA K/V pack.
  Section 6.1 of PR B notes that this subtree's 6.4 comes from tiny leaves.
- **`mla_prefix_gather` covers the two small pieces.** In `main/profiling/kernels/mla_prefix_gather.py` it is the
  "chunked-prefix MLA index and latent-KV gather", so it spans both the kv-index copy and `get_mla_kv_buffer`. The
  agent listed it in iter_00, before targeting either piece, and cites it in `iter_03/hypothesis.md`. But its visible
  reasoning at [232] and [296] quotes the measured 50 µs and 34 µs kernel times. Whether VibeSim's 7.2× steered that
  choice cannot be determined from the transcript.
- **Static before-state.** In iter_02 VibeSim scored attn_res at r0/r5 1.00 and had no node for the residual add
  [225–226]. The prediction is fixed (*"simulate is idempotent, so the post-edit analysis is measured"*).
- **`kernels` was never usable:** it returned `kernels requires kernel_set` [23–24], and the agent never retried it with a `kernel_set`.

**Verdict on VibeSim: not causal for the main lever (−4% from the K/V pack); at most consistent with the two small
gather fixes.** It ranked a closed cuBLAS GEMM first and had no node for the copies. The agent found the target in
its own trace and the mechanism in sglang source.
