# PR A — MLA mixed (EXTEND) batches: serve the decode rows with the absorbed decode kernel

**Status: recommended first upstream PR.** The only stand-alone patch with a ≥ 10% end-to-end gain.

| | |
|---|---|
| patch (vs stock sglang v0.5.20, `patch -p1` inside `python/sglang`) | `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang/pr_candidates/PR_A_mla_mixed_batch_decode_rows.patch` |
| git | `/raid/yilegu/k3_merge`, branch `prA_mixed`, commit `22036f7` (parent = branch `pristine`) |
| exported tree used in the benchmarks | `/raid/yilegu/k3_e2e_trees/prA_mixed` |
| size | 3 files, +273 / −27 |
| origin | hand-port of the loop's `mla_mixed_01_r2` + `mla_mixed_02_r4` (T39 + T40) onto stock, without the prefill-lineage levers they were built on |
| upstream status (main `0d0be8fc98`, 2026-10-11) | problem still present (`handle_attention_trtllm_mla` unchanged); precedent: open #43238 does the same for the MHA backend |

## 1. The problem

With `--enable-mixed-chunk`, sglang appends the running requests' decode tokens to a prefill chunk, and the batch
becomes `EXTEND`. For MLA models on `trtllm_mla`, `handle_attention_trtllm_mla`
(`srt/models/deepseek_common/attention_backend_handler.py`) sends every `is_extend_without_speculative()` batch down
the **chunked-prefix MHA** path. Each 1-token decode row is then treated like a prefill request:

- its whole cached context is re-read through the non-absorbed MHA (K/V up-projected from the latent cache), in
  prefix chunks of `capacity // batch_size`;
- with 64–128 decode rows, that is 64–128 tiny prefix passes per layer.

Production decode does the same work with one **absorbed** MLA decode kernel over the latent cache. So, in stock
sglang, turning on mixed chunks makes serving *slower*: at 128 × 8k, TPOT is 77.7 ms with mixed chunks vs 50.7 ms
without.

## 2. The change, by file

### `srt/models/deepseek_common/attention_forward_methods/forward_mha.py` — split the batch, route the decode rows

```python
def _mixed_decode_split(self, backend, forward_batch):
    """(prefill requests, prefill tokens) when the batch ends in 1-token
    requests with a cached prefix (the decode rows a MIXED batch appends)..."""
    ...
    first = len(seq_lens)
    while first > 0 and seq_lens[first - 1] == 1 and prefix_lens[first - 1] > 0:
        first -= 1
    if first == 0 or first == len(seq_lens):
        return None            # pure prefill or pure decode -> unchanged path
    return first, sum(seq_lens[:first])
```

- **Detects the tail of decode rows:** mixed batches put the decode requests last, each with `extend_seq_len == 1`
  and a non-empty prefix.
- **Returns `None` otherwise:** for pure-prefill and pure-decode batches, so their behaviour is byte-identical to stock.
- **Conservative gates:** off for HIP, DSA, deep-gemm bmm, non-bf16 `w_kc`/`w_vc`, dense-prefix KV, and during CUDA
  graph capture.

In `forward_normal_chunked_kv_core`:

```python
decode_split = self._mixed_decode_split(backend, forward_batch)
...
q_prefix = q if decode_split is None else q[:num_prefill_tokens]   # causal + prefix passes: prefill rows only
...
decode_output = self._mixed_decode_rows_attn(backend, q_decode, forward_batch, num_prefill_reqs)
attn_output = torch.cat([attn_output[:num_prefill_tokens].reshape(-1, hidden),
                         decode_output.view(-1, hidden)], dim=0)
```

`_mixed_decode_rows_attn` is the absorbed MLA decode for those rows:

```python
q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc).transpose(0, 1)    # absorb W_UK into q
latent = backend.forward_mixed_decode_rows(q_nope_out, q_pe, self.attn_mqa, forward_batch, first_req)
torch.bmm(latent.transpose(0, 1), self.w_vc, out=output.transpose(0, 1))      # W_UV on the latent output
```

- **Optional overlap:** when the prefill part is ≥ 8192 tokens and an alt stream exists, the decode rows run on the
  alt stream (`_mixed_decode_rows_attn_on_alt`). Their small grids then fill the SMs that the causal and prefix
  passes' last waves leave idle.

### `srt/layers/attention/trtllm_mla_backend.py` — decode kernel for a sub-range of an extend batch

- **New `supports_mixed_decode_rows(forward_batch)`.** It requires:
  - trtllm-gen backend and fp8 KV;
  - no DCP;
  - an extend batch that is not target-verify or draft-extend;
  - no FlashInfer fallback.
- **New `forward_mixed_decode_rows(...)`.** It runs the existing `_run_decode_kernel` on rows `[first_req:]`:
  - Builds a block table for them **once per forward**, cached on `forward_batch.mixed_decode_metadata`, so all 12
    MLA layers reuse it.
  - Quantizes the query to fp8 the way the decode path does.
  - Uses **its own workspace buffer**, so it can run concurrently with the extend passes.
- **Prefix and causal passes now honour a request count:**
  - `prefix_chunk_num_reqs` for the prefix pass, and `causal_num_reqs` for the causal pass;
  - they slice `cum_seq_lens` / `seq_lens` to the prefill requests.

### `srt/model_executor/forward_batch_deepseek_mha_mixin.py` — prefix chunks for the prefill requests only

- **New fields:** `prefix_chunk_num_reqs`, `causal_num_reqs` and `mixed_decode_metadata`.
- **`prepare_chunked_prefix_cache_info(num_reqs=...)`** builds the chunk tables only for the leading prefill requests.
- **`prefix_chunk_len` is still `capacity // batch_size` over the whole batch, on purpose.** The kernels write log2
  LSE that `merge_state` combines as natural log, so moving chunk boundaries would change the prefill rows' output.
  For the same reason the prefix chunks are not merged into one.

## 3. What was dropped from the loop's version

- **`K3_AB` env knob:** removed; the behaviour is unconditional.
- **The prefill-lineage dependencies:** the CuTe-DSL prefix FMHA, the KV split, the alt-stream causal-pass variant and
  `seq_lens_cpu` plumbing. Stock's TRT-LLM ragged prefix path is kept.

## 4. Correctness (single-layer judge vs stock: seed 1, fresh goldens, job 6858)

| case | backend | points | Δ step time | max rel error | verdict |
|---|---|---|---|---|---|
| MLA mixed | trtllm_mla | 64 decodes @8k + 16k chunk | **−60.2%** (21,239 → 8,456 µs) | 0.0113 | PASS |
| | | 128 decodes @8k + 4k chunk | **−89.1%** (24,737 → 2,699 µs) | 0.0097 | |
| | | 64 decodes @8k + 16k chunk @48k prefix | **−65.0%** (44,626 → 15,611 µs) | 0.0111 | |
| MLA decode shapes | cutedsl_mla | 1×1M / 16×64k mix / 128×8k / 32×8k / 1×8k | −0.7 / +0.4 / 0.0 / −0.7 / 0.0% | **0.0 (bit-exact)** | PASS: no effect outside mixed batches |

For comparison, the merged all-optimizations tree reaches −61.5 / −89.9 / −71.2% on the mixed points, so PR A carries
nearly all of that single-layer gain.

## 5. End to end (48-layer K3, dummy weights, 8 × B200, TP8/EP8, `--enable-mixed-chunk`)

Setup: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/k3_e2e/RECIPE_K3_48L_8xB200.md`.

- Workloads: `smoke w3 w2`. 3 runs each, the same order for every run.
- Jobs: stock 6810 / 6899 / 6902, PR A 6859 / 6900 / 6903, merged-tree 6811.

| workload | metric | stock | PR A | Δ | merged tree (1 run) Δ |
|---|---|---|---|---|---|
| W3: 96 × 16k → 512 arriving at 2 req/s, concurrency 64 (decodes run while new prompts prefill) | TPOT | 59.9 / 59.5 / 60.1 ms | 46.1 / 46.1 / 46.1 ms | **−23.1%** | −46.3% |
| | output tok/s | 744 / 748 / 743 | 919 / 919 / 919 | **+23.5%** | +38.1% |
| | mean TTFT | 2.89 / 2.87 / 2.87 s | 1.73 / 1.77 / 1.73 s | **−39.7%** | −57.7% |
| W2: 128 × 8k → 256, concurrency 128 | TPOT | 77.7 / 76.9 / 76.7 ms | 57.8 / 57.5 / 57.6 ms | **−25.1%** | −36.0% |
| | output tok/s | 909 / 917 / 921 | 1,148 / 1,172 / 1,141 | **+25.2%** | +47.5% |

- **Run-to-run spread:** ≤ 1% within each tree.
- **Scope:** the patch only acts on mixed batches. Without `--enable-mixed-chunk` it is a no-op (see the bit-exact
  decode-shape check).

## 6. Before submitting upstream

- **Rebase onto main.** All three files have moved upstream (#42753 reworked the mixin and backend), so this is a hand
  re-port.
- **Use a `msgspec.Struct` instead of new mixin fields.** The contribution guide asks for that.
- **Real-model check:**
  - gsm8k with `--enable-mixed-chunk`, plus throughput on a real MLA model (DeepSeek-V2/V3 or K3).
  - The dispatch code is shared by every DeepSeek-family model on `trtllm_mla`, so DeepSeek is a valid test.
- **Unit test:** a mixed batch (decode rows with prefix + one prefill chunk) compared against running the two parts
  separately.
- **Cite #43238** as the MHA-backend precedent.

## 7. How the agent found it (case `mla_mixed_claude`, Claude Opus 5.5)

Evidence: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/iter_opt_eval_k3_mla_mixed_claude/`:
- `trial_2_agent.log` and `trial_2_opt_run/{log.md, iter_00/analysis.md, iter_01/hypothesis.md}` (round 2);
- `trial_4_*` (round 4);
- `trial_1_opt_run/stage.diff` (round 1).

### 7.1 Round 2: the dispatch fix (−45% layer step)

1. **Warm start.** The agent read the knowledge base. It contained the decode kernel to reuse (trtllm-gen) but no hint
   of the routing change.
2. **VibeSim and code reading, while the baseline ran.** It queried the oracle and read `forward_normal_chunked_kv_core`
   (`forward_mha.py`) and the driver's MIXED-batch construction.
3. **Profile.** The primary point's prefill chunk has no prefix, yet the kernel table showed prefix FMHA ×5 (3.4 ms),
   `fixup_zero_kv_rows` (2.8 ms) and kv_b GEMMs (1.1 ms).
   - So all of that prefix work belonged to the **64 decode rows**: each was re-reading its 8k context through chunked
     MHA, in chunks of 128k // 65 = 2016 tokens.
4. **Localization by code reading.** The mixin's `prepare_chunked_prefix_cache_info`, the backend's prefix and decode
   branches, and the existing absorb path in `forward_mla.py` (`w_kc` / `w_vc`).
5. **Implementation, first replay.** It split the batch: decode rows go `q·w_kc` → trtllm-gen decode over the fp8
   latent cache → `·w_vc`. CHECK passed on all 3 points.
6. **Alternative tried and rejected.**
   - It also tried sizing the prefill request's prefix chunks by the prefill requests only, i.e. one 49k chunk. That
     was faster (11.1 ms at p49152) but failed CHECK.
   - Debugging (`dbg/lse_test.py`) showed the kernels emit **log2** LSE while `merge_state` combines it as **natural
     log**, so the output depends on chunk boundaries.
   - It restored `capacity // batch_size`. This is the "do not merge prefix chunks" rule in the patch.
7. **Validation.**
   - `LAYER_SMOKE_OK`.
   - Golden replay: 15.05 → 8.35, 11.51 → 2.56 and 37.90 → 13.18 ms.
   - The re-profile showed the prefix FMHA, fixup and kv_b GEMMs gone, replaced by a 69 µs MLA decode kernel.
   - An nsys check found no host syncs.

**The idea was reproducible.** Round 1, which was VOID because no GPU was available, and two aborted round-2 sessions
reached the same design independently, before any profile came back.

### 7.2 Round 4: causal pass restricted to prefill rows (−2.3%)

- **nsys showed the waste:** *"causal FMHA 976 µs runs over all 16,448 rows (65 sequences) although the 64 decode rows'
  causal output is discarded"*.
- **The fix:** `causal_num_reqs` slices the causal pass. It is bit-exact, 8.56 → 8.36 ms.
- **Ranking ignored.** VibeSim's ranking put the merged-front GEMM first and the causal pass fifth. The agent skipped
  the GEMM as a closed cuBLAS kernel.
- **Unfinished experiment shipped.** A later alt-stream experiment (−0.65%) was cut off by the time limit, and its
  `K3_AB` knob was left in the tree. The stand-alone port removed it.

### 7.3 Two caveats about where the idea came from

1. **The task prompt named the mechanism.** The workload text in `issue_k3_mla_mixed_claude.json`, which we wrote
   before round 1, says the decode rows *"ride along as 1-token extends and, for MLA, add prefix chunks to the
   chunked-prefix attention"*. That is a strong hint. The agent still had to find the code, design the split, debug the
   LSE convention and validate it, but the "where" was given.
2. **VibeSim's "before" model already assumed the fix.** The oracle's baked prediction modelled the decode rows of a
   mixed batch as absorbed decode, with prefix attention at 0
   (`context_k3/repo/logs/k3_mla_mixed/raw/cost_log/worker_predict_0.parquet`).
   - As a result, its "before" (24.2 ms over the 3 points) matched the measured **after** (24.1 ms), not the measured
     before (64.5 ms).
   - The agent wrote "R0 24.2 ms" next to its measured 64.5 ms and never reconciled the two.
   - Its leaf list for the node (`q_absorb`, `mla_decode_attention`, `v_up`) effectively encoded the target design,
     and the hypothesis cites it: *"That is what the VibeSim leaf list for this node expects."* That is the arch recipe
     leaking the answer, not a roofline finding.
   - The node's R0/R5 headroom came from a different request's legitimate prefix, so the actual waste (~6.7 ms) was
     invisible to the model.

**Verdict on VibeSim.**
- **Round 2: marginal to helpful, as a hint and not as a measurement.**
  - The prompt and the code already pointed at the mechanism, and three other sessions found it the same way.
  - The profiler, not VibeSim, measured the cost.
  - VibeSim's one useful signal (the absorbed-decode leaves) came from a model that silently assumed the optimized
    dataflow. Its headroom numbers described a better program than the one that ran.
- **Round 4: marginal.** nsys found the discarded causal work. VibeSim ranked other nodes higher.

(`vibesim_usage_audit.py` reports only "2 calls" for these trials. It misses calls made through a `$B/<verb>` URL
variable; the transcript shows ~12 in round 2.)
