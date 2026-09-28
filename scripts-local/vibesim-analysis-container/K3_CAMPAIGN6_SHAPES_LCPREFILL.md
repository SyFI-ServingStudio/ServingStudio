# Kimi-K3 loop — Campaign 6: shape-matched decode + long-context chunked prefill, slurm mode

*2026-09-26 22:24 – 2026-09-27 07:30 · agent = Claude Opus 5.5 (Bedrock) · branch `kimi-k3-loop` · companion to
`K3_PIPELINE_AND_RESULTS.md` (this file is self-contained; the main write-up's working copy was left untouched, see §7).*

**Sign convention:** Δ is the change in step latency, **negative = faster**. Every number below is the judge's
5-repetition median on a fresh `lmsysorg/sglang:v0.5.20` container, correctness against pristine-tree goldens.

## 1. What the user asked for

1. **Same input shapes for both layers.** MLA had been judged at 1×1M but KDA had not (and vice versa for 32×8k /
   1×8k). Both layers now run one identical decode point set.
2. **Long-context chunked prefill**: 16k and 32k chunks on contexts of 128k up to 262k tokens, for MLA and KDA.
3. **Claude as the agent, GPU work only through slurm `main` jobs**, without holding a GPU for the whole campaign.

## 2. Slurm mode (new; `K3_GPU_MODE=slurm`)

The agent container gets **no GPU**. Its `/tmp/kimi_single_layer_decode.py` is `k3_gpu_shim.py`, which writes a
request file; the host broker `k3_gpu_broker.py` turns each request into **one `sbatch` job** (`slurm_gpu.sh` →
`k3_slurm_step.sh --docker …`): a fresh `before_image` container bound to the slurm-allocated GPU, the agent's
edited tree mounted at `/sgl-workspace/sglang/python/sglang`, `/workspace/opt_run` shared, a per-case sglang JIT
cache persisted. Output streams back into the agent's terminal with the driver's exit code (`[step] EXIT_RC=`).
`/tmp/gpu_run.sh <cmd>` is the generic variant (the agents used it for `nsys`, micro-benchmarks and A/B sweeps).
The judge runs as one sbatch job too. `k3_slurm_step.sh` waits (≤5 min) if the allocated GPU has a non-slurm tenant.

Measured overhead: a driver call round-trips in ~40–75 s (queue + container start + JIT), `gpu_run.sh nvidia-smi`
in 9 s. A 45-minute Claude round issued 17–31 GPU jobs; GPU occupancy per round ≈ 15–20 min of 45.

Files: `run_iter_opt_eval.sh` (slurm branch, broker lifecycle, `slurm_run`), `k3_gpu_shim.py`, `k3_gpu_broker.py`,
`k3_slurm_step.sh`, `gpu_run.sh`, `gpu_note_slurm.md` (prompt insert `$GPU_NOTE`), `run_k3_continuous.sh` (`slurm`
GPU arg), `run_k3_shapes_slurm.sh`, `run_k3_pair_slurm.sh` (generic pair campaign with transfer-checked seeding),
`reseed_shapes_case.sh`, `recheck_best_trees_seeds.sh` (`slurm` arg).

## 3. Shape-matched decode campaign (`k3_{kda,mla}_shapes_claude`)

Points (both layers): **1×1M (primary)**, 16×64k (`mix` per-request lengths for MLA; uniform for KDA, whose state
makes it length-independent), 128×8k, 32×8k, 1×8k. Driver flags add `--bf16-gemm-init --flashinfer-autotune`
(production fidelity). Oracles: 8801 (KDA) / 8802 (MLA). Seed: the B=512 Claude best tree of the same layer,
transfer-checked on the new points (fallback chain → decode best tree → pristine).

**Seed transfer (b512 Claude best tree vs pristine, every point correct):**

| layer | 1×1M | 16×64k | 128×8k | 32×8k | 1×8k |
|---|---|---|---|---|---|
| KDA (pristine µs) | 136.6 | 200.0 | 403.0 | 261.6 | 136.6 |
| KDA seed Δ | −7.4% | −8.1% | −6.6% | −7.0% | −6.0% |
| MLA (pristine µs) | 304.5 | 292.3 | 493.0 | 263.6 | 146.9 |
| MLA seed Δ | −20.8% | −10.5% | −6.2% | −6.5% | −11.2% |

KDA costs the same at 1×1M and 1×8k (recurrent state, no KV read). The MLA b512 levers (fp8 KV-concat satfinite
kernel, persistent-decode tail split, out-gate overlap) also pay at B=1: 241.1 µs beats the Codex decode best (253.5).

**Rounds** (45-min agent budget, judged vs the tree the round started from; primary = 1×1M):

| round | verdict | primary | other points | change |
|---|---|---|---|---|
| KDA 1 | PASS | 124.3 → 109.9 (−11.5%) | 16×64k −6.8%, 32×8k −5.4%, 1×8k −13.0%, 128×8k −0.5% | split the merged MoE front at m≤32: routed rows first, gate-up + shared experts on the alt stream, latent tail before the join; fp32 TGV halves (bit-exact at B=1/16) |
| KDA 2 | PASS | 109.9 → 103.8 (−5.6%) | 16×64k −3.5%, 32×8k −2.0%, 128×8k flat, 1×8k +0.9% (noise floor) | new `l2_prefetch` JIT kernel: pulls o_proj + routed-front weights into L2 on the alt stream during the KDA recurrence (32 CTAs) |
| KDA 3 | PASS | 103.8 → 101.8 (−2.0%) | 16×64k −2.4%, 1×8k −7.2%, 32×8k −0.9%, 128×8k flat | attn-res direct kernel for T≤32 rows (fused_tma.cuh / attn_res.py); the T≤128 variant was dropped by the agent (rel 0.019) |
| MLA 1 | PASS | 241.1 → 216.6 (−10.2%) | all four flat (≤0.2%) | B=1 KV split k=8 + LSE merge for L≥64k (256 MB MLA workspace); fused fp8 KV-concat writes the split q replicas |
| MLA 2 | PASS | 218.4 → 216.4 (−0.9%) | 1×8k −1.6%, 32×8k +0.8% (noise) | attn-res `cluster_small` kernel + fused MoE finalize+RMSNorm |
| MLA 3 | null | 217.4 → 218.3 (+0.4%) | flat | custom small-m MXFP4 MoE kernel — correct, not faster; MLA 1×1M plateau ≈216 µs |

**Cumulative vs pristine** (seed rechecks under seeds 1 and 2 with fresh goldens: all 20 points PASS):

| layer | point | pristine µs | final µs | Δ | seed-1 / seed-2 Δ |
|---|---|---|---|---|---|
| KDA | 1×1M | 136.6 | 101.8 | **−25.5%** | −21.7% / −21.7% |
| KDA | 1×8k | 136.6 | 105.7 | −22.6% | −22.6% / −24.0% |
| KDA | 32×8k | 261.6 | 226.5 | −13.4% | −14.7% / −14.2% (rel 0.0198, closest to the 0.02 tolerance) |
| KDA | 16×64k | 200.0 | 163.2 | −18.4% | −16.1% / −16.3% |
| KDA | 128×8k | 403.0 | 374.2 | −7.1% | −7.2% / −7.2% |
| MLA | 1×1M | 304.5 | 216.4 | **−28.9%** | −28.5% / −28.4% |
| MLA | 1×8k | 146.9 | 128.4 | −12.6% | −12.6% / −12.8% |
| MLA | 32×8k | 263.6 | 247.2 | −6.2% | −5.7% / −5.9% |
| MLA | 16×64k mix | 292.3 | 261.4 | −10.6% | −11.1% / −11.8% |
| MLA | 128×8k | 493.0 | 462.5 | −6.2% | −6.4% / −6.3% |

## 4. Long-context chunked prefill (`k3_{kda,mla}_lcprefill_claude`)

Points (both layers, 1 request, `B,L,pf<prefix>`): **16k chunk completing a 262,144-token prompt (prefix 245,760,
primary)**, 32k chunk completing 262,144 (prefix 229,376), 16k and 32k chunks at prefix 131,072. Eager step time;
row-wise prefill CHECK. Oracles: the existing prefill oracles 8805/8806 (baked for ≤48k prefixes — see §6). Seed:
the earlier prefill Claude best tree, transfer-checked.

**Seed transfer (prefill Claude best tree vs pristine, all bit-exact):**

| layer | 16k @ 245,760 | 32k @ 229,376 | 16k @ 131,072 | 32k @ 131,072 |
|---|---|---|---|---|
| KDA pristine ms | 9.03 | 17.94 | 9.20 | 18.07 |
| KDA seed Δ | −4.2% | −4.8% | −6.4% | −4.6% |
| MLA pristine ms | 27.69 | 48.36 | 18.37 | 35.09 |
| MLA seed Δ | −9.0% | −4.9% | −7.5% | −4.7% |

KDA is linear in chunk length and flat in prefix (the prefix enters only via the carried-in state); MLA grows with the
prefix (prefix-chunk attention, 66% of the pristine step at 245k).

**Rounds:**

| round | verdict | primary | other points | change |
|---|---|---|---|---|
| KDA 1 | PASS | 8.61 → 8.47 ms (−1.6%) | −0.6% / −3.1% / −2.1% (bit-exact) | CUDA `mma.sync` h-scan kernel (`kda_chunk_h.cuh`, ported from the earlier campaign's unaccepted round via the warm-start KB) + conv1d BLOCK_M=16 / 2 warps |
| KDA 2 | PASS | 8.31 → 8.19 ms (−1.4%) | −1.9% / −4.0% / −1.8% (rel 0.007) | fused SiLU-and-mul JIT kernel for the MoE activation (`situ_and_mul.cuh`); the agent's side-stream in-proj overlap was rejected by its own A/B (≤0.3%, persistent GEMMs serialize) |
| KDA 3 | FAIL (3σ) | 8.33 → 8.27 ms (+0.7%) | −2.4% / −3.6% / −1.6% | correct but below the floor on the primary; no iteration log |
| MLA 1 | PASS | 25.20 → 22.00 ms (−12.7%) | −7.5% / −7.4% / −4.4% (bit-exact) | **16-byte alignment of the prefix-chunk `cum_seqlen_k` row** so the CuTe-DSL prefix FMHA accepted in the earlier prefill campaign actually engages at long prefixes (it had silently fallen back to trtllm-gen) + empirical KV-split policy (S=4 up to 6 waves, else S=2) |
| MLA 2 | PASS | 22.00 → 20.97 ms (−4.7%) | −5.7% / −2.1% / −3.1% (bit-exact) | prefix-FMHA softmax: 40 of every 128 exp2 per tile evaluated as a degree-2 polynomial on the FMA pipe (MUFU relief; was 20 with degree 3) |
| MLA 3 | FAIL (correctness) | +1.4% | 32k @ 229k rel 0.30, row-wise rule failed | a numerics-changing prefix-attention variant; rejected |

**Cumulative vs pristine (primary):** KDA 9.03 → 8.19 ms (**−9.3%**); MLA 27.69 → 20.97 ms (**−24.3%**). Other
points: KDA 32k@229k 17.94 → 16.31 (−9.1%), 16k@128k 9.20 → 8.20 (−10.9%), 32k@128k 18.07 → 16.32 (−9.7%);
MLA 32k@229k 48.36 → 39.89 (−17.5%), 16k@128k 18.37 → 15.39 (−16.2%), 32k@128k 35.09 → 30.90 (−11.9%).
Seed rechecks (seeds 1/2) of both lcprefill trees: slurm job 1880, results appended in §8 when done.

## 5. Patches (absolute paths; each chain validated to reproduce the case's best tree exactly)

Base: `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang/claude/`.
Apply in order inside `python/sglang` of `lmsysorg/sglang:v0.5.20` with `patch -p1`, on top of the seed tree named.

**KDA shape-matched (seed = `iter_opt_eval_k3_kda_b512_claude/best_tree`):**
- `kda_shapes_01_r1_split_moe_front_small_m_alt_stream_tgv_fp32.patch` — `srt/models/kimi_k3.py`, `srt/layers/attn_residual.py`,
  `kernels/ops/kimi_k3/attn_res.py`, `kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh` (264 lines). *Why it works:* at m≤32 the
  merged front GEMM ran serially ahead of a latency-bound routed-MoE chain (41 µs on the critical path); issuing the routed rows
  first and the gate-up/shared GEMMs on the alt stream hides them behind the MoE, and the fp32 TGV halves keep routing exact.
- `kda_shapes_02_r2_l2_prefetch_oproj_routed_front.patch` — `kernels/jit/csrc/elementwise/l2_prefetch.cuh`,
  `kernels/ops/elementwise/l2_prefetch.py`, `srt/models/kimi_k3.py` (138 lines). *Why:* at B=1 the o_proj and routed-front weight
  reads are bandwidth-bound and follow the KDA recurrence, which leaves HBM idle; a 32-CTA prefetch on the alt stream warms L2 so
  the GEMMs hit cache (VibeSim: both nodes at R0/R5 1.3–1.7, i.e. bandwidth-bound, not compute-bound).
- `kda_shapes_03_r3_fused_decode_attnres_tuning.patch` — `kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh`,
  `kernels/ops/kimi_k3/attn_res.py` (228 lines). *Why:* a direct (non-TMA) attn-res kernel for tiny token counts (T≤32) removes
  the TMA-ring setup that dominated the kernel at B=1/16; the T≤128 variant would have raised rel error to 0.019 and was dropped.

**MLA shape-matched (seed = `iter_opt_eval_k3_mla_b512_claude/best_tree`):**
- `mla_shapes_01_r1_b1_kv_split8_lse_merge_kvconcat_q_replicas.patch` — `srt/layers/attention/trtllm_mla_backend.py`,
  `kernels/ops/attention/mla_decode_tail_split.py`, `kernels/ops/attention/set_mla_kv_concat_q.py`,
  `kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh` (266 lines). *Why:* at B=1 the MLA decode kernel runs in a
  min-latency single-CTA-per-head mode and reads 1M fp8 KV tokens with 12 CTAs; splitting the KV into 8 chunks (LSE merge)
  puts 96 CTAs on the read and the fused concat kernel writes the 8 q replicas in the same pass, so the split adds no launch.
- `mla_shapes_02_r2_attnres_cluster_small_moe_finalize_rmsnorm.patch` — `kernels/jit/csrc/kimi_k3/attn_res/cluster_small.cuh`,
  `kernels/ops/kimi_k3/attn_res.py`, `kernels/ops/moe/moe_finalize_rmsnorm.py`, `srt/models/kimi_k3.py` (425 lines). *Why:*
  two launch-bound tails at B=1 (attn-res, MoE finalize + RMSNorm) fused into one kernel each; small (−0.9%) but above 3σ.

**KDA long-context prefill (seed = `iter_opt_eval_k3_kda_prefill_claude/best_tree`):**
- `kda_lcprefill_01_r1.patch` — `kernels/jit/csrc/attention/kda_chunk_h.cuh`, `kernels/ops/attention/kda_chunk_h.py`,
  `kernels/ops/attention/fla/chunk_delta_h.py`, `kernels/ops/attention/fla/kda.py`, `kernels/ops/mamba/causal_conv1d_triton.py`,
  `srt/layers/attention/linear/kernels/kda_triton.py`, `srt/layers/attention/linear/kernels/kda_ptx.py` (743 lines). *Why:* the
  Triton chunk-state scan (`chunk_delta_h`) is a serial recurrence over chunks; the CUDA `mma.sync` version halves it (516→281 µs
  in the earlier campaign) — below the 3σ floor there, above it here because the long-context judge's σ is smaller.
- `kda_lcprefill_02_r2.patch` — `kernels/jit/csrc/kimi_k3/situ_and_mul.cuh`, `srt/models/kimi_k3.py` (78 lines). *Why:* SiLU and
  the gate multiply were two elementwise passes over 32–64k rows of MoE activations; one fused pass.

**MLA long-context prefill (seed = `iter_opt_eval_k3_mla_prefill_claude/best_tree`):**
- `mla_lcprefill_01_r1.patch` — `srt/layers/attention/trtllm_mla_backend.py`, `kernels/ops/attention/cute_dsl_fmha_fp8.py`
  (149 lines). *Why:* the CuTe-DSL prefix FMHA requires 16-byte-aligned cumulative sequence lengths; at prefixes that are not a
  multiple of the prefix-chunk size the second row was misaligned and the backend silently used the slower trtllm-gen ragged
  kernel for 66% of the step. Aligning the row re-enables the fast path; the KV-split policy then balances waves at 32k chunks.
- `mla_lcprefill_02_r2.patch` — `kernels/ops/attention/cute_dsl_fmha_fp8.py` (231 lines). *Why:* the prefix FMHA softmax is
  MUFU-bound (exp2 per score); evaluating 40 of every 128 exp2 with a degree-2 polynomial on the otherwise idle FMA pipe
  rebalances the two pipes without changing the fp8-quantized P (bit-exact vs golden).

Unchanged from earlier campaigns: `patches/sglang/{mla,kda}/…`, `patches/sglang/claude/{mla,kda}_{b512,prefill}_*` and the
`patches/harness/*`, `patches/vibesim/*` bundles; `patches/README.md` lists them.

## 6. Was VibeSim used? (trajectory evidence, `vibesim_usage_audit.py`)

Every accepted round fetched the oracle's prediction and ran `optimality`/`analyze` before its first edit (the agents saved
the raw JSON as `vs_*.json` in their iteration folders; the audit's call counter undercounts Claude because it fetches
through small Python scripts). What the analysis contributed:
- **KDA shapes r1–r3:** `optimality?scope=iter` ranked `mxfp4_fused_moe` (R0/R5 43.6, closed cubin) first and `merged_front`
  (1.32) / `o_proj` (1.3–1.7) as bandwidth-bound nodes; the agents skipped the cubin and attacked the front/o_proj critical
  path (r1) and their HBM reads (r2 prefetch) — node-level guidance that named the edited nodes.
- **MLA shapes r1:** top node `mla_decode_attention` "HVPerCta256 min-latency mode at bs=1" → the KV-split change directly.
- **KDA lcprefill r1–r2:** `kda_chunk_prefill` R0 2.89 ms vs R5 76 µs, "no cached alternative", `run_summary` hardware_gap 84.7% →
  the h-scan kernel port; r2's agent read the same numbers and went for elementwise fusion after its overlap idea failed A/B.
- **MLA lcprefill r1–r2:** the oracle's prediction covers ≤48k prefixes; the agent wrote "VibeSim does not model the 245k prefix,
  so I use the measured profile" and ranked nodes from `--profile-kernels` (prefix FMHA 66%). Guidance here was the run_summary
  (`hardware_gap 81.8%`, attention subtree R0/R5 6.4) plus the measured table. **Open item:** bake a long-context prefill oracle
  (presets with 128k/262k prefixes → JIT rows → `build_context_k3.sh`) so the prefix-attention share is predicted, not measured.

## 7. Harness lessons from this campaign

- **Read-only FlashInfer cache made a false FAIL.** With `--flashinfer-autotune`, the tuner persists newly tuned shapes into
  `k3_flashinfer_autotune_cache.json`; the judge mounted that cache read-only, so the first MLA 1×1M baseline died with EROFS and
  both MLA seed candidates "failed". Fixed: judge mounts the cache rw (as the agents always did). KDA passed only because its shapes
  were already cached — a fidelity flag can turn into an infrastructure dependency.
- **A full `/raid` (100% for ~20 min, ~900 GB written by another tenant) killed the broker** on a log write and stranded one agent
  call. Fixed: the broker requeues on I/O errors; the stranded request was replayed by hand. Disk is now a monitored line in the checks.
- **Stray GPU job at agent timeout:** a request filed seconds before the 45-min cut became a job after the smoke. Fixed: queued
  requests are dropped at timeout.
- **Path rewriting in the shim** must only touch path-valued flags: the first version rewrote `--hidden-scale 1.0` into a path.
- **`sbatch` needs absolute script paths** for the step script (the first recheck submission failed on `exec: not found`).
- **The main write-up's working copy differs from HEAD** (380 lines removed incl. the container list and Campaign 5, tables
  reformatted) — not my edit; left untouched. This file is separate so nothing of yours is overwritten. Say which version should
  carry Campaign 6 and I will merge it.

## 8. Seed rechecks of the long-context prefill trees (slurm job 1880, fresh pristine goldens per seed)

All 16 points PASS (4 points × 2 seeds × 2 layers); MLA bit-exact on every point, KDA max rel 0.0074.

| layer | point | seed 1: pristine → best (Δ) | seed 2: pristine → best (Δ) |
|---|---|---|---|
| KDA | 16k @ 245,760 | 8.91 → 8.16 ms (−8.4%) | 8.90 → 8.21 ms (−7.7%) |
| KDA | 32k @ 229,376 | 17.68 → 16.36 (−7.4%) | 17.65 → 16.31 (−7.6%) |
| KDA | 16k @ 131,072 | 9.12 → 8.23 (−9.8%) | 9.03 → 8.18 (−9.4%) |
| KDA | 32k @ 131,072 | 17.63 → 16.34 (−7.3%) | 17.72 → 16.28 (−8.1%) |
| MLA | 16k @ 245,760 | 27.51 → 20.76 (−24.5%) | 27.49 → 20.67 (−24.8%) |
| MLA | 32k @ 229,376 | 47.95 → 39.97 (−16.6%) | 47.87 → 39.95 (−16.6%) |
| MLA | 16k @ 131,072 | 18.18 → 15.11 (−16.9%) | 18.16 → 15.07 (−17.1%) |
| MLA | 32k @ 131,072 | 34.68 → 30.38 (−12.4%) | 34.62 → 30.50 (−11.9%) |

Verdict files: `iter_opt_eval_k3_{kda,mla}_lcprefill_claude/best_tree_seed{1,2}_verdict.json`; shape-matched trees:
`iter_opt_eval_k3_{kda,mla}_shapes_claude/best_tree_seed{1,2}_verdict.json` (§3).

## 9. Follow-ups (user 2026-09-27 13:45: "lets do 1-4 one by one")

### 9.1 Long-context prefill oracle + MLA rounds 4–6
- **Oracle.** New VibeSim presets `predict_kimi_k3_b200_rank1_layer_{kda,mla}_lcprefill` (cases = the four lcprefill points) were
  predicted and baked as oracles 8807 (KDA) / 8808 (MLA). Comparing the first bake with the measured kernel tables showed the
  **prefill model itself was wrong in composition**: the MLA prefix-attention cost did not scale with prefix length (9.4 ms at
  both 128k and 245k; measured 8.9 / 16.7 ms), the merged-front GEMM row was 5.6 ms vs 2.3 measured, the MXFP4 MoE row 2.9 vs
  1.2 ms, and the KDA step predicted 16.9 vs 9.0 ms. The totals had happened to land near the measured MLA step at 245k, which
  hid it (and the earlier "predictions within 3%" claim for the ≤48k prefill oracles was wrong: they predicted 16.5 / 18.3 ms
  against 9.07 / 12.12 measured).
- **Codex B12c** (`codex_tasks/b12c_k3_lcprefill_fidelity.md`, VibeSim commit `427d75f8` on `kimi-k3-arch`, fast-forwarded to
  `kimi-k3`): the MLA prefix runner profiles one FlashInfer launch per prefix chunk with `prefix_len`/`num_prefix_chunks` in
  the row key (rows 9.38 ms @16k/128k, 17.63 @16k/245k, 31.36 @32k/229k); the fp32-output and bf16 GEMM runners use the sglang
  cuBLAS path; the MXFP4 MoE prefill row was fixed; decode rows untouched. All 14 prefill cases now within ±5%
  (`doc/alignment/kimi_k3_b12c_prefill.md` in the VibeSim repo), prefix attention ranked first for every long-context MLA point
  (67% at 245k vs 66% measured). GPU work for the fix went through slurm jobs only. Image re-baked (`rebake_prefill_oracles_b12c.sh`),
  oracles 8805–8808 restarted (8808 between the MLA rounds so the running agent was not interrupted).
- **MLA lcprefill rounds 4–6** (round 4 on the first bake, 5 on it too, 6 on the corrected oracle):

| round | verdict | primary 16k @ 245k | other points | change |
|---|---|---|---|---|
| 4 | null | 20.93 → 20.86 ms (−0.4%) | flat | agent spent its budget on one hypothesis |
| 5 | PASS | 20.84 → 20.61 (−1.1%) | 16k@128k −0.7%, 32k flat (bit-exact) | fp8 packing of the prefix-chunk `kv_b_proj` output fused into the GEMM epilogue (`mla_kv_b_proj_pack_fp8`, `dense_gemm_sm100_fp8_via_bf16_epilogue`); a single-launch prefix FMHA was tried and reverted (neutral) |
| 6 | PASS | 20.79 → 20.45 (−1.6%) | 16k@128k −0.8%, 32k@128k −0.6% (bit-exact) | causal in-chunk attention pass issued on the attention alt stream, overlapping the prefix passes; join before `merge_state` |

  MLA cumulative on the primary vs pristine: **27.69 → 20.45 ms (−26.1%)**; seed rechecks 1/2 of the final tree: all 8 points
  bit-exact (−25.4% / −25.7% on the primary). Patches: `<base>/claude/mla_lcprefill_03_r5.patch` (4 files, 1289 lines),
  `<base>/claude/mla_lcprefill_04_r6.patch` (4 files, 319 lines); chain r1→r2→r5→r6 validated.

### 9.2 Near-miss re-judging at 15 reps (`rejudge_near_misses.sh`)
| round | 5-rep judge | 15-rep re-judge | outcome |
|---|---|---|---|
| KDA B=512 r2 (fused-decode prologue hoist + quad-row reduction) | +1.4% vs 3.65% floor (σ 7.3 µs) | 599.5 → 593.5 µs @512 (−1.0%), −1.1% @256, −0.5% @128; σ 0.3 µs; rel 0.017 | **accepted**, promoted; KDA @512 cumulative 675.2 → 593.5 (−12.1%); patch `<base>/claude/kda_b512_02_r2_fused_decode_prologue_hoist_quad_row_reduction.patch` |
| KDA lcprefill r3 | +0.7% vs 3σ | −0.9% vs 1.31% floor (prefill σ ≈0.4% is intrinsic) | not accepted |
| KDA lcprefill r5 (see 9.3) | −1.2% vs 3σ | −1.1% vs 3.09% floor (σ 85 µs, noisy window) | not accepted |

### 9.3 KDA long-context prefill rounds 4–6 (60-min agent budget, corrected oracle 8807)
| round | verdict | primary 16k @ 245k | other points | change |
|---|---|---|---|---|
| 4 | PASS | 8.26 → 8.13 ms (−1.6%) | 32k@229k −1.6%, 16k@128k −3.5%, 32k@128k −1.9% (rel 0.007) | warp-specialized 4-stage `kda_chunk_h` (taken from the rejected r3 tree) + pinned BK32 / 1-warp config for the Triton inter-chunk solve |
| 5 | FAIL (3σ) | 8.02 → 7.93 (−1.2%) | −1.4% / −3.2% / −1.4% (rel 0.017) | consistent but sub-floor even at 15 reps |
| 6 | FAIL (3σ) | 8.24 → 8.15 (−1.2%) | −1.6% / −3.0% / −2.5% | new fused conv+l2norm and chunk-intra CUDA kernels; passes the row-wise rule but max_rel 0.30 — a numerics-changing rewrite |

  KDA cumulative on the primary vs pristine: **9.03 → 8.13 ms (−10.0%)**. Patch `<base>/claude/kda_lcprefill_03_r4.patch`
  (7 files, 853 lines); chain r1→r2→r4 validated. Seed rechecks of the final tree: slurm job 2101 (results in 9.5).

### 9.4 Mixed prefill+decode and speculative verify (driver extension, commit `a8452d8`)
Survey and field-by-field spec in `K3_MIXED_VERIFY_DRIVER_SPEC.md`. The driver gained two step kinds:
- `B,L,mx<C>[p<P>]` — **MIXED**: B decode requests at context L (one token each) + one prefill chunk of C tokens on a P-token
  prefix, built as the scheduler does (prefill rows first, decode rows as 1-token extends with prefix L−1; the eager runner
  runs MIXED as EXTEND). `KDAMixedState` (slot 0 = chunk, slots 1..B = decode; per-request initial state) and `MLAMixedState`
  (KV pool with the chunk range + B decode ranges; `MHA_CHUNKED_KV` with the decode rows counted in `prefix_chunk_len =
  capacity // batch_size`). Eager metric; row-wise correctness rule over all C+B rows + post-step state.
- `B,L,vk<k>` — **TARGET_VERIFY**: B requests × (k+1) tokens, chain (topk=1). `KDAVerifyState` adds the speculative scratch
  (`intermediate_ssm` [slots, D, HV, K, V] fp32, `intermediate_conv_window` [slots, D, 3, dim]) the non-fused verify path writes
  (`causal_conv1d_update` + Triton `target_verify` with `disable_state_update`, `lower_bound` gate); `MLAVerifyState` gives each
  request L + page_size slots and runs the absorbed decode kernel with q_len = k+1 (`trtllm_mla_backend.forward_extend` verify
  branch). The speculative config (`speculative_algorithm=EAGLE`, `speculative_num_draft_tokens=k+1`, `speculative_eagle_topk=1`,
  `speculative_attention_mode=prefill`) is published through `RuntimeContext.override` before any backend is built. Strict
  max-rel rule; eager metric (production graph-captures verify; both sides of a comparison are eager here).

Smoke on the pristine tree (slurm jobs 2102–2105): all four kinds run and **replay bit-exact** against their goldens.
Reference eager steps at B=8 @8k: KDA mixed (4k chunk) 2.84 ms, KDA verify ×4 1.58 ms, MLA mixed (4k chunk @ prefix 4k)
3.50 ms, MLA verify ×4 1.47 ms.

Cases: `issue_k3_{kda,mla}_mixed_claude.json` (primary 64×8k + 16k first chunk; 128×8k + 4k chunk; 64×8k + 16k chunk @ prefix
49,152; oracles 8809/8810 = new VibeSim mixed-batch predictions, e.g. MLA 64×8k+16k predicted 8.1 ms) and
`issue_k3_{kda,mla}_verify_claude.json` (primary 64×4 @8k; 128×4 @8k; 16×4 @64k; oracle = the decode prediction at the same
B/L, since VibeSim has no verify branch — stated in the prompt). Seeds: mixed ← lcprefill best trees, verify ← shape-matched best
trees, transfer-checked. Campaigns launched 18:51 (`run_k3_pair_slurm.sh mixed_claude 1 3`, `… verify_claude 1 3`); results
appended in §9.6 when they land. **Blocker at launch:** GPUs 0/5/6/7 were taken by `slurm-manager` placeholder jobs
(`gpumgmt-ph-cayenne-gpu*`, partition `placeholder`, 24 h limit, PreemptMode=OFF) at 18:46, the other four by a 3-day job, so
the transfer checks sat in PENDING (Resources) until a GPU was released.

### 9.6 Mixed / verify campaign results

**GPU starvation (2026-09-27 20:40 → 2026-09-28 01:39).** The user's 4-GPU job plus four `slurm-manager` placeholder
jobs (`gpumgmt-ph-cayenne-gpu*`, partition `placeholder`, priority tier 10) held all eight GPUs for five hours; every
measurement request of the running agents queued. Consequences, all recorded in the cases' `rounds.jsonl`:
- KDA mixed r1: first verdict "non-compiling" was false (the rebuild smoke timed out after 90 min in the queue); the
  re-judge showed the agent had edited blind and made the tree +52–77% slower → FAIL.
- MLA mixed r1 (judged 01:46, slurm 2150): FAIL null, +0.04% — the agent tree is byte-identical to the seed; it never
  got past `iter_00`. The round is a no-op, so MLA mixed runs rounds 2–4.
- Both campaigns were paused (runner loops killed, in-flight judges kept) and resumed at 01:55 with
  `finish_and_resume.sh` → `resume_mixed_verify.sh` once the placeholders released the GPUs.

**Harness fidelity bug: the eager verify metric is host-bound.** KDA verify r1 measured −21%/−16%/−22% at 5 reps and
−14.7%/−12.8%/−10.7% at 15 reps (slurm 2152), all points CHECK-exact, and was rejected twice: the pristine baseline's
σ was 92 µs on a 1259 µs step (3σ floor 22%); r2 (−21.7%/−21.9%/−20.5% eager, exact) hit σ = 418 µs → a 99% floor.
Cause: the eager TARGET_VERIFY step is ~65% CPU launch overhead (KDA 64×8k vk3: 1420 µs eager vs 502 µs as a CUDA
graph; MLA 1523 vs 493), so its wall time follows host contention, not the kernels. Production graph-captures verify
(`DecodeCudaGraphRunner` at the captured draft width). Fix (commit `fa5067a`): driver flag `--verify-graph` —
`graph_capture_and_time` takes `num_tokens = B·(k+1)` so the backends' `init_cuda_graph_state(max_bs, max_num_tokens)`
sees the draft width; the judged `latency_us` becomes the graph replay. GPU smoke on the pristine image (slurm 2162,
`k3_vkgraph_dev/out/test.slurm.out`): graph replay is bit-exact against the eager step on both layers (golden captured
in graph mode, replayed eagerly: max rel err 0.0, state ok), and 5 graph-mode reps span 501.2–502.1 µs (KDA) /
492.9–493.0 µs (MLA) — σ ≈ 0.05 µs instead of 90–400 µs. Both verify cases carry the flag from KDA r3 / MLA r1 on
(the KDA r2 judge had already snapshotted the old driver; the swap waited for it — `k3_vkgraph_dev/swap_vkgraph_driver.sh`).
The KDA r1 and r2 trees are re-judged in graph mode (slurm 2199, `rejudge_vkgraph_trial_{1,2}_verdict.json`).

**KDA verify (`k3_kda_verify_claude`, seed = the shapes tree; graph metric, 5 reps, pristine goldens).**

| round | metric | primary 64×8k vk3 | 128×8k vk3 | 16×64k vk3 | verdict |
|---|---|---|---|---|---|
| seed (shapes tree) vs pristine | eager | −12.0% | −10.7% | +3.6% | transfer check only |
| r1 | eager 5 / 15 reps | −21.1% / −14.7% | −16.3% / −12.8% | −22.4% / −10.7% | FAIL (3σ floor 29% / 22%) |
| r1 tree re-judged | graph | −0.05% | 0.0% | 0.0% | null — the eager gain was host time only |
| r2 | eager | −21.7% | −21.9% | −20.5% | FAIL (σ 418 µs → 99% floor) |
| r2 tree re-judged | graph | 490.0 → 486.9 µs (−0.62%) | −0.15% | −1.06% | PASS, superseded by r3 (separate branch) |
| r3 | graph | 490.0 → **472.5 µs (−3.55%)** | 651.7 → 620.1 (−4.85%) | 298.5 → 292.3 (−2.08%) | **PASS**, promoted |

r3 changed 14 files incl. a new `kda_verify_recurrent.{cuh,py}` JIT kernel, `kda_backend.py`, `kda_triton.py`,
`kda_fused_decode`, `attn_res`, `l2_prefetch`. Best tree: `iter_opt_eval_k3_kda_verify_claude/best_tree`
(= `trial_3_tree_judged`).

**MLA verify (`k3_mla_verify_claude`, from PRISTINE — every accepted MLA tree regressed the verify step (§9.4); graph
metric, 5 reps).**

| round | primary 64×8k vk3 | 128×8k vk3 | 16×64k vk3 | verdict |
|---|---|---|---|---|
| r1 | 492.9 → **472.4 µs (−4.15%)** | 658.7 → 626.1 (−4.96%) | 425.3 → 400.6 (−5.80%) | **PASS**, promoted (2 files: `set_mla_kv_concat_q.cuh`, `kimi_k3.py`) |
| r2 | 483.7 → 484.6 (−0.18%) | −0.01% | −0.04% | FAIL null (variants of the same KV-concat kernel) |
| r3 | 483.7 → **473.6 µs (−2.09%)** | 628.1 → 626.1 (−0.31%) | 399.6 → 396.6 (−0.76%) | **PASS**, promoted (+ `attn_res` fused TMA path, 5 files) |

(Each round's "before" is the judge's own re-measurement of the then-best tree in the same slurm job — the 472.4 → 483.7
shift between r1's after and r2/r3's before is the between-job spread of the graph replay on different GPUs, ~2%;
comparisons within a round are same-job.)

**Seed rechecks of the final verify trees (slurm 2339, fresh pristine goldens per seed, graph metric) — all PASS:**

| tree | seed | 64×8k vk3 | 128×8k vk3 | 16×64k vk3 | max rel err |
|---|---|---|---|---|---|
| KDA verify best | 1 | 509.3 → 480.6 µs (−5.6%) | 653.8 → 599.4 (−8.3%) | 322.9 → 302.4 (−6.3%) | 0.017 |
| KDA verify best | 2 | 503.2 → 474.5 (−5.7%) | 662.8 → 609.5 (−8.1%) | 339.3 → 318.8 (−6.0%) | 0.013 |
| MLA verify best | 1 | 497.0 → 466.4 (−6.2%) | 650.6 → 620.2 (−4.7%) | 417.1 → 388.4 (−6.9%) | 0.0 |
| MLA verify best | 2 | 513.3 → 482.7 (−6.0%) | 670.7 → 639.4 (−4.7%) | 423.3 → 394.9 (−6.7%) | 0.0 |

KDA's cumulative gain includes the shapes-tree seed (its bf16-state fused decode kernel gives the 0.01–0.017 rel err, well
inside the 0.02 tolerance); MLA's is r1 + r3 on pristine and bit-exact.

**KDA mixed (`k3_kda_mixed_claude`, seed = the KDA lcprefill tree, transfer check −13.4/−18.2/−12.7% vs pristine; eager
metric — sglang runs mixed batches eagerly; 5 reps).**

| round | primary 64 dec @8k + 16k chunk | 128 dec @8k + 4k chunk | 64 dec + 16k chunk @ 48k prefix | verdict |
|---|---|---|---|---|
| r1 | +76% | +52% | +77% | FAIL (edited blind while GPU-starved) |
| r2 | 8659 → **8361 µs (−3.44%)** | 2766 → 2764 (−0.08%) | 8775 → 8382 (−4.48%) | **PASS**, promoted (incremental patch vs the lcprefill seed: 4 files, 70 lines — the one-wave guard on the CUDA `kda_chunk_h` scan lifted for mixed batches) |
| r3 (two voided starts: Bedrock 503) | 8302 → 8368 (−0.8%) | 2695 → 2720 (−0.9%) | 8565 → 8429 (+1.6%) | FAIL null (agent lost Bedrock after 133 turns at iter_01) |

**MLA mixed (`k3_mla_mixed_claude`, seed = the MLA lcprefill tree, transfer check −29.0/−54.4/−17.0% vs pristine; eager
metric; 5 reps).** Two starts of r2 were voided (09:13 Bedrock 503 storm; 12:20 all 8 GPUs held) before it ran 12:53–13:32.

| round | primary 64 dec @8k + 16k chunk | 128 dec @8k + 4k chunk | 64 dec + 16k chunk @ 48k prefix | verdict |
|---|---|---|---|---|
| r1 | +0.04% | 0.0% | 0.0% | VOID (tree identical to the seed; GPU-starved) |
| r2 | 15065 → **8248 µs (−45.2%)** | 11512 → 2590 (−77.5%) | 37898 → 13166 (−65.3%) | **PASS**, promoted (3 files, 224 lines: `trtllm_mla_backend.py`, `forward_batch_deepseek_mha_mixin.py`, `forward_mha.py`) |

r2's lever: in a MIXED (EXTEND) batch the decode rows — 1-token extends with an 8k prefix each — went through the chunked-prefix
MHA path (prefix_chunk_len = capacity // batch_size, so 64–128 tiny MHA prefix passes per layer); the agent routes them to
the absorbed MLA decode kernel (trtllm-gen) and keeps the chunked-prefix MHA only for the prefill request. CHECK: max rel
0.013, 0 of 16,448 rows over tolerance, state ok. A variant merging the prefix into one big chunk was faster still
(11095 µs @ p49152) but FAILED CHECK — log2-LSE vs ln `merge_state` makes the output chunk-dependent — and was reverted.
**Production relevance (checked in the pristine v0.5.20 source):** `handle_attention_trtllm_mla` in
`srt/models/deepseek_common/attention_backend_handler.py` returns `MHA_CHUNKED_KV` for every `is_extend_without_speculative()`
batch (unless chunked-prefix caching is disabled), with no per-row split — so in a real server a mixed chunk's decode rows
take the same chunked-prefix MHA path the harness's seed took. The lever applies whenever the deployment forms mixed
prefill+decode batches (sglang `--enable-mixed-chunk`); it is a no-op for pure-decode and pure-prefill batches.

| r3 | 8441 → 8361 (−0.95%, below the 1.7% 3σ floor) | 2659 → 2773 (+4.3%) | 13376 → 13269 (−0.8%) | FAIL (secondary regression) |

| r4 | 8483 → **8290 µs (−2.28%)** | 2518 → 2481 (−1.47%) | 13348 → 13081 (−2.00%) | **PASS**, promoted (3 files, 131 lines: the causal in-chunk attention pass restricted to the prefill rows of a mixed batch — the decode rows already go to the absorbed decode kernel; max rel err 0.0) |

r3's own lever (MoE tail add3 → in-place `addmm_` accumulation into the attention-residual prefix sum, −1.2% eager)
was exact but below the noise floor and regressed the 128-decode point; a CuTe causal variant failed CHECK.

**Follow-up #4 summary (all vs the round's baseline tree, judge-measured, CHECK pass on every point):**

| case | seed | accepted rounds | primary point | cumulative vs pristine (seed rechecks) |
|---|---|---|---|---|
| KDA verify (graph metric) | shapes tree | r3 | 64×8k vk3: 490.0 → 472.5 µs (−3.55%) | −5.6% / −5.7% (seeds 1/2) |
| MLA verify (graph metric) | pristine | r1, r3 | 64×8k vk3: 492.9 → 472.4 → 473.6* µs (−4.15%, −2.09%) | −6.2% / −6.0% |
| KDA mixed (eager) | lcprefill tree | r2 | 64 dec + 16k chunk @8k: 8659 → 8361 µs (−3.44%) | (slurm 2467, pending) |
| MLA mixed (eager) | lcprefill tree | r2, r4 | 64 dec + 16k chunk @8k: 15065 → 8248 → 8290* µs (−45.2%, −2.28%) | (slurm 2467, pending) |

\* each round's "before" is re-measured in its own judge job; the between-job spread is ~2% (verify) / ~3% (mixed).
Patch chains under `patches/sglang/claude/{kda,mla}_{verify,mixed}_*.patch`, each validated to reproduce its best tree.
Rounds lost to infrastructure (recorded as VOID in `rounds.jsonl`, artifacts kept): KDA mixed r3 ×2 + MLA mixed r2 ×3
(Bedrock Opus 5.5 503 storms 07:00–11:45; all eight GPUs held by placeholders + the user's job 12:20–12:47).

### 9.5 Seed rechecks of the final KDA lcprefill tree (slurm job 2101, fresh pristine goldens per seed)
All 8 points PASS, max rel 0.0074.

| point | seed 1: pristine → best (Δ) | seed 2: pristine → best (Δ) |
|---|---|---|
| 16k @ 245,760 | 8.91 → 8.30 ms (−6.9%) | 8.90 → 8.18 ms (−8.1%) |
| 32k @ 229,376 | 17.68 → 16.44 (−7.0%) | 17.65 → 16.30 (−7.6%) |
| 16k @ 131,072 | 9.12 → 8.34 (−8.5%) | 9.03 → 8.27 (−8.4%) |
| 32k @ 131,072 | 17.63 → 16.47 (−6.6%) | 17.72 → 16.32 (−7.9%) |

Final state of the long-context prefill trees after follow-ups #1 and #3: MLA 27.69 → 20.45 ms (−26.1%), KDA 9.03 → 8.13 ms
(−10.0%) on the primary; both seed-verified.

`<base>` = `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang`.
