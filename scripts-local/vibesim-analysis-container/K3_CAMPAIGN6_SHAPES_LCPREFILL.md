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
