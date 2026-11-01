# Kimi-K3 decoder layer on B200 — optimization report (all campaigns, as of 2026-10-11)

*One document for everything the K3 optimization loop produced: what was measured, every case's cumulative result,
the techniques that worked, what is in the accepted trees that must not ship, and a plan for upstream sglang PRs.
Companion documents (unchanged): `K3_TECHNIQUES_ALL_SHAPES.md` (technique × shape matrix T1–T40, dead ends, transfer
rules), `K3_CAMPAIGN6_SHAPES_LCPREFILL.md` (campaign 6 + follow-ups), `K3_PIPELINE_AND_RESULTS.md` (pipeline,
campaigns 1–5), `K3_HANDOFF_VIBEOPTIMIZER.md` (loop code map).*

**Conventions.** Δ = change in step latency, **negative = faster**. Decode and verify points are CUDA-graph replay µs;
prefill and mixed points are eager-step µs. "Pristine" = unmodified sglang v0.5.20. Point tags: `pf<N>` = prefill
chunk at prefix N, `mix` = mixed per-request context lengths, `mx<N>` = mixed batch with an N-token prefill chunk,
`vk3` = speculative verify with 3 draft + 1 bonus tokens.

---

## Where the code lives (added 2026-10-20)

Everything below is **local to this node**; nothing has been pushed to a remote.

**1. Merged sglang tree with every accepted lever ("ALL" in §6.6)**

| | |
|---|---|
| path | `/raid/yilegu/k3_merge` (a git repo whose root is the `python/sglang` package of sglang v0.5.20, not a full clone; no remote) |
| branch / commit | `all` @ `c8d52d4` — 51 files, +8234/−449 vs `pristine` (`313d169`, stock v0.5.20) |
| how to run | bind-mount the directory over `python/sglang` inside `lmsysorg/sglang:v0.5.20` (see §6.6, `k3_e2e/k3_e2e_bench.sbatch`) |
| not in `all` | MLA verify levers; MLA b512 r5, MLA prefill r4/r6, KDA prefill r4 |
| other branches | per-lineage: `mla_all`, `kda_all`, `{mla,kda}_decode`, `{mla,kda}_shapes`, `{mla,kda}_mixed`, `kda_verify`; PR candidates: `prA_mixed`, `prA_mixed_on_pristine`, `prBp_on_F`, `prC_b1_kvsplit`, `prE_kda_prefill`, `prE2_no_conv`, `prF_kvpack`, `ladder_F`; ablations: `cand_*` |

`all` still contains the temporary env knobs and debug logging listed in §5, so it is a benchmark tree, not a
shippable branch. To share it, push `/raid/yilegu/k3_merge` to a fork.

**2. Patch files, one per accepted round** — `scripts-local/vibesim-analysis-container/patches/sglang/`
(layout in `patches/README.md`; apply with `patch -p1` inside `python/sglang` of v0.5.20)

- `mla/01..06`, `kda/01..02`: the original decode levers in acceptance order; `mla_best_tree_vs_pristine.patch` and
  `kda_best_tree_vs_pristine.patch` are the cumulative equivalents.
- `claude/*.patch` (36 files): the Claude Opus 5.5 campaign chains — `{kda,mla}_{b512,prefill,lcprefill,shapes,verify,mixed}_<nn>_r<k>`.
  Each chain is validated by `export_k3_patches.sh` to reproduce its case's best tree exactly (regenerated 2026-10-11).
  `UNACCEPTED_*` = exact but below the judge's 3σ floor.
- `pr_candidates/PR_{A,B,Bprime,C,E,F,G}_*.{patch,md}`: the curated upstream candidates of §6 and the second wave, each
  with its write-up.

Per-round trees, judge outputs and agent transcripts are in `k3_opt_history/` (see §7).

---

## 0. Summary

- **Scope.** One Kimi-K3 decoder layer (KDA linear-attention layer and MLA full-attention layer), one B200 rank of the
  cookbook TP8/EP8 recipe (12 attention heads, 112 local MXFP4 experts, fp8-e4m3 KV, bf16 KDA state), sglang v0.5.20,
  12 workload cases (6 workloads × 2 layer types). An agent (Codex gpt-5.6-luna, then Claude Opus 5.5) edits a real
  sglang checkout; a judge re-measures each round against the tree it started from and checks numerics against the
  pristine tree's outputs.
- **Result.** 93 judged rounds in the loop histories, 42 accepted (the one-off decode trials 4–19 before the
  continuous loop are not counted). Every case's current best tree beats pristine on every point, with the
  same numerics tolerance (max rel error ≤ 0.0185 on all points, limit 0.02), re-measured under a fresh seed:

  | workload (primary point) | KDA pristine → best | MLA pristine → best |
  |---|---|---|
  | decode 1 × 1M | 132.6 → 97.9 µs (**−26.2%**) | 304.6 → 202.2 µs (**−33.6%**) |
  | decode 512 × 8k | 708.0 → 584.2 µs (−17.5%)¹ | 990.0 → 868.0 µs (−12.3%) |
  | prefill 16k chunk | 9026 → 8235 µs (−8.8%) | 12113 → 10669 µs @ 48k prefix (−11.9%) |
  | long-context prefill 16k @ 245k | 8907 → 8296 µs (−6.9%) | 27508 → 20442 µs (**−25.7%**) |
  | speculative verify 64 × 4 @ 8k | 509.3 → 476.6 µs (−6.4%) | 497.0 → 466.4 µs (−6.2%) |
  | mixed: 64 decodes + 16k chunk | 10106 → 8355 µs (−17.3%) | 21239 → 8353 µs (**−60.7%**) |

  ¹ recheck of the r1 tree; the r2 lever (+1.0%) was accepted later at 15 reps and is not in this number.
- **Largest single lever:** routing the decode rows of a mixed prefill+decode batch to the absorbed MLA decode kernel
  (−45…−77% in its own round). It is a dispatch problem in sglang's shared DeepSeek-family MLA code, not a kernel
  problem, and it is still present on upstream main.
- **Status of the loop:** plateaued. The last balanced run (2026-10-04/05, one round per case, KDA and MLA on
  identical workloads) accepted only 2 of 12 rounds (KDA prefill −5.2%, KDA verify −0.8%).
- **Upstream PRs: yes, but as a small curated series, not a port of the trees.** Upstream main is ~1600 commits past
  v0.5.20, `kimi_k3.py` was refactored, two of our changes already exist upstream (one merged, one in an open PR),
  and the accepted trees contain temporary env knobs and debug logging. Recommended first PRs: the MLA mixed-batch
  dispatch fix, the bf16-state fused KDA decode kernel, and the CUDA KDA prefill scan kernel (§6).

## 1. What was measured, and what that does and does not prove

| aspect | setup |
|---|---|
| model | one `KimiK3DecoderLayer` built with seeded synthetic weights (unit-scale inputs, norm gains init fixed so routing is not collapsed), real sglang attention backends, MoE and kernels |
| deployment shape | one rank of TP8/EP8 on B200: 12 heads, 112 local experts with local top-2 (an emulation of the 896-expert/top-16 EP8 share), `trtllm_mla`/`cutedsl_mla`, `flashinfer_mxfp4`, fp8-e4m3 KV, bf16 KDA state, FlashInfer autotune warmup on |
| metric | decode/verify: median CUDA-graph replay; prefill/mixed: median eager step; 5 reps in one slurm job for both trees |
| acceptance | primary point gain ≥ max(3σ, 0.5%), no secondary point regressed beyond noise, numerics vs pristine goldens on every point (max rel ≤ 0.02 decode/verify; row-wise rule for prefill/mixed) |
| hardening | seed rechecks (new weights/state/inputs, fresh goldens) of every best tree vs pristine; 15-rep re-judges of near misses |

**What it proves:** the accepted changes make this layer faster on B200 at these shapes, with outputs within bf16/fp8
tolerance of stock sglang. **What it does not prove:** end-to-end serving gains (no full-model run; there is no K3
checkpoint on this node), multi-rank behavior (all-reduce/all-to-all, DP attention, real expert popularity), other GPUs,
or upstream main (all numbers are against v0.5.20). Every PR needs its own real-model accuracy and throughput check.

## 2. Cumulative results per case (pristine → current best tree)

Seed-1 rechecks against pristine. Six cases whose best tree changed after 2026-09-28 were re-run on 2026-10-11
(jobs 6045–6050, preemptible); the other six are the 2026-09-25…28 rechecks (their best trees have not changed since).

| case | point | pristine µs | best µs | Δ | max rel |
|---|---|---|---|---|---|
| **KDA shapes** | 1×1M (primary) | 132.6 | 97.9 | **−26.2%** | 0.0000 |
| | 16×64k | 210.4 | 177.7 | −15.6% | 0.0137 |
| | 128×8k | 398.8 | 369.2 | −7.4% | 0.0088 |
| | 32×8k | 251.4 | 214.6 | −14.6% | 0.0185 |
| | 1×8k | 136.7 | 101.9 | −25.4% | 0.0000 |
| **MLA shapes** | 1×1M (primary) | 304.6 | 202.2 | **−33.6%** | 0.0058 |
| | 16×64k mix | 291.2 | 249.4 | −14.3% | 0.0039 |
| | 128×8k | 466.4 | 435.7 | −6.6% | 0.0084 |
| | 32×8k | 284.1 | 257.6 | −9.3% | 0.0089 |
| | 1×8k | 146.9 | 120.2 | −18.2% | 0.0058 |
| **KDA b512** (r1 tree) | 512×8k (primary) | 708.0 | 584.2 | −17.5% | 0.0158 |
| | 256×8k | 507.2 | 474.5 | −6.5% | 0.0128 |
| | 128×8k | 398.7 | 372.1 | −6.7% | 0.0088 |
| **MLA b512** | 512×8k (primary) | 990.0 | 868.0 | −12.3% | 0.0104 |
| | 256×8k | 653.7 | 589.4 | −9.8% | 0.0117 |
| | 16×64k mix | 292.4 | 261.6 | −10.5% | 0.0039 |
| | 128×8k | 466.4 | 435.6 | −6.6% | 0.0084 |
| **KDA prefill** | 16k (primary) | 9025.9 | 8235.3 | −8.8% | 0.0071 |
| | 16k @ 48k | 9296.0 | 8304.0 | −10.7% | 0.0071 |
| | 4×4k | 8980.6 | 8373.6 | −6.8% | 0.0071 |
| **MLA prefill** | 16k @ 48k (primary) | 12113.2 | 10668.7 | −11.9% | 0.0141 |
| | 16k | 8505.4 | 8137.8 | −4.3% | 0.0070 |
| | 4×4k | 8100.8 | 7668.8 | −5.3% | 0.0071 |
| **KDA long-ctx prefill** | 16k @ 245k (primary) | 8906.9 | 8295.7 | −6.9% | 0.0071 |
| | 32k @ 229k | 17677.3 | 16442.4 | −7.0% | 0.0074 |
| | 16k @ 131k | 9119.8 | 8344.9 | −8.5% | 0.0071 |
| | 32k @ 131k | 17633.4 | 16473.3 | −6.6% | 0.0074 |
| **MLA long-ctx prefill** | 16k @ 245k (primary) | 27508.2 | 20441.5 | **−25.7%** | 0.0000 |
| | 32k @ 229k | 47945.1 | 39950.8 | −16.7% | 0.0000 |
| | 16k @ 131k | 18177.5 | 15014.2 | −17.4% | 0.0000 |
| | 32k @ 131k | 34682.2 | 30406.1 | −12.3% | 0.0000 |
| **KDA mixed** | 64 dec + 16k chunk (primary) | 10106.4 | 8355.1 | −17.3% | 0.0072 |
| | 128 dec + 4k chunk | 3376.6 | 2759.9 | −18.3% | 0.0071 |
| | 64 dec + 16k chunk @ 48k | 10042.8 | 8416.6 | −16.2% | 0.0072 |
| **MLA mixed** | 64 dec + 16k chunk (primary) | 21239.3 | 8353.1 | **−60.7%** | 0.0113 |
| | 128 dec + 4k chunk | 24737.3 | 2541.9 | **−89.7%** | 0.0097 |
| | 64 dec + 16k chunk @ 48k | 44626.3 | 13057.5 | **−70.7%** | 0.0111 |
| **KDA verify** | 64×4 @ 8k (primary) | 509.3 | 476.6 | −6.4% | 0.0148 |
| | 128×4 @ 8k | 653.8 | 601.6 | −8.0% | 0.0170 |
| | 16×4 @ 64k | 322.9 | 292.4 | −9.4% | 0.0122 |
| **MLA verify** | 64×4 @ 8k (primary) | 497.0 | 466.4 | −6.2% | 0.0000 |
| | 128×4 @ 8k | 650.6 | 620.2 | −4.7% | 0.0000 |
| | 16×4 @ 64k | 417.1 | 388.4 | −6.9% | 0.0000 |

Notes:
- Between-job spread of the same tree is ~2% (graph) / ~3% (eager); only same-job comparisons are exact. That is why
  pristine values differ slightly between cases measuring the same point.
- These numbers supersede the cumulative table in `K3_TECHNIQUES_ALL_SHAPES.md` §2 for the six re-run cases (that
  table was frozen on 2026-09-28; e.g. MLA 1×1M moved from −28.9% to −33.6% with the 2026-09-30 rounds).
- The cases are separate trees. The decode, prefill, verify and mixed trees have not been merged into one tree; the
  MLA decode levers are known to *regress* verify (+5.8…+31%), so a merged tree needs q_len-gating first (§5).

## 3. Round history

Per case, in order; P = accepted, F = rejected; the number is the judged change of the primary point vs the round's
start tree (positive = faster); VOID rounds (no information: GPU starvation, Bedrock outage, docker restart) are omitted.

| campaign | case | rounds |
|---|---|---|
| 2–3 (Codex, 09-23/24) | KDA decode | r20F r21F r22P+1.3 r23F r24F r25F |
| | MLA decode | r20P+0.8 r21P+1.5 r22F r23P+1.5 r24P+2.4 r25P+0.8 r26F r27F |
| 4 (Codex, 09-24/25) | KDA b512 | r1F r2F r3F |
| | MLA b512 | r1F r2F r3F r4F |
| 5 (Claude, 09-25/27) | KDA b512 | r1P+11.1 r2P+1.0 r3F · *(10-04/05)* r4F r5F |
| | MLA b512 | r1P+7.3 r2P+1.8 r3P+1.1 · r4F r5P+0.6 r6F r7F r8F |
| | KDA prefill | r1P+5.6 r2F r3F · r4P+5.2 |
| | MLA prefill | r1P+4.6 r2P+2.7 r3F · r4P+4.3 r5F r6P+1.5 r7F |
| 6 (Claude, 09-26/28) | KDA shapes | r1P+11.5 r2P+5.6 r3P+2.0 · r4P+2.3 r5P+1.4 r6F r7F r8F |
| | MLA shapes | r1P+10.2 r2P+0.9 r3F · r4P+4.7 r5P+2.0 r6F r7F |
| | KDA long-ctx prefill | r1P+1.6 r2P+1.4 r3F r4P+1.6 r5F r6F · r7F |
| | MLA long-ctx prefill | r1P+12.7 r2P+4.7 r3F r4F r5P+1.1 r6P+1.6 · r7F r8F r9F |
| | KDA verify | r1F r2P+0.6 r3P+3.5 · r4P+0.8 |
| | MLA verify | r1P+4.2 r2F r3P+2.1 · r4F |
| | KDA mixed | r1F r2P+3.4 r3F · r4F |
| | MLA mixed | r1F r2P+45.2 r3F r4P+2.3 · r5F (+1.9, 15-rep re-judge also FAIL vs a 2.3% noise floor) |

"·" separates campaign 7 (2026-09-29…10-05: continued rounds on the non-plateaued cases, then one balanced round per
case). Campaign 7 accepted 9 rounds: KDA shapes r4/r5, MLA shapes r4/r5, MLA b512 r5, MLA prefill r4/r6, KDA prefill
r4, KDA verify r4.

## 4. Techniques that worked

### 4.1 Catalogue T1–T40 (campaigns 2–6)

Full descriptions, the per-shape matrix and patch paths are in `K3_TECHNIQUES_ALL_SHAPES.md` §3–§4. Grouped:

| group | techniques | biggest effect |
|---|---|---|
| MLA attention kernel choice and scheduling | T1 trtllm-gen decode, T2 persistent schedule, T3 tail split, T4 B=1 KV split ×8 + LSE merge, T5 output gate in decode tail | T1 −12.1% @1×1M, T4 −10.2% @1×1M |
| small-m GEMM dispatch (m ≤ 16…128) | T6/T7 CuTe TGV for fp32-output front / latent_up / shared_down, T8 CuTe bf16 shared down, T9 split merged MoE front (routed first, shared on alt stream) | T9 −11.5% KDA @1×1M |
| critical-path overlap, cache warming | T10 shared experts ⟂ routed MoE, T11 bfa overlap limit, T12 L2 prefetch of o_proj/routed-front weights | T12 −5.6% KDA @1×1M |
| KDA recurrent decode | T13 packed fast path for lower-bounded gate, T14 bf16-state fused decode kernel, T15 vectorized ld/st, T16 num_warps | carried at every batch size |
| KDA chunked prefill | T25 strided chunk kernels + no host sync, T26/T34 CUDA mma.sync h-scan, T27 conv1d tuning, T28 fused SiLU-and-mul | T25 −5.6% |
| elementwise / fusion | T17/T18/T20 attn-res kernels, T19 satfinite fp8 KV concat, T21 fused fp8 K/V pack+quantize | T21 −4.6% prefill |
| MLA prefix attention | T22 CuTe-DSL fp8 FMHA, T23 16-byte cu_seqlen alignment + KV-split policy, T24 exp2 on the FMA pipe, T32 fp8 pack in kv_b epilogue, T33 causal pass on alt stream | T23 −12.7% @245k |
| verify / mixed | T35 CUDA KDA verify recurrence, T36/T37 MLA verify overlap, T38 KDA h-scan guard for mixed batches, **T39 MLA mixed-batch decode rows → absorbed decode**, T40 causal pass on prefill rows only | **T39 −45…−77%** |
| harness only (do not port) | T29 route+quant JIT for 112/top-2, T30 one-shot MoE autotune | — |

### 4.2 New in campaign 7 (T41–T49)

| # | case/round | technique | judged (primary, vs start tree) | numerics | patch (`patches/sglang/claude/`) |
|---|---|---|---|---|---|
| T41 | KDA shapes r4 | 4-CTA cluster "V-split" of the fused KDA decode kernel for B ≤ 3 (DSMEM norm reduction, PDL early loads); prefetch trimmed at ≤3 tokens | 1×1M −2.3%, 1×8k −3.1% | bit-exact | `kda_shapes_04_r4.patch` (+348/−18) |
| T42 | KDA shapes r5 | pinned TGV tactic (56 CTAs) for the shared-down GEMM at m ≤ 8 so the overlapped routed chain keeps SMs | 1×1M −1.4% | bit-exact | `kda_shapes_05_r5.patch` (+6) |
| T43 | MLA shapes r4 | T9 ported to MLA: router + latent-down rows as their own fp32 TGV GEMM on the main stream, shared gate_up/SiTU/down on the alt stream, latent_up before the join | 1×1M −4.7%, 32×8k −4.9%, 1×8k −7.9% | bit-exact | `mla_shapes_03_r4.patch` (+90/−19) |
| T44 | MLA shapes r5 | one Triton kernel for KV-split merge + v_up bmm + sigmoid output gate at B=1, L ≥ 64k; split factor 8 → 12 | 1×1M −2.0% (1×8k +1.6%, within the regression allowance) | bit-exact | `mla_shapes_04_r5.patch` (+192/−3) |
| T45 | MLA b512 r5 | MoE tail 3-way add folded into the shared_down / latent_up GEMM epilogues (beta = 1); SM carveout 32 on shared_down | 512×8k −0.6%, 256×8k −1.5% | rel 0.008–0.009 | `mla_b512_04_r5.patch` (+50/−22) |
| T46 | MLA prefill r4 | port of the long-context prefill levers (T23 KV split + alignment, T24 exp2 rebalance, T32 fp8 kv_b pack, T33 causal overlap) + host `seq_lens_cpu` (no D2H sync) + 4-items/thread SiTU | 16k @ 48k −4.3% | rel 0.0117 @48k | `mla_prefill_03_r4.patch` (+1904/−73) |
| T47 | MLA prefill r6 | output-gate GEMM issued behind the attention launches (fills their tail wave) + beta = 1 tail fold for prefill | 16k @ 48k −1.5% | gate exact; fold rel 0.007 | `mla_prefill_04_r6.patch` (+88/−3) |
| T48 | KDA prefill r4 | port of T34 CUDA h-scan (477 → 205 µs) + new fused conv1d+SiLU+q/k L2 norm (162 → 104 µs) + pinned inter-solve config + approximate SiTU + beta = 1 fold | 16k −5.2%, 16k @ 48k −4.8% | rel 0.0074 | `kda_prefill_02_r4.patch` (+1342/−32) |
| T49 | KDA verify r4 | split MoE front extended to ≤ 64 tokens (cuBLAS fp32-out halves above the TGV range); strided gate in `fla/fused_norm_gate` (no copy) | 64×4 −0.8%, 16×4 @ 64k −3.5% | rel 0 (agent replay) | `kda_verify_02_r4.patch` (+77/−7) |

Most campaign-7 gains are at the edge of the 3σ floor (T41, T42, T45, T47, T49), and 8 of 9 agents ended on the
time limit, so several diffs contain edits made after the agent's last notes. Treat T41–T49 as candidates to
re-verify, not established levers.

## 5. What is in the accepted trees that must not ship

The trees passed the judge, but they are research artefacts. Before any port:

| item | where | why it must go |
|---|---|---|
| env knobs left by agents: `K3_KVSPLIT` (`# TEMP_KNOB`), `K3X_FOLD`/`K3X_GATE` (`# K3X_TMP`), `K3_PTX`/`K3_PTX_SUFFIX`/`K3_PTX_ARCH` (`# TMP_AB`) | MLA shapes r4/r5, MLA prefill r6, KDA prefill r4 | undocumented runtime switches |
| `logger.warning("DGDBG …")` on every in-proj call + an unvalidated `deep_gemm` bf16 in-proj path | KDA prefill r4 | debug output; path never smoke-tested |
| approximate SiTU (`tanh.approx`, ~2⁻¹¹) in the shared `situ_and_mul` kernel | KDA prefill r4 | changes numerics for every caller |
| `torch._C._set_sm_carveout_experimental(32)` | MLA b512 r5 | private experimental API, magic constant |
| exact-shape tactic `(n, k) == (7168, 6144), m <= 8` inside a generic GEMM heuristic | KDA shapes r5 | K3 TP8 dimensions hard-coded in shared code |
| route+quant JIT for 112 experts / top-2 (T29), one-shot autotune (T30) | MLA decode r20, b512 r1 | harness emulation / harness gap |
| `is_var_seq=False` hard-coded (T2) | MLA decode r25 | overfit to the harness's uniform lengths; must be derived from the batch |
| small-m dispatch thresholds (≤16, ≤32, ≤64), KV-split factors (8, 12), exp2 count 40, B ≤ 3 cluster cutoff | many | tuned on one rank, 12 heads, 148 SMs, B200 |
| decode levers that assume one query row per request | MLA shapes/decode trees | regress verify by +5.8…+31%; gate on `q_len` |
| `layer._defer_kv_split_merge` / `_pending_kv_split` attributes | MLA shapes r5 | backend ↔ model coupling through mutable layer state |

## 6. Upstream sglang PRs — recommendation and plan

### 6.1 Reality check (sglang main checked 2026-10-11, commit `0d0be8fc98`)

- **Distance.** v0.5.20 (2026-09-18) is a release branch; main is ~1600 commits past its fork point and v0.5.21 is out.
  Under `python/sglang` 2727 files changed. Only 4 of our standalone patches still apply cleanly to main
  (mla/01, mla/03, mla/05, kda/02). `kimi_k3.py` was reworked (stage-boundary refactor; shared experts now TP-sharded),
  so every overlap/dispatch hunk in `kimi_k3.py` must be re-derived, not rebased.
- **Already upstream or in flight:**
  - fp8 KV-concat satfinite (T19) — **merged** as #42659 (ours adds a NaN fixup on top; at most a small follow-up).
  - KDA packed decode for lower-bounded gates (T13) — **open #41570** contains the same hunk; main now gates it on
    `supports_packed_decode_safe_gate` (#42857), which only the Helion kernel sets. Do not send a duplicate.
  - fused prefix K/V pack + fp8 quant (T21) — overlaps **open #41716**; host q/kv lengths — **open #43272**;
    chunked-prefix LSE base — **open #40927 / issue #40903** (our split-KV merge uses log2-LSE; must agree with it).
  - B=1 MLA KV split — related **open #42535** (opt-in cute-dsl split-KV); MLA decode backend default — **open #34437**
    argues the opposite direction for q > 1.
  - bf16 KDA state — upstream is moving there (open #41714, #43657, #41570) but the fused decode kernel is still
    fp32-only on main (`kda_fused_decode.py` `covered()`).
  - Kimi-K3 KDA MTP verify — **merged #42857** (Helion) on 2026-10-11; our CUDA verify kernel (T35) must be compared
    against it before proposing anything.
- **Still open problem upstream:** mixed-batch MLA dispatch (T39). `handle_attention_trtllm_mla` on main is identical
  to v0.5.20. The closest precedent is open #43238 (same idea for the MHA backend, 3201 → 3777 tok/s), and merged
  #43461 makes `--enable-mixed-chunk` more usable, so the fix gets more valuable over time.
- **sglang requirements** (`.github/pull_request_template.md`, `docs/docs/developer_guide/contribution_guide.mdx`):
  motivation, accuracy test (`sgl-eval run gsm8k` minimum), speed test and profile; tests under `test/registered/`
  with `CustomTestCase` + `register_*_ci`; `pre-commit run --all-files`; no mixins (use `msgspec.Struct`), cache
  per-layer checks in `__init__`, minimize host syncs; JIT kernels in-tree; CI needs the `run-ci` label from a
  maintainer in `CI_PERMISSIONS.json`; approval by CODEOWNERS + merge oncall.

### 6.2 Recommendation

Do it, as **3–4 focused PRs written by a person on top of main**, using the agent's patches as references, not as
code to submit. Rank by upstream value × confidence ÷ effort:

| # | PR | contents (our techniques) | value | risk / effort | validation needed |
|---|---|---|---|---|---|
| **1** | **MLA: route decode rows of mixed (EXTEND) batches to the absorbed decode kernel** | T39 + T40 (`mla_mixed_01_r2`, `mla_mixed_02_r4`, 355 lines) | very high: −45…−77% layer step with `--enable-mixed-chunk`; affects every DeepSeek-family MLA model on `trtllm_mla`, not only K3 | medium: touches shared `deepseek_common` dispatch and `forward_batch_deepseek_mha_mixin.py` (reworked upstream; reviewers dislike new mixin fields → `msgspec.Struct`) | e2e with `--enable-mixed-chunk` on a real MLA model (DeepSeek-V2-Chat is on this node; K3 if a checkpoint is available), gsm8k, a mixed-batch unit test; cite #43238 |
| **2** | **KDA: bf16-state support in the fused decode kernel** | T14 + T15 + T31 (`kda/02` applies cleanly to main) | high for any deployment with `--mamba-ssm-dtype bfloat16` (the cookbook recipe): one launch instead of three | low–medium: self-contained kernel + `covered()`; style it after #41714 (fp32 accumulate, stage through storage dtype) | kernel test vs the unfused chain (rel ≤ 0.013 today), K3 e2e decode throughput |
| **3** | **KDA prefill: CUDA chunk h-scan (+ fused conv/L2 norm as a follow-up)** | T26/T34/T38 + the conv+L2 fusion from T48 (bit-exact, with Triton fallback) | medium-high: scan 477 → 205 µs per 16k chunk; −2…−5% prefill step | medium: 900+ lines CUDA; needs a review owner for FLA kernels | kernel tests over chunk counts / mixed batches (the guard from T38), prefill e2e |
| 4 | MLA chunked prefill on CuTe-DSL fp8 FMHA, incl. the 16-byte `cu_seq_lens` alignment and KV-split policy | T22/T23/T24/T32 | high for long context (−12…−26% @128–262k) | high: new kernel path (the alignment bug only exists once this path does), overlaps #41716/#43272/#40927 | after #40927 settles the LSE base; long-context e2e |
| 5 | small, independent clean-ups | strided gate in `fla/fused_norm_gate` (T49), strided KDA chunk kernels + drop `int(query_start_loc[-1])` host sync (T25), host `seq_lens_cpu` (T46) | low-medium each | low | unit tests; can go anytime |
| — | coordinate, don't duplicate | T13 → comment on / support #41570 (or one line: set `supports_packed_decode_safe_gate = True` on the Triton KDA kernel, with B200 numbers); T19 NaN fixup → follow-up to #42659; T1 → share benchmarks on #34437 rather than flip the default | | | |
| — | re-derive on main later, after measuring | stream-overlap and small-m dispatch in `kimi_k3.py` (T6–T12, T36, T41–T45, T47) | possibly high at small batch | high: code moved, TP-sharded shared experts, single-rank assumptions, verify regressions | multi-rank e2e |
| ✗ | do not upstream | T29, T30, T42, carveout, approximate SiTU, env knobs, `is_var_seq` hard-code | | | |

### 6.3 Steps (per PR)

1. **Port by hand onto main** in a fork branch; keep only the lever, drop every item in §5.
2. **Re-measure on main.** The single-layer driver binds v0.5.20 internals, so it needs a main-compatible variant
   (or use sglang's own `bench_one_batch` / profiler for the touched op). Same judge rule: before/after in one job.
3. **Real-model check:** accuracy (`sgl-eval run gsm8k`, plus a long-context eval for prefill PRs) and throughput on a
   full deployment. K3 at TP8 needs a K3 checkpoint and 8 B200s held for the run; PR 1 can be validated on
   DeepSeek-V2-Chat as well, because the dispatch path is shared.
4. **Tests + style:** registered unit/kernel tests, pre-commit, PR template with numbers and profiles.
5. **Coordinate** before opening: comment on #41570 / #43238 / #40927 where relevant; find a reviewer from CODEOWNERS
   for `deepseek_common` (PR 1) and the KDA/FLA kernels (PRs 2–3).

Rough effort: PR 1 ≈ 3–5 days including the e2e check, PR 2 ≈ 2–3 days, PR 3 ≈ 4–6 days, PR 5 ≈ 1 day each.

### 6.4 Decisions needed

- Go / no-go on PRs 1–3, and **who authors and submits** them (opening PRs and commenting on upstream issues is public
  and outward-facing; nothing has been posted).
- Access to a **Kimi-K3 checkpoint** and an 8×B200 window for the real-model checks (not available on this node today).
- Whether to first **merge the case trees** into one K3 tree with `q_len` gating (useful as a reference for PR 4 and
  the re-derived overlap work, not required for PRs 1–3).

### 6.5 First wave (picked 2026-10-12): the three largest, upstream-unsolved wins

Selection rule: large measured effect, the problem still exists on main, and the change is not in `kimi_k3.py`
(heavily refactored upstream; main already has its own shared-expert and front overlap, `_sbo_shared_overlap` /
`_ep_front_overlap`, so the T9/T10/T43 overlap levers are not first-wave material). All three are in shared
DeepSeek-family MLA code, so they also help DeepSeek-V2/V3/R1 on B200 with `trtllm_mla`.

| PR | what | measured effect | reference patches (`patches/sglang/claude/`) | port notes |
|---|---|---|---|---|
| **A** | MLA: decode rows of a mixed (EXTEND) batch → absorbed decode kernel; causal in-chunk pass only on the prefill rows | MLA mixed case −60.7% / −89.7% / −70.7% vs pristine; the lever alone −45.2% (r2) + −2.3% (r4) | `mla_mixed_01_r2.patch` (+198/−26), `mla_mixed_02_r4.patch` (+109/−22) — files `srt/layers/attention/trtllm_mla_backend.py`, `srt/model_executor/forward_batch_deepseek_mha_mixin.py`, `srt/models/deepseek_common/attention_forward_methods/forward_mha.py` | does not apply to main (all three files moved); drop the `K3_AB` env knob (`# AB_TMP`) in r4; new batch fields as a `msgspec.Struct`, not mixin attributes; cite #43238; keep the "do not merge prefix chunks" rule (LSE base) |
| **B** | MLA chunked prefill: non-causal prefix pass on CuTe-DSL fp8 FMHA, 16-byte-aligned `prefix_chunk_cu_seq_lens` rows, wave-aware KV-split (S=4/S=2) | long-context prefill −12.7% @245k (lever), −25.7% case total @245k; −2.7% @48k | `mla_prefill_02_r2_prefix_attention_cutedsl_fmha.patch` (+147, creates `kernels/ops/attention/cute_dsl_fmha_fp8.py`), `mla_lcprefill_01_r1.patch` (+146/−3); optional follow-up `mla_lcprefill_02_r2.patch` (exp2 on the FMA pipe, −4.7%) | the alignment fix only matters once this path exists, so ship together; base on flashinfer's public cute-dsl FMHA API (the exp2 follow-up subclasses internals — leave it out of the first PR); agree the split-KV merge with #40927's LSE base; check overlap with #41716 / #43272 |
| **C** | MLA B=1 long-context decode: split the KV ×8 with an LSE merge, write the q replicas from the fused concat kernel; follow-up: one kernel for merge + v_up + output gate | 1×1M −10.2% (r1) + −2.0% (r5) | `mla_shapes_01_r1_b1_kv_split8_lse_merge_kvconcat_q_replicas.patch` (+246/−20), follow-up `mla_shapes_04_r5.patch` (+192/−3); both build on `mla_decode_tail_split.py` from `mla_b512_02_r2_persistent_decode_tail_split.patch` | gate L ≥ 64k, B = 1 (derive from the batch, not constants); remove `K3_KVSPLIT` (`# TEMP_KNOB`); split factor from a heuristic over SMs/heads, not 8/12; r5's layer-attribute hand-off needs a redesign; coordinate with #42535 (opt-in cute-dsl split-KV) |

Quick win to send alongside (small effect, clean): **D** — bf16-state support in the fused KDA decode kernel,
`patches/sglang/kda/02_r22_bf16_state_fused_kda_decode_kernel.patch` (+181/−61; applies cleanly to main; −1.3…−3.1%
per decode step), modelled on #41714. And one coordination item instead of a PR: support #41570 for the KDA
lower-bound fast path (T13).

Order: A first (largest, smallest diff, likely testable on DeepSeek-V2-Chat here — not yet verified that V2-Chat runs on `trtllm_mla`), then B and C in parallel, D anytime.
Before each: port by hand onto main, re-measure before/after in one job, gsm8k + throughput on a real model.

## 6.6 End-to-end serving benchmark (2026-10-13/14, 8×B200)

**Setup.** The real K3 architecture with layers 1–48 of the real layout (12 MLA + 36 KDA, 896 experts top-16, MXFP4
MoE, attention residuals) and **dummy weights** (`--load-format dummy`; there is no K3 checkpoint on the node and the
full 93-layer model, ~1.44 TB in MXFP4, does not fit one 8×B200 node; 48 layers ≈ one pipeline stage of the cookbook
B200 PP2×TP8 recipe). Config dir `/raid/yilegu/models/kimi-k3-L48-dummy` (real `config.json`/tokenizer/code from
`moonshotai/Kimi-K3`, `num_hidden_layers` 93 → 48). Server: `lmsysorg/sglang:v0.5.20`, TP8 attention, EP8 MoE,
`trtllm_mla` (K3's SM100 default), fp8-e4m3 KV, bf16 KDA state, chunked prefill 16k, context 1M; the tree under test
is bind-mounted over the package. Client: `bench_serving --dataset-name random-ids --tokenize-prompt` (exact
lengths), one warmup request. Server + client in one slurm job (`k3_e2e/k3_e2e_bench.sbatch`, partition main, 8 GPUs,
~11 min per run). Performance only: dummy weights cannot measure accuracy.

**Trees.** stock v0.5.20 vs **ALL** = one tree with every accepted KDA and MLA lever (`/raid/yilegu/k3_merge` branch
`all`, commit `c8d52d4`: merge of the MLA shapes + MLA mixed lineages and the KDA shapes + verify + mixed lineages;
where both lineages changed shared code the variant is gated by layer type; MLA verify excluded; MLA b512 r5,
MLA prefill r4/r6 and KDA prefill r4 not included).

**Results** (median of 3 runs each, interleaved; mixed-chunk server 1 run each):

| workload | metric | stock | ALL | Δ |
|---|---|---|---|---|
| 1 req × 1M in → 32 out | TTFT (1M prefill) | 54.5 s | 45.3 s | **−16.8%** |
| | TPOT | 6.13 ms | 5.70 ms | −7.0% |
| 128 × 8k → 256, concurrency 128 | TPOT | 50.7 ms | 47.8 ms | −5.8% |
| | output tok/s | 1262 | 1371 | +8.7% |
| 16 prefills × 16k, concurrency 4 | TTFT | 831 ms | 589 ms | **−29.1%** |
| | input tok/s | 77.2k | 106.0k | **+37.3%** |
| 512 × 8k → 128, concurrency 512 | TPOT | 106.9 ms | 91.5 ms | **−14.4%** |
| | output tok/s | 705 | 848 | **+20.2%** |
| 16 × 64k → 64, concurrency 16 | TTFT | 13.0 s | 11.6 s | **−10.7%** |
| | output tok/s | 42.3 | 46.7 | **+10.4%** |
| `--enable-mixed-chunk`: 96 × 16k → 512 at 2 req/s, concurrency 64 | TPOT | 59.9 ms | 32.2 ms | **−46.3%** |
| | throughput | 744 out tok/s | 1027 | **+38.1%** |
| `--enable-mixed-chunk`: 128 × 8k → 256 | TPOT | 77.7 ms | 49.2 ms | **−36.7%** |
| | output tok/s | 909 | 1352 | **+48.8%** |

Run-to-run spread (stock): ≤ 1% on the 1M TTFT and 16×64k, 3–4% on TPOT at 128/512, up to 20% on the 512×8k TTFT and
the 16k-prefill TTFT (queueing-dominated); the ALL-vs-stock gaps on the ≥10% rows exceed the spread. The mixed-chunk
rows are the T39 dispatch fix at work: with mixed chunks on, stock sglang is *slower* than without them (TPOT 77.7 vs
50.7 ms at 128×8k), and ALL removes that penalty entirely.

**Single first-wave patches on stock (1 run each):** D (bf16 fused KDA decode): TPOT −0.5…−2.8%. B (CuTe-DSL prefix
FMHA + alignment): 1M TTFT −5.3%, but **16k-prefill TTFT +19% and 1M TPOT +8.9%** — B must be investigated before it
is proposed. A and C do not apply cleanly to stock (their patches stack on earlier rounds) and were measured only
inside ALL.

**Correctness of the merged tree (single-layer judge, seed 1, fresh goldens, vs pristine; job 6814 + 6800).** All 8
cases PASS with max rel ≤ 0.0185, and the per-layer gains match the separate best trees (so the merge lost nothing):
KDA 1×1M −26.2%, MLA 1×1M −32.9%, MLA long-context prefill @245k −26.5%, KDA long-context prefill @245k −8.5%,
MLA mixed −61.5% / −89.9% / −71.2%, KDA mixed −17.5%, KDA verify −6.8%, MLA 512×8k −12.3%. (A first attempt with 8
judges running concurrently on one shared tree directory failed 7 of 8 on `__pycache__`-cleanup and JIT-build races;
those verdicts are kept as `merged_ALL_c8d52d4_seed1_verdict.racefail.json`.)

Raw results: `k3_e2e/results/<label>_<job>/` (`*.jsonl` per workload, `server.log`); jobs 6794/6806/6808 (stock),
6797/6807/6809 (ALL), 6810/6811 (mixed chunk), 6795 (B), 6796 (D).

## 7. Artifacts

| what | where (under `scripts-local/vibesim-analysis-container/` unless absolute) |
|---|---|
| per-round patches, validated chains | `patches/sglang/{mla,kda,claude}/*.patch` — regenerated 2026-10-11 by `export_k3_patches.sh`; all 16 chains reproduce their best tree exactly (the old `kda_shapes_01` patch had been diffed against the wrong seed and silently reverted a b512 lever; fixed) |
| best trees | `iter_opt_eval_k3_<case>/best_tree` (untracked) |
| round history, verdicts, agent transcripts | `iter_opt_eval_k3_<case>/rounds.jsonl`, `trial_<k>_verdict.json`, `trial_<k>_agent.log`, `trial_<k>_opt_run/` |
| seed rechecks | `iter_opt_eval_k3_<case>/best_tree_seed1_verdict.json` (2026-10-11 for six cases; older files kept as `*.pre1011.*`) |
| warm-start knowledge base | `k3_opt_history/` (last rebuilt 2026-09-28; does not yet contain campaign 7) |
| upstream check clone | `/tmp/sglang_upstream_check` (main `0d0be8fc98`), `/tmp/sglang_upstream_check_v0520` |
| loop handoff | VibeOptimizer branch `kimi-k3-loop`, folder `kimi-k3-loop/` |
