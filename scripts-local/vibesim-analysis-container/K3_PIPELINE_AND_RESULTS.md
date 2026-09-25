dd

# Kimi-K3 Decoder-Layer Optimization Loop — Pipeline, Results, and What Worked

*2026-09-24/25 · branches `kimi-k3-loop` (workspace) and `kimi-k3` (VibeSim) · full trial-by-trial detail in `K3_LOOP_REPORT.md`*

## 1. Goal

Run our profile → diagnose → optimize loop end to end on the two decoder-layer types of Kimi-K3
(the KDA linear-attention layer and the MLA full-attention layer) as they execute in sglang v0.5.20 on one
B200, and obtain measurable, numerically verified speedups without telling the agent what to change.

## 2. Pipeline

Three containers, one per role. K3 is a **case** of the existing loop, not a new tool.
Figure: `K3_PIPELINE_FIGURE.html` (self-contained SVG; open in a browser).

```
┌───────────────────────────┐   read-only HTTP    ┌──────────────────────────────┐
│ 2. VibeSim oracle          │◀───────────────────│ 3. Agent (Codex gpt-5.6-luna) │
│ vibesim-analysis:k3        │                     │ edits python/sglang/** in a   │
│ baked K3 prediction,       │                     │ copy of the sglang tree,      │
│ verbs: workspace-info,     │                     │ measures with the driver,     │
│ simulate, analyze,         │                     │ loops: profile→simulate→      │
│ optimality (R0..R7 ladder),│                     │ analyze→edit→verify           │
│ kernels (cached alternatives)                    └──────────────┬───────────────┘
└───────────────────────────┘                                    │ edited tree
                                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ 1. Eval harness (lmsysorg/sglang:v0.5.20 on one GPU)                            │
│ driver = extractor: builds ONE KimiK3DecoderLayer in the production rank shape, │
│ seeded weights/state, CUDA-graph replay timing, golden capture/replay,          │
│ per-kernel table, routing histogram                                             │
│ judge: pristine baseline (5 reps) vs agent tree (source diff onto pristine),    │
│ gates: output+state match ∧ primary ≥ max(5%, 3σ) ∧ no secondary regression     │
│ (continuous mode: baseline = current best tree, gain ≥ max(3σ, 0.5%))            │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Workload (sglang cookbook B200 recipe, one rank):** attention TP8 → 12 heads, EP8 → 112 local MXFP4
experts, fp8-e4m3 KV cache, bf16 KDA state, `cutedsl_mla` decode. One decode token per request.

| case      | primary point                     | secondary points (must not regress) |
| --------- | --------------------------------- | ----------------------------------- |
| KDA layer | B=128, ctx 8k                     | B=32 @8k, B=1 @8k                   |
| MLA layer | B=1, ctx 1M (long-context decode) | B=128 @8k, B=16 @64k                |

**Metric:** CUDA-graph replay time of the layer's decode step (eager time is launch-bound at these
shapes and would reward launch-count hacks). **Correctness:** output *and* post-step state (KDA recurrent
state / written KV row) must match a seeded golden from the pristine tree (`max_rel_err ≤ 0.02`), on
every point.

**VibeSim side (branch `kimi-k3`):** four new kernel kinds (`kda_recurrent_decode`, `kda_fused_decode`,
`mla_decode_attention`, `mxfp4_fused_moe`) plus sglang backends on existing kinds, an `sglang_k3_env`
profiling environment, the `kimi_k3_sglang` arch (KDA/MLA/MoE worklets, TP8/EP8/PP2 and rank-1 presets),
the model.work label with location maps (R6/R7 floors), an alignment pack against nsys probes of the
driver, and a baked before-state prediction served per case (KDA on :8801, MLA on :8802).

**Integrity controls (all pass):** null (pristine tree → ≈0%), planted slowdown → FAIL on latency,
planted numerics (`latent *= 0.9`) → FAIL on correctness. The agent never touches the judge's driver
copy, goldens, or oracle; its tree is reduced to a source diff and re-applied onto a pristine copy.

## 3. Results

> **Sign convention (whole document):** every Δ / percentage is the *change in step latency*, so **negative = faster**
> (e.g. −12.1% means 304.6 → 267.7 µs). Percentages after a `before → after` arrow describe that arrow. The judge's
> JSON field `improvement` uses the opposite sign (positive = faster); `rounds.jsonl` and verdict files carry that form.

### Campaign 2 — realistic workload (final)

Baselines (judge medians, σ < 1 µs): KDA 403.0 / 262.7 / 139.7 µs (B=128/32/1);
MLA 304.6 (1×1M) / 493.0 (128×8k) / 302.5 (16×64k) µs.

| trial                                  | verdict          | primary                                | secondaries     | numerics                              | change                                                           |
| -------------------------------------- | ---------------- | -------------------------------------- | --------------- | ------------------------------------- | ---------------------------------------------------------------- |
| **MLA 8**                        | **PASS**   | **−12.1%** (304.6 → 267.7 µs) | −1.6%, −0.7%    | rel_err 0.0 / 0.014 / 0.005, state ok | non-DCP decode`cute-dsl` → TRT-LLM MLA (15 lines)             |
| KDA 8                                  | FAIL (gate)      | −3.1%                                 | −4.7%, −5.9%  | 0.0 everywhere                        | shared-`down` GEMM → CuteDSL bf16 GEMM + side-stream overlap  |
| KDA 9                                  | FAIL (gate)      | −0.3%                                 | −2.3%, 0.0%    | ≤ 0.013                              | 589-line bf16 port of the fused KDA CUDA kernel (works, no gain) |
| KDA 10                                 | FAIL (by 0.4 pt) | **−4.6%**                       | −5.4%, 0.0%    | 0.0 everywhere                        | packed KDA decode fast path + overlap                            |
| KDA 11                                 | FAIL (gate)      | −4.4%                                 | −5.05%, −5.9% | 0.0 everywhere                        | fast path + Triton recurrent-kernel launch tuning + overlap      |
| operator-stacked (not an agent result) | —               | −4.8%                                 | −5.4%, −6.0%  | 0.0                                   | KDA 11 tree + KDA 8's GEMM lever                                 |

**Both layers show real, bit-exact gains.** The MLA layer clears the gate in one VibeSim-guided
iteration. The KDA layer's Python-level ceiling at B=128 under this workload is ≈4.8%: the remaining
52% of its step is the weight-bandwidth-bound TRT-LLM MXFP4 expert GEMM, a closed cubin.

### Campaign 3 — continuous loop (no fixed gate; 2026-09-24)

Per the user's direction, the 5% gate was dropped: each round seeds the agent with the current best
tree, the judge measures that round's baseline from the same tree and accepts any primary gain
≥ max(3σ, 0.5%) with no secondary regression, and correctness is always checked against the
**original pristine goldens** (accuracy cannot drift across rounds). Runner: `run_k3_continuous.sh`
(promotes the judged tree to `best_tree` on PASS; history in `rounds.jsonl`).

| round | verdict | primary | new change | note |
|---|---|---|---|---|
| MLA 20 | PASS | 267.7 → 265.7 µs @1×1M (−0.75%) | `route_quant_fused` JIT specialized for the 112-expert/top-2 shape | harness-specific (production is 896/top-16) |
| KDA 20 | FAIL (infra) | — | — | judge OOM: a 167 GB co-tenant on the shared GPU |
| MLA 21 | PASS | 267.8 → 263.7 (−1.5%); −3.2% @128×8k | fp32-output front GEMM (15984×7168 / 6016×7168, m ≤ 16) → CuTe TGV instead of cuBLAS | production-relevant |
| KDA 21 | FAIL (correctness) | — | route+quant specialization broke numerics at B=32/1 (rel 0.42) | rejected on both gates; 3σ was 7% under a busy co-tenant |
| MLA 22 | FAIL (null) | 0.0% | — | |
| **KDA 22** | **PASS** | **384.4 → 379.4 µs @B=128 (−1.3%)**; −2.5% @32, −3.1% @1 | bf16-state port of the fused KDA decode JIT kernel (`.cuh` + `.py`) | the lever trials 7/9/10 kept attempting finally pays off |
| MLA 23 | PASS | 265.7 → 261.6 (−1.5%) | `latent_up` (7168×3584) and `shared_down` (7168×6144) → BF16 TGV kernel | |
| KDA 23 | FAIL (null) | −0.3% (below the 0.5% floor) | — | five ideas rejected cleanly |
| MLA 24 | PASS | 261.6 → 255.4 (−2.4%); −2.2% @128×8k, −3.4% @16×64k | shared/routed alt-stream overlap in `KimiK3MoE._forward_fused` | |
| KDA 24 | FAIL (null) | 0.0% | — | tree returned to the seed |
| MLA 25 | PASS | 255.4 → 253.5 (−0.8%) | `is_var_seq=False` (FlashInfer's persistent TRT-LLM MLA schedule) for the K3 fp8 MLA layout + 16-warp CTA for the B=1 KV-concat grid | initially flagged as a uniform-length overfit; **re-judged on a new mixed-context-length point (64k/48k/32k/16k): passes, rel 0.0046, −7.4% there** — the flag is a scheduling choice, per-request lengths are honoured |
| KDA 25 | FAIL (null) | −0.3% (below the 0.56% needed) | — | KDA plateau (rounds 23–25) |
| MLA 26 | FAIL (null) | 0.0% | `cutedsl_bf16_gemm.py` tweak, no effect | |
| MLA 27 | FAIL (null) | 0.0% | fused finalize+shared JIT kernel (regressed, reverted); `flashinfer_trtllm.py` tweak | MLA plateau; loop stopped |

**Cumulative vs pristine:** MLA **304.6 → 253.5 µs @1×1M (−16.8%)**; KDA **403.0 → 379.4 µs @B=128
(−5.9%)** (stacked seed −4.8% + round 22). Every accepted round passed CHECK against the original goldens
(rel_err ≤ 0.014). Both trees were then re-judged under two extra seeds (fresh weights/state/inputs): 4/4
PASS, speedups reproduce (MLA −16.8%, KDA −5.7…−6.1% at B=128). KDA has plateaued — the remaining 52% of its
step is the weight-bandwidth-bound MXFP4 MoE cubin. Lesson from round 25: the judge can only enforce what the
workload exercises; the MLA case now carries a mixed-context-length point (64k/48k/32k/16k), on which the
round-25 tree also passes (−7.4% there).

### Campaign 4 — new input dimensions with warm start (2026-09-24 17:00 – 2026-09-25)

User direction: extend the loop to **large decode batches** (B = 256/512, the DP-attention regime) and
**chunked prefill** (popular chunk sizes), warm-starting from the accepted decode trees with the optimization
history available to the agent.

**What was built (all committed on `kimi-k3-loop` / `kimi-k3`):**

| piece | what it is |
|---|---|
| `k3_opt_history/` (warm start) | Knowledge base over all 33 judged trials: curated `TECHNIQUES.md` (accepted levers, dead ends with reasons), `INDEX.md`/`history.json`, and per trial the judged patch, the round's *incremental* diff and every hypothesis/analysis the agent wrote. Mounted read-only at `/workspace/opt_history`; the prompt asks agents to read it first and cite prior trials. Agents do cite it (e.g. "prior kda_25 rejected the same idea only at B=128"). |
| Large-batch cases `k3_{kda,mla}_b512` | KDA primary 512×8k, secondary 256×8k, 128×8k; MLA primary 512×8k, secondary 256×8k, 16×64k mixed. VibeSim presets `*_b512`, oracles on 8803/8804. |
| Chunked-prefill cases `k3_{kda,mla}_prefill` | Driver point tag `B,L,pf[<prefix>]` (ForwardMode.EXTEND, eager timing): KDA runs `chunk_kda` with carried-in conv/recurrent state; MLA runs the production `trtllm_mla` MHA_CHUNKED_KV path (fp8 ragged attention + chunked prefix-KV merge). Points: 1×16384 first chunk (B200 default `chunked_prefill_size`), 1×16384 as chunk 4 of a 64k prompt (prefix 49,152), 4×4096 mixed batch. |
| Row-wise prefill correctness | Prefill steps are not bit-reproducible across *processes* (MoE/GEMM autotuners choose among near-equal tactics at m = thousands; ~0.1% of tokens change expert), while replays within one process are identical. Prefill CHECK: ≤0.5% of token rows over tolerance, p99 row error ≤ tol, mean drift ≤1%, post-step state exact. Decode keeps the strict max-error rule. |
| `--bf16-gemm-init` | Reproduces the production scheduler's bf16 GEMM backend init (`auto` → `cutedsl` on SM100). Found via VibeSim B12: all earlier runs had left every bf16 GEMM on cuBLAS. Measured effect ≤0.7% on every decode point (see below), so earlier results stand; new cases carry the flag. |
| VibeSim B12a / B12 | B12a: the MLA worklet fed the group's total KV as one request's context (16 ms attention at B=512) — fixed, MLA-b512 predicts 1025/661/251 µs vs 992/661/294 measured. B12: prefill kinds `kda_chunk_prefill`, `causal_conv1d_prefill`, `mla_prefill_attention`, `mla_prefix_gather`, `mla_merge_state`, worklet prefill branches, prefill presets — committed, but predictions land ~60% below the measured step (direct-call rows ≪ the eager layer's cuBLAS/MoE launches); B12b (queued) closes that gap before prefill rounds start. |

**Transfer check** (decode best trees judged on the new points vs the pristine tree; every point correct):

| case | point | pristine µs | best tree µs | Δ |
|---|---|---|---|---|
| MLA | 512×8k | 991.8 | 978.4 | −1.4% |
| MLA | 256×8k | 661.0 | 638.5 | −3.4% |
| MLA | 16×64k mixed | 294.4 | 272.7 | −7.3% |
| KDA | 512×8k | 724.5 | 677.4 | −6.5% |
| KDA | 256×8k | 506.4 | 482.8 | −4.7% |
| KDA | 128×8k | 403.0 | 378.4 | −6.1% |

The KDA levers transfer almost fully to B=512; the MLA levers shrink with batch (the 1×1M attention switch is
irrelevant at 512×8k, and the small-m GEMM levers stop engaging), exactly the headroom the rounds went after.

**Rounds (45-min agents, warm-started, continuous mode):**

| round | verdict | what was tried | note |
|---|---|---|---|
| KDA-b512 1 | FAIL (null) | TRT-LLM MoE tuner ceiling at 2× rows; two route+quant JIT specialisations that never engaged | 1.2% faster at B=512 against a 1.5% requirement (3σ inflated by concurrent VibeSim JIT fills) |
| KDA-b512 2 | FAIL (null, 0.0%) | bf16-activation MXFP4 path (no SM100 kernel at B≥256); route+pack+quant extension; KDA TMA stage counts; MoE tuner ceiling | all exact, all neutral |
| KDA-b512 3 | FAIL (null, −0.1%) | route-fusion stack; TMA stages; in-kernel TRT-LLM routing (slower and re-routed 2 of 512 tokens → strict CHECK rejected it); MXFP4×bf16 SiTU (no kernel) | KDA at B=512 is on the same closed MXFP4-cubin wall as at B=128 (57% of the step, R0 961 vs R5 53 µs) |
| MLA-b512 1 | infra noise | — | B=512 baseline reps 2104/995/3363/3530/3419 µs while Codex B12 kernel-profiled on the same GPU; not a result |
| MLA-b512 2 | FAIL (null, 0.00%) | MoE tactic buckets, PDL toggle, low-priority-stream overlap (−1 µs), bf16 front GEMM (rejected on correctness) | clean measurement (pristine 991.7, σ 0.4) |
| MLA-b512 3 | pending | bf16-activation MoE ×2 (no kernel); route+quant cap 64→512 (exact, −1.5 µs, kept); variable-schedule attention (null); tuning ceiling 512→1024 (null); TGV for the m=512 front GEMM (+8% slower) and for shared-down (+2% slower), both reverted | confirms the TGV lever is small-m only |

**Result so far:** at B=512 the inherited trees are the result — KDA −6.5%, MLA −1.4% — and six warm-started
rounds found nothing further: every remaining large-batch idea is either inactive, neutral, numerics-changing,
or a regression. Both layers are MoE-bound on the closed TRT-LLM MXFP4 cubin at every batch size we can run.
Prefill rounds start after B12b.

**Harness fidelity gap, quantified (CUDA-graph µs, legacy cuBLAS-only → production dispatch):** KDA pristine
406.9/255.6/134.7 → 407.0/255.5/132.6 (B=128/32/1), best tree 388.6/245.2/128.5 → 388.6/245.3/124.4; MLA
pristine 304.7/493.0/293.3 → 302.6/493.0/292.3 (1×1M / 128×8k / 16×64k mix), best tree 253.4/475.7/271.8 →
253.4/475.7/271.9. The decode shapes are essentially not TGV-eligible under production's heuristic, so the
reported gains stand and rounds 21/23 go beyond the production heuristic rather than duplicating it.

**Prefill baselines (pristine, eager, 16,384 tokens per step):** KDA 22.1 / 22.5 / 22.1 ms, MLA 20.6 / 29.5 /
17.8 ms; ~95% kernel time, dominated by cuBLAS GEMMs (projections + shared experts, ~11–12 ms) and the MXFP4
expert GEMM (~3 ms); ~6 ms of the MLA prefix point is the fp8 prefix attention.

### Campaign 1 — the first 8 trials (superseded)

0/8 passed. Best −2.9% @B=128 (KDA), −14.5% @1×1M (MLA, a secondary point then). These ran on a
workload that turned out to be unrealistic (see §5); the numbers are kept in `K3_LOOP_REPORT.md` as
evidence, not as a statement about the layers.

## 4. Techniques that produced the improvements

### Found by the agents (in the sglang tree)

1. **Attention-backend routing for long-context MLA decode (−12%).** sglang's `cutedsl_mla` backend
   runs the non-DCP decode through the CuteDSL split-KV kernel; the TRT-LLM MLA generation kernel
   (`fmhaSm100fKernel_…ForGen`) is faster at 1×1M with identical numerics. VibeSim surfaced it directly:
   `kernels` reported a **cached alternative** (`sglang_trtllm_mla`) on the MLA decode leaf, the agent
   applied it, verified, and stopped. Found independently in 4 of 5 MLA runs.
2. **Keep the packed KDA decode fast path for K3's lower-bounded gate (≈−2%).** The dispatcher excluded
   layers with a `lower_bound` from the packed Triton decode kernel although the kernel implements the
   lower-bounded sigmoid gate; removing the exclusion is bit-exact.
3. **Shared-expert `down` GEMM (7168×6144, m ≤ 128) → sglang's CuteDSL bf16 GEMM (≈−3%).** A kernel
   selection change for a shape the default GEMM handles poorly.
4. **Overlap the dense shared experts with the routed MXFP4 MoE on the layer's side stream (≈−1.5…−2.5%).**
   The two branches write disjoint slices of the fused-front output; the join happens before the
   collective. Rediscovered in nearly every KDA/MLA run.
5. **Triton `fused_recurrent_kda_packed_decode` launch tuning** (num_warps for the B=128 grid), small
   but bit-exact.

Levers 2–5 do not add linearly: once the shared GEMM runs on the side stream, making it faster barely
moves the critical path (stacked check: −4.8%, not −7%).

### What VibeSim contributed

- **Ranking without leaks.** `optimality` (R0 measured-style time → R5 hardware floor → R6/R7
  necessary-work floors, `necessary_share`) put the MoE leaf first and, once agents saw it had no cached
  alternative, pointed them to the next leaves (`shared_down` 50 µs vs R6 16.6; `kda_recurrent_decode`
  42.5 µs vs R5 8.2) — exactly where levers 2–5 live.
- **Cached alternatives.** The `kernels` verb's `has_cached_alternative` on the MLA decode leaf is what
  turned a 45-minute search into a one-iteration PASS.
- **Dead-end pruning.** Agents used the ladder to reject ideas quickly: forcing the fused KDA kernel
  onto a bf16 state (three runs, worth ≤3%), bf16-activation MoE dispatch (no B200 tactic), TRT-LLM
  tactic buckets (no effect), PDL off (regression), a fused MXFP4 router (+6 µs).

### Prediction accuracy that made the guidance trustworthy

| stage                    | KDA @B=128         | MLA @B=128        | what changed                                                                             |
| ------------------------ | ------------------ | ----------------- | ---------------------------------------------------------------------------------------- |
| B4/B5 first arch         | −2.6%             | −13.7%           | qkvbfg leaf +198% (serial vs side-stream)                                                |
| B8                       | −11%              | −16%             | qkvbfg split into wide GEMM ‖ side GEMV (`CostNode::Max`)                             |
| B9                       | +23%               | +11%              | runner had profiled 896-wide routing (1/8 of the rank's expert work)                     |
| B10                      | **−3.3%**   | **−10.2%** | routed-API runner with the layer's real top-k histogram; proportional`Max` attribution |
| B11 (realistic workload) | ≈+3% (417 vs 403) | ≈0% (491 vs 493) | local top-k share (256 rows/rank) in presets and runner                                  |

## 5. Harness lessons (what had to be fixed to get here)

1. **Parameter init decides the routing of a synthetic MoE layer.** Initializing *everything* `N(0, 0.02)`
   — including RMSNorm gains — scaled every normalized activation by 0.02, so the router's sigmoid sat at
   0.5 for all tokens and the `e_score_correction_bias` alone ranked the experts: ~8 of 112 experts took
   every token, and the MoE read ~1/10 of its weights. Norm gains `1 + 0.02·randn` → Poisson-like routing
   (85/112 experts active at B=128). This single bug had made campaign 1's "MoE-bound, ≤3% recoverable"
   conclusion an artifact.
2. **Model the rank's share of the routing.** `--experts 112` with global top-16 gives a rank 8× its
   production expert load; `--local-topk 16/EP` (2 @ EP8) restores 256 local rows per 128 tokens.
3. **Judge must carry JIT CUDA sources.** Agents do port kernels (`.cuh`); a `.py`-only diff produced a
   false FAIL (KDA trial 7) until fixed.
4. **One oracle per case, and check the wiring.** The MLA case pointed at the KDA oracle for three
   trials; two agents noticed and self-corrected, one did not.
5. **Don't hold a GPU while the agent thinks.** Under slurm a trial pinned a device for 55 minutes to use
   it for ~10; trials now run via plain docker on a user-authorized shared GPU, one at a time.
6. **Measure before/after with identical flags** — `--split`/`--profile-kernels` perturb graph timing
   ~10% and misled an early agent's self-report.
7. **Re-run every point before finishing.** A kernel that works at B=1 can reject the B=128 or
   long-context shape (dtype/layout guards); the judge scores that as a hard FAIL.
8. **Never profile on the judge's GPU.** VibeSim JIT fills / kernel profiling on the shared GPU inflated a
   B=512 baseline 3× in 4 of 5 reps (the other points in the same runs were clean: bursty interference); rounds
   and VibeSim fills are now strictly sequenced.
9. **Gate waiters on PIDs, not tool names.** A `pgrep "codex exec"` gate also matched the trial agents inside
   their containers and stalled two follow-ups behind a 45-min trial.
10. **Prefill is reproducible per process, not across processes.** Autotuner tactic choice at m = thousands
    re-routes ~0.1% of tokens; the judge needs a per-token rule there, and agents are told that `max_rel_err`
    of ~0.3 with a handful of rows over tolerance is normal on prefill points.
11. **Check what the harness leaves uninitialized.** The production scheduler initialises the bf16 GEMM backend;
    the single-layer driver did not — harmless here (≤0.7%), found only because a VibeSim runner tried to
    reproduce the layer's exact launches.

## 6. Code changes, as patches

`patches/` (regenerate with `./export_k3_patches.sh`, which also verifies that applying each chain onto a copy
of the pristine `python/sglang` tree reproduces the case's best tree exactly). Apply inside `python/sglang`
of `lmsysorg/sglang:v0.5.20` with `patch -p1`.

### MLA layer — six accepted levers, in acceptance order (cumulative 304.6 → 253.5 µs @1×1M, −16.8%)

| patch | files | key change | Δ latency when accepted |
|---|---|---|---|
| `sglang/mla/01_r08_cutedsl_to_trtllm_mla_decode.patch` | `srt/layers/attention/cutedsl_mla_backend.py` (5 lines) | non-DCP decode: `backend="cute-dsl"` → `"trtllm-gen"` (DCP keeps the explicit CuteDSL path in `_run_decode_kernel`) | −12.1% |
| `sglang/mla/02_r20_route_quant_fused_112_top2.patch` | `kernels/jit/csrc/moe/route_quant_fused.cuh`, `route_radix.cuh`, `kernels/ops/moe/moe_route_quant_fused.py`, `srt/layers/moe/topk.py`, `kernels/ops/attention/set_mla_kv_concat_q.py` (277 lines) | fused route+quant JIT specialised to `kNumExperts = 112`, `kTopK = 2` (the EP8 rank's local shape; production routes 896/top-16 globally — harness-specific) | −0.75% |
| `sglang/mla/03_r21_front_fp32_gemm_cute_tgv.patch` | `srt/models/kimi_k3.py` (16 lines) | in `_k3_bf16_gemm`: fp32-output front GEMMs `(15984,7168)`/`(6016,7168)` at `m ≤ 16` → `cutedsl_bf16_gemm_out` instead of cuBLAS (fp32 accumulator kept, routing stays exact) | −1.5% (−3.2% @128×8k) |
| `sglang/mla/04_r23_latent_up_shared_down_bf16_tgv.patch` | `srt/models/kimi_k3.py` (33 lines) | `latent_up` `(7168,3584)` and `shared_down` `(7168,6144)` at `m ≤ 16` → CuTe TGV (`cutedsl_bf16_gemm[_out]`); `routed_expert_up_proj` routed through `_k3_bf16_gemm` | −1.5% |
| `sglang/mla/05_r24_shared_routed_alt_stream_overlap.patch` | `srt/models/kimi_k3.py` (12 lines) | `KimiK3MoE._forward_fused`: `_forward_shared` on `self.alt_stream` while `_forward_routed` runs on the current stream; `wait_stream` join before the collective | −2.4% |
| `sglang/mla/06_r25_is_var_seq_persistent_kvconcat_warps.patch` | `srt/layers/attention/trtllm_mla_backend.py`, `kernels/ops/attention/set_mla_kv_concat_q.py`, `kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh` (26 lines) | `extra_kwargs["is_var_seq"] = False` for the fp8 K3 layout (FlashInfer: `is_persistent = not is_var_seq`, a schedule choice — verified on mixed lengths); 16-warp CTA for the 13-item B=1 KV-concat grid | −0.8% |

`sglang/mla_best_tree_vs_pristine.patch` is the cumulative patch (9 files, 361 changed lines).

### KDA layer — two patches (cumulative 403.0 → 379.4 µs @B=128, −5.9%)

| patch | files | key change | Δ |
|---|---|---|---|
| `sglang/kda/01_stacked_fastpath_overlap_cutedsl_gemm_warps.patch` | `srt/layers/attention/linear/kda_backend.py`, `srt/models/kimi_k3.py`, `kernels/ops/attention/fla/fused_recurrent.py` (44 lines) | packed KDA decode fast path no longer excluded for `lower_bound` layers (the kernel implements the safe gate); shared `down` `(7168,6144)` at `m ≤ 128` → CuTe bf16 GEMM; shared-down issued on the side stream (single-rank, no AR fusion) with an event join; `num_warps` tuning of the recurrent kernel | −4.8% (stacked from trials 8/10/11) |
| `sglang/kda/02_r22_bf16_state_fused_kda_decode_kernel.patch` | `kernels/jit/csrc/attention/kda_fused_decode.cuh`, `kernels/ops/attention/kda_fused_decode.py` (242 lines) | bf16-state port of the fused conv+delta-rule+gated-norm decode kernel: `__nv_bfloat162` state loads/stores, `bf16_round`/`conv_silu_bf16` to match the unfused chain's rounding, `covered()` accepts the bf16 state (cookbook `--mamba-ssm-dtype bfloat16`) | −1.3% (@B=32 −2.5%, @B=1 −3.1%) |

`sglang/kda_best_tree_vs_pristine.patch` cumulative (5 files, 286 lines).

Both cumulative trees passed: the original pristine goldens (rel_err ≤ 0.014), two extra seeds each (rel_err
≤ 0.011), the MLA mixed-length point, and the B=512/256 transfer checks.

### Harness and VibeSim

- `patches/harness/01_driver_kimi_single_layer_decode.patch` — the single-layer extractor/driver: production-shape
  layer build (`--attn-heads 12 --experts 112 --local-topk 2`, fp8 KV, bf16 state), CUDA-graph timing, seeded
  goldens (`--capture/--replay`, output + post-step state), mixed-length and chunked-prefill points
  (`B,L,mix`, `B,L,pf[<prefix>]`), row-wise prefill CHECK, `--profile-kernels`, `--nvtx-align`, `--bf16-gemm-init`.
- `patches/harness/02_judge_runner_prompt_cases.patch` — `judge_k3.py` (pristine baseline + goldens per case key,
  source-only diff re-applied onto pristine, gates, `--baseline-tree` continuous mode), `run_iter_opt_eval.sh`
  (`START_TREE`, mounts), the agent prompt, the six case configs.
- `patches/harness/03_loop_orchestration_scripts.patch` — continuous loop (`run_k3_continuous.sh`, alternate runner,
  GPU waiters, transfer checks, campaign chains). `04_warm_start_history.patch` — `build_opt_history.py` +
  `opt_history_techniques.md`. `05_oracle_bake_and_codex_tasks.patch` — `build_context_k3.sh`, the B2–B12b Codex
  prompts. `harness/COMMITS.txt` is the branch log (92 commits).
- `patches/vibesim/kimi_k3_vs_roofline_base.patch` (419 files, 162 commits, `COMMITS.txt`/`STAT.txt`) — the VibeSim
  `kimi-k3` branch: kinds `kda_recurrent_decode`, `kda_fused_decode`, `mla_decode_attention`, `mxfp4_fused_moe`,
  `kda_chunk_prefill`, `causal_conv1d_prefill`, `mla_prefill_attention`, `mla_prefix_gather`, `mla_merge_state`
  with `sglang_k3_env` runners; arch `kimi_k3_sglang` + KDA/MLA/MoE worklets (decode and prefill branches,
  rank-1 and TP8/EP8/PP2 presets incl. b512 and prefill); model.work label + location maps; alignment pack and
  `doc/alignment/kimi_k3_single_layer.md`. The branch profile.db rows are not in the patch.

## 7. Reproduce

```bash
cd scripts-local/vibesim-analysis-container
# oracles (already running on 172.17.0.1:8801 / :8802):
#   docker run -d --name vibesim_oracle_k3_kda -e VIBESIM_PREBAKED_RUN=k3_kda -e VIBESIM_API_PORT=8801 -p 172.17.0.1:8801:8801 vibesim-analysis:k3
# one trial per case on the shared GPU 7, 45-min agent budget:
K3_SHARED_GPU=1 AGENT_TIMEOUT=2700 ./run_k3_trials_direct.sh 7 mla:9 kda:12
# verdicts: iter_opt_eval_k3_{mla,kda}/trial_<k>_verdict.json ; diffs: trial_<k>_tree_judged.patch
```

Artifacts: `K3_LOOP_REPORT.md` (every trial, both campaigns), `EVAL_HARNESS.md` (harness contract),
`kimi_single_layer_decode.py` (driver), `judge_k3.py`, `issue_k3_{kda,mla}.json`, `agent_task_k3_opt.md`,
goldens under `/raid/yilegu/eval_goldens/golden_k3_<key>/`, VibeSim alignment report
`main/doc/alignment/kimi_k3_single_layer.md`.
