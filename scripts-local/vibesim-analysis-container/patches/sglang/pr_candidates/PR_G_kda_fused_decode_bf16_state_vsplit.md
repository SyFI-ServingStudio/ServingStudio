# PR G — KDA fused decode kernel: bf16 recurrent state + small-batch 4-CTA cluster split

**Status: candidate.**
- **The gain:** −4.0% decode time per token at 1M context end to end (2 runs). It is the whole gain
  of the KDA decode lineage at batch 1.
- **Scope:** 2 files, kernel only. No `kimi_k3.py` change.
- **Upstream status:** the problem is still present on `main`. The fused decode kernel there is fp32-state only, so
  with the cookbook's `--mamba-ssm-dtype bfloat16` it never runs.

| | |
|---|---|
| patch (vs stock sglang v0.5.20, `patch -p1` inside `python/sglang`) | `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/patches/sglang/pr_candidates/PR_G_kda_fused_decode_bf16_state_vsplit.patch` |
| git | `/raid/yilegu/k3_merge`, branch `cand_G3_fused_kernel_only` (parent = branch `pristine`) |
| exported tree used in the benchmarks | `/raid/yilegu/k3_e2e_trees/cand_G3_fused_kernel_only` |
| size | 2 files, `kernels/jit/csrc/attention/kda_fused_decode.cuh` and `kernels/ops/attention/kda_fused_decode.py` |
| origin | KDA decode campaign 3 r22 (T14 bf16 state), `kda_b512_claude` r1/r2 (T15 vectorized ld/st, T31 prologue hoist), `kda_shapes_claude` r4 (T41 4-CTA cluster V-split) |
| related upstream | #41714, #43657 and #41570 move KDA toward bf16 state; on main, `kda_fused_decode.py` `covered()` still requires fp32 |

## 1. The problem

- **What the fused kernel does.** sglang's `kda_fused_decode` does a whole KDA decode step in one launch: conv1d
  update, delta-rule recurrence and gated RMSNorm.
- **Why it never runs for K3 on the recommended setup.** Its `covered()` accepts only an **fp32** recurrent state:
  `ssm_states.dtype == torch.float32`. The Kimi-K3 cookbook recipe uses `--mamba-ssm-dtype bfloat16`. So in stock,
  every KDA decode step logs `KDA fused decode rejected by covered(): … ssm_states … torch.bfloat16`.
- **What stock runs instead.** Not the packed kernel either, because that one requires `lower_bound is None` and K3
  has a lower-bounded gate. Decode falls back to the **unfused chain**: separate conv, recurrence and norm launches.
- **The second cost, at batch ≤ 3.** Even with the fused kernel, the grid is one CTA per (token, head). That is
  12 CTAs at batch 1 on 148 SMs, so the step is latency-bound.

## 2. The change

### `kernels/ops/attention/kda_fused_decode.py`

```python
-        and ssm_states.dtype == torch.float32
+        and ssm_states.dtype in (torch.bfloat16, torch.float32)

+def _v_split(B, H):
+    # Measured on B200 (H=12): the split wins while B*H*4 CTAs fit one wave;
+    # past that (B=16, 32, 128) the base kernel is as fast or faster.
+    return 4 if B * H * 4 <= _num_sms() else 0
```

### `kernels/jit/csrc/attention/kda_fused_decode.cuh`

- **bf16 state (T14).** The state is templated on its storage type (fp32 or bf16) and accumulated in fp32.
- **Vectorized loads/stores (T15).** Each bf16 load and store moves 8 bytes (4 values), so a warp writes full
  32-byte sectors. With bf16 state, 32 KB of smem allows 5 blocks/SM (48 regs, no spills): 93 → 82 µs at B = 512.
- **Prologue hoist and quad-row reduction (T31).** The lower warps do the q conv + decay while the upper warps do the
  k and v conv. One butterfly shuffle computes two sums, bitwise equal to two `reduce_sum`s.
- **4-CTA cluster V-split for small grids (T41).** Each (token, head) runs on a cluster of 4 CTAs.
  - Each CTA owns 32 of the 128 V rows. Every CTA redundantly computes the q/k conv, decay and L2 norm.
  - The output-norm partial sums go through DSMEM (`mapa` + a split cluster barrier). The sum order matches the base
    kernel, so output and state are **bitwise equal** to the unsplit kernel.
  - Inputs the previous kernel doesn't write (slot index, TMA'd state, conv taps, params) are read **before the PDL
    wait**, hiding their latency under the qkvg GEMM tail.

## 3. Which piece matters (1M-context decode TPOT, stable `w1l` workload, 511 steps; stock = 6.15 / 6.10 ms)

| tree | contents | TPOT | Δ |
|---|---|---|---|
| `cand_G_kda_decode_stack` (7581) | full KDA decode lineage (+ `kimi_k3.py` overlaps, MoE front split, L2 prefetch…) | 5.89 ms | −3.8% |
| `cand_G1` (7643) | this patch + T13 (packed decode for the lower-bound gate) | 5.89 ms | −3.8% |
| `cand_G2` (7662) | T13 only | 6.10 ms | ≈ 0 |
| **this patch** (`cand_G3`, 7663, 7672) | fused kernel only | **5.87 / 5.89 ms** | **−4.0%** |
| D = `patches/sglang/kda/02_r22_bf16_state_fused_kda_decode_kernel.patch` (7671) | T14 only: bf16 state in the fused kernel, without T15/T31/T41 | 5.92 ms | **−3.3%** |

- **Most of the gain is just letting the fused kernel run.** bf16 state eligibility alone (D, T14) gives −3.3%;
  T15/T31/T41 add ~0.9 pt at batch 1.
  - **Correction:** the first-wave "PR D ≈ 0" was measured with the noisy 31-step W1 TPOT. On the stable 511-step
    workload, D alone clears 3%.
  - **Packaging option:** submit D first (smaller), and the cluster split + tunings as the follow-up.
- **The fused kernel carries the whole gain.** The `kimi_k3.py` overlap levers add nothing at batch 1.
- **T13 doesn't matter here.** It overlaps open #41570. The fused path is tried before the packed path, so once the
  fused kernel engages, T13 is dead code.

## 4. Correctness and per-layer speed (single-layer judge vs stock: seed 1, fresh goldens, job 7676)

| case | point | Δ layer step | max rel err |
|---|---|---|---|
| KDA decode shapes | 1 × 1M / 16 × 64k / 128 × 8k / 32 × 8k / 1 × 8k | −4.6* / −2.9 / −4.1 / −3.3 / −7.5% | 0.0 / 0.0046 / 0.0088 / 0.0 / 0.0 |
| KDA large batch | 512 × 8k / 256 × 8k / 128 × 8k | −6.2 / −4.8 / −3.6% | 0.0158 / 0.0128 / 0.0088 |
| KDA verify (MTP, 3 draft tokens) | 64 × 8k / 128 × 8k / 16 × 64k | 0.0 / 0.0 / +0.1% | 0.0: verify doesn't use this kernel |

All points PASS (limit 0.02).

\* The judge's stored stock time at 1 × 1M is 3.1% faster than a fresh stock run (stock-vs-stock control, job 7618).
Against a fresh run, this point is ≈ −7.5%.

**What the errors mean.** They are the bf16-state fused path vs stock's unfused bf16 chain; there is no bit-identity
expectation between the two. They grow with batch size, to 0.016 at B = 512. The cluster split itself (B ≤ 3) is
bit-exact vs the base fused kernel.

## 5. End to end (48-layer K3, dummy weights, 8 × B200, TP8/EP8)

| workload | metric | stock | this patch | Δ |
|---|---|---|---|---|
| W1l 1 × 1,040,000 → 512 | TPOT (511 steps) | 6.125 ms (2 runs) | 5.87 / 5.89 ms (jobs 7663, 7672) | **−4.0%** |

- **Larger batches show less.** The full lineage (`cand_G`) gave only +1.0…+1.4% throughput at 128 × 8k and
  512 × 8k, and +1.4% at 16 × 64k. There the KDA decode is a small part of a step dominated by the MoE and by
  prefills.
- **Ordering vs PR C.** On 1M decode, PR C (MLA KV split, −5.9%) and this patch (KDA, −4.0%) act on different
  layers, and their sum is about the merged tree's −9.6%.

## 6. Before submitting upstream

- **Coordinate with #41714, #43657 and #41570** (bf16 KDA state work). Upstream may want bf16 state for the fused
  kernel in a specific accumulate/round convention.
- **Split it.** Send (1) bf16 state support in the fused kernel, then (2) the cluster V-split for small grids. T15
  and T31 are internal tunings of (1).
- **Kernel tests:**
  - fused bf16 vs the unfused chain, over B ∈ {1, 2, 3, 4, 16, 128, 512}, with the error bound stated;
  - cluster-split vs base kernel, bitwise;
  - the `covered()` fallbacks.
- **Hardware:** the cluster/DSMEM path needs SM90+; it was measured on B200 only.
