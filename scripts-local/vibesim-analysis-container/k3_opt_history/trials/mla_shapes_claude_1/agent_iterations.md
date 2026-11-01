# MLA_SHAPES_CLAUDE trial 1: agent iterations

## agent log.md

- iter_00: baseline; 1x1M 241.09 | 16x64k 261.54 | 128x8k 462.43 | 32x8k 247.23 | 1x8k 130.50 us; top node mla_decode_attention (HVPerCta256 min-latency mode at bs=1)
- iter_01: bs=1 KV split k=8 + LSE merge, 256MB MLA workspace, gate L>=64k; 1x1M 241.09->222.69, others flat; CHECK pass
- iter_02: fused set_mla_kv_concat_q_fp8 writes KV-split q replicas; 1x1M 222.69->216.54, others flat (261.54/462.37/247.26/130.50); CHECK pass
- final: same tree as iter_02, fresh smoke (LAYER_SMOKE_OK) + replay: 1x1M 218.43 | 16x64k 261.47 | 128x8k 462.30 | 32x8k 247.23 | 1x8k 130.46; CHECK pass all (run-to-run ~2 us at 1x1M vs iter_02's 216.54)

## iter_00
### analysis.md

# iter_00 -- baseline
VibeSim prediction `.:k3_mla` = p_c7687e585abe4b5fbacdc89caf0c9761 (vs_sim.json, vs_run_summary.json,
vs_operator.json, vs_optimality.json, vs_kernels.json). Run optimality 0.127.

Plain replay (plain.log): 1x1M 241.09 | 16x64k mix 261.54 | 128x8k 462.43 | 32x8k 247.23 | 1x8k 130.50 us.

Top nodes by headroom (optimality, scope=iter):
- unified.mla.moe.mxfp4_fused_moe: r0/r5 44, necessary share 0.60 -- closed TRT-LLM MXFP4 cubin; history
  (TECHNIQUES B: tactic buckets, bf16 act, fused finalize) lists it as dead ends at B<=128.
- unified.mla.attention.mla_decode_attention: r0 358.8 us / r5 227.6 us (summed over points), r0/r5 1.58,
  headroom 131 us, r6>0 -> kernel-efficiency bottleneck on necessary work. Largest actionable node.
- unified.mla.moe.merged_front: r0/r5 1.31 (TGV fp32 GEMM, already tuned in A2).

Profiler table 1x1M (profile_B1_L1048576.json): fmhaSm100f...HVPerCta256...MultiCtasKv 130.1 us of a 241 us
step = 4.6 TB/s over the 604 MB fp8 KV; HBM read floor (torch.sum of 604 MB) is 102.2 us.
Mapping: trtllm_mla_backend.py `_run_decode_kernel` -> flashinfer trtllm_batch_decode_with_kv_cache_mla.
Root cause (fmhaKernels.cuh computeCtaAndClusterConfig): batch==1 with 576/512 dims enables
isDsv3MinLatencyMode -> headDimPerCtaV=256, so 2 CTAs re-read every KV tile for only 12 heads.

### result.json

```
{
 "iteration": 0,
 "desc": "baseline (starting tree)",
 "command": "./run.sh \"1,1048576;16,65536,mix;128,8192;32,8192;1,8192\"",
 "points": {
  "1;1048576": {
   "latency_us": 241.09,
   "check_pass": null,
   "max_rel_err": null,
   "state_ok": null,
   "nan": null
  },
  "16;65536;mix": {
   "latency_us": 261.54,
   "check_pass": null,
   "max_rel_err": null,
   "state_ok": null,
   "nan": null
  },
  "128;8192": {
   "latency_us": 462.43,
   "check_pass": null,
   "max_rel_err": null,
   "state_ok": null,
   "nan": null
  },
  "32;8192": {
   "latency_us": 247.23,
   "check_pass": null,
   "max_rel_err": null,
   "state_ok": null,
   "nan": null
  },
  "1;8192": {
   "latency_us": 130.5,
   "check_pass": null,
   "max_rel_err": null,
   "state_ok": null,
   "nan": null
  }
 }
}
```

## iter_01
### hypothesis.md

# iter_01 hypothesis -- bs=1 KV split for the trtllm-gen MLA decode
Node: unified.mla.attention.mla_decode_attention (iter_00 analysis: r0/r5 1.58, 131 us headroom).
If the bs=1 request is presented to the kernel as k page-aligned pseudo-requests (batch k != 1), FlashInfer
leaves the DSv3 min-latency mode (HVPerCta256, KV read twice) and uses the full-HV MultiCtasKv schedule;
merging the k partial outputs through the returned LSE (base 2) is exact up to rounding.
Micro-bench (scratch/mla_bench*.py): 1x1M unsplit 131.6 us; k=8 106.4; k=16 104.4 -> expect ~-20 us/step.
1x8k k=16: 10.7 -> 7.6 us in isolation; 16x64k/32x8k/128x8k: no gain -> gate to bs==1.

Built on:
- mla_b512_claude_2 (tail-split machinery: extra pseudo rows, `tail_block_kv_indices`, fused
  `extra_q_rows`, LSE merge) -- reused its plumbing, replacing the tail rule by an even k-way split.
- A8 / mla_25 (is_var_seq=False persistent schedule) -- kept.
Rules out / revises: mla_b512_claude_3 "closed cubin, no knob left" -- the batch shape is a knob at bs=1.

### analysis.md

# iter_01 -- bs=1 KV split (k-way page-aligned pseudo-requests + LSE merge)
VibeSim prediction `.:k3_mla` (static prediction; no new-profile ingest verb) re-queried: vs_optimality.json.
Node targeted: mla_decode_attention (r6>0, r0/r5 1.58 -> kernel inefficiency).

First replay failed at B=1: FlashInfer autotune profiled the new LSE variant at batch bucket 8192 and
needed a 202,375,168 B softmax-stats workspace > 150 MB MLA workspace. Fix: DEFAULT_WORKSPACE_SIZE_MB 256
(MHA backend already uses 512).
First working replay: 1x1M 224.64, but 1x8k regressed 130.50 -> 134.56 (in-graph replicate/merge/LSE cost
beats the gain at 8k) -> gated on max_seq_len >= 65536 (KV_SPLIT_MIN_SEQ_LEN); 1x8k back to 130.53.
k sweep at 1x1M (plain replay): k4 239.17, k8 222.69, k12 241.06, k16 224.35, k32 226.78 -> k=8
(non-monotonic: k=4/12 pick a worse kernel schedule).
Profile 1x1M k=8 (profile.json): fmha now ...HV512...MultiCtasKv (no HVPerCta256) 103.9 us (was 130.1);
added: float8 copy (query replicas) 2.7, _kv_split_merge 2.4, ComputeLSEFromMD 1.6 us.
Note: TGV GEMM per-kernel times look inflated in this profile (PDL wait accounted to the kernel); the
graph latency is authoritative.
CHECK sensitivity: a deliberately broken merge (chunk 0 only) still passed with max_rel_err 0.0 at 1x1M
(scratch/broken_merge.log) -- the layer output is insensitive to 1M-context attention with the seeded data,
so the split was verified independently (see iter_02 / scratch/test_split.py).

### result.json

```
{
 "iteration": 1,
 "desc": "bs=1 KV split (k=8 page-aligned pseudo-requests + LSE merge), 256MB MLA workspace",
 "sweep_1x1M_us": {
  "k4": 239.17,
  "k8": 222.69,
  "k12": 241.06,
  "k16": 224.35,
  "k32": 226.78
 },
 "points": {
  "1;1048576": {
   "latency_us": 222.69,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 241.09,
   "note": "final k=8 (iter_01/k8b.log; reproduces k8.log 222.69). k=16 full replay gave 224.35; other points unaffected by k (split gated to bs==1 and max_seq_len>=65536)"
  },
  "16;65536;mix": {
   "latency_us": 259.65,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 261.54
  },
  "128;8192": {
   "latency_us": 463.46,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 462.43
  },
  "32;8192": {
   "latency_us": 247.26,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 247.23
  },
  "1;8192": {
   "latency_us": 130.53,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 130.5
  }
 }
}
```

(diff.patch: 296 lines, files: /sgl-workspace/sglang/python/sglang/kernels/ops/attention/mla_decode_tail_split.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py)

## iter_02
### hypothesis.md

# iter_02 hypothesis -- write the KV-split query replicas in the fused KV-write/q-concat kernel
iter_01 profile: the k-1 query replicas cost a separate aten float8 copy (2.7 us) and break the PDL chain
set_mla_kv_concat_q_fp8 -> fmha. The fused kernel already allocates `extra_q_rows` (mla_b512_claude_2
plumbing); letting each Q warp store its converted row k times removes the launch.
Ruled out first: passing an expanded (stride-0) query to trtllm-gen -- gives wrong output
(max diff 1.76 vs 0.085 max value; scratch/mla_bench3.py), the kernel/TMA does not honour stride 0.
Builds on A8 (16-warp CTA for the B=1 KV-concat grid, kept). Expect ~-3 us at 1x1M; no effect elsewhere
(q_replicas=1).

### analysis.md

# iter_02 -- fused query replicas
VibeSim `.:k3_mla` re-queried (vs_optimality.json): mla_decode_attention remains the node; measured fmha
101.6 us at 1x1M is now at the HBM floor (102.2 us torch.sum reference) -> remaining node headroom is the
split overhead only (merge 2.5 + ComputeLSEFromMD 1.7 us) and launch/scheduling (r6~r5 region).
Replay (replay.log): 1x1M 222.69 -> 216.54 (-6.2 us, more than the removed 2.7 us copy: PDL chain restored);
others unchanged (261.54 / 462.37 / 247.26 / 130.50). CHECK pass everywhere, state_ok.
Profile (profile.json): set_mla_kv_concat_q_fp8 3.2 -> 4.0 us (writes 8x the q rows), float8 copy gone.
Independent numerics (scratch/test_split.py, peaked + spread attention, L in {1M, 1M-37, 64k, 70013, 300001}):
q replicas byte-identical, KV row written correctly, split vs unsplit trtllm-gen max rel err <= 0.0068,
split vs fp32 exact 0.0036-0.0044 (unsplit 0.0026-0.0052).
Next candidates judged not worth it: MoE cubin (history dead ends), merge fusion (<=2.5 us, would need
flashinfer's internal MD stats layout).

### result.json

```
{
 "iteration": 2,
 "desc": "KV-split query replicas written by the fused set_mla_kv_concat_q_fp8 kernel (drops the aten fp8 copy)",
 "points": {
  "1;1048576": {
   "latency_us": 216.54,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 222.69,
   "baseline_us": 241.09
  },
  "16;65536;mix": {
   "latency_us": 261.54,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 259.65,
   "baseline_us": 261.54
  },
  "128;8192": {
   "latency_us": 462.37,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 463.46,
   "baseline_us": 462.43
  },
  "32;8192": {
   "latency_us": 247.26,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 247.26,
   "baseline_us": 247.23
  },
  "1;8192": {
   "latency_us": 130.5,
   "check_pass": true,
   "max_rel_err": 0.0,
   "state_ok": true,
   "nan": false,
   "before_us": 130.53,
   "baseline_us": 130.5
  }
 },
 "independent_attention_check": "scratch/test_split.py: split vs unsplit trtllm-gen max rel err <= 0.0068 on mixed-peak distributions (L=1M,1M-37,64k,70013,300001); split vs fp32 exact 0.0036-0.0044 (unsplit 0.0026-0.0052)"
}
```

(diff.patch: 461 lines, files: /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/mla_decode_tail_split.py, /sgl-workspace/sglang/python/sglang/kernels/ops/attention/set_mla_kv_concat_q.py, /sgl-workspace/sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py)
