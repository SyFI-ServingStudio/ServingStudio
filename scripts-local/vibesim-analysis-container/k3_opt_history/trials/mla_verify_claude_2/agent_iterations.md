# MLA_VERIFY_CLAUDE trial 2: agent iterations

## iter_00
### analysis.md

# iter_00 — baseline of the starting tree (no edit)

Plain run (fixed flags): 64x8k vk3 **472.4 us**, 128x8k vk3 **628.1 us**, 16x64k vk3 **399.8 us** (graph replay).
Golden captured -> /workspace/opt_run/golden_*.pt (per point).

## VibeSim
- workspace-info: predictions k3_mla = p_c7687e585abe4b5fbacdc89caf0c9761 (and k3_kda). The container
  serves a fixed *decode* before-state (4 cases, B=1.. decode), `simulate` is idempotent; the
  prediction id handle is not resolvable by optimality/kernels (0 matches) -> used the default log dir
  (logs/k3_mla), which is the same prediction.
- run_summary: optimality_ratio 0.127; gap buckets batching 33%, hardware_gap 37%, imbalance 18%.
- optimality(scope=iter), top by r0-r5 headroom (sum over the 4 decode cases):
  | node | r0 us | r5 us | r6 us | necessary_share |
  |---|---|---|---|---|
  | mxfp4_fused_moe | 379 | 8.6 | 228 | 0.60 |
  | attention + output gate | 436 | 129 | 462 | 1.06 |
  | mla_decode_attention | 359 | 228 | 454 | 1.27 (BW bound) |
  | merged_front | 159 | 121 | 30 | 0.19 |
  | shared_down | 65 | 45 | 11 | 0.17 |
  | latent_up | 43 | 26 | 6 | 0.15 |
  | output_gate | 33 | 11 | 2.8 | 0.08 |
  | fused_qkv_a_proj | 30 | 15.5 | 3.8 | 0.13 |
- Reading: the prediction is for decode (m<=128) not the vk3 verify shape (m=256..512), so the ladder
  is used qualitatively: MoE and MLA attention are the two big nodes; MLA r6>r0 => bandwidth-bound;
  the dense GEMMs (merged_front / shared_down / latent_up / output_gate / qkv_a) have
  necessary_share <0.2 => implementation/batching gap.

## Measured per-kernel (profile_B64_L8192vk3.json, diagnostic run)
MoE fc1 169.0 + fc2 87.4 (=256 us; ~1.97 GB MXFP4 weights -> ~7.7 TB/s, already at HBM roofline,
cf. mla_b512_claude_1: same cubin tile t128x16x256 already selected) | MLA cute-dsl split-KV 62.8
(302 MB fp8 KV -> 4.8 TB/s, floor ~38 us) | nvjet 112x128 x3 48.0 | nvjet 224x128 (merged front fp32)
47.3 | cublasLt splitKreduce 25.5 (!) | nvjet 64x64 x2 19.5 | nvjet 96x64 splitK 13.6 | attn_res x2 12.6
| situ_and_mul 10.4 | finalize 8.0 | set_mla_kv_concat_q_fp8 7.3 | ...  kernel sum 567.7 vs graph 478.5
(alt-stream overlap).

## Pick
1. MLA attention (62.8 us vs ~38 us floor): the verify path runs the cute-dsl split-KV kernel;
   lever A1 (TRT-LLM-gen for non-DCP) is not in this tree -> try it for verify (iter_01).
2. dense GEMMs at m=256 (splitK reduce anomaly) — next.
MoE: at roofline, skip.
