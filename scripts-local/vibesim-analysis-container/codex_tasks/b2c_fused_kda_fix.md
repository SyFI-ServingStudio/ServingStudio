# Task: fix the `kda_fused_decode:sglang_fused` profiling runner (VibeSim repo, branch `kimi-k3`)

You are in the VibeSim repository on branch `kimi-k3` (do not switch). CPU only, `uv run --no-sync`. Touch ONLY
`profiling/runners/attention/kda_fused_decode.py` (+ its test in `tests/test_kimi_k3_profiling.py` if needed);
`git add` only those; commit with trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
Another agent works in a separate worktree; do not touch `simulator/` or `model/`.

## The failure (GPU smoke on B200, slurm 888)
`uv run python -m launcher kernel-profile run kda_fused_decode --backend sglang_fused --spec
'{"batch_size":128,"num_heads":12,"head_k_dim":128,"head_v_dim":128,"dtype":"bf16","state_dtype":"bf16","lower_bound":-5.0}'`
→ `Tensor match failed for Tensor<1536>[strides=<1>, dtype=float32] at
/sgl-workspace/sglang/python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh:974 - Size mismatch for
shape#0: expected 128 but got 1536`. The runner hands a 1536-element fp32 tensor where the kernel expects a
128-element one — an argument order / layout mismatch against the real call site.

## Ground truth to match exactly
- Call site: `sglang/srt/layers/attention/linear/kda_backend.py` (`KDAAttnBackend.forward_decode`, the
  `kda_fused_decode` branch around lines 663-712) and the stash it consumes, built by
  `KimiK3DeltaAttention._prepare_fused_decode` (`sglang/srt/models/kimi_k3.py` ~1870-1912): the stash is
  `(conv_w_q[4,seg], conv_w_k[4,seg], conv_w_v[4,seg], conv_bias[3*seg] fp32, A_log[12] fp32 (reshape(-1) of
  A_log), o_norm_weight[128] fp32, o_norm_eps)`, seg = 12*128 = 1536, conv weights transposed fp32 contiguous
  slices of `conv_weights.t()`; dt_bias is `[seg]` fp32; recurrent state `[slots, 12, 128, 128]` fp32 or bf16;
  conv state `[slots, 3, 3*seg]` bf16; `mixed_qkv [B, 3*seg]` bf16, `a`(forget-gate input, `[B, seg]` bf16),
  `b` (`[B, 12]` bf16), `lower_bound=-5.0`. Read those files inside the image:
  `docker run --rm --entrypoint bash lmsysorg/sglang:v0.5.20 -c 'sed -n 640,720p /sgl-workspace/sglang/python/sglang/srt/layers/attention/linear/kda_backend.py; grep -n "def kda_fused_decode\|def forward\|Tensor" /sgl-workspace/sglang/python/sglang/kernels/ops/attention/kda_fused_decode/__init__.py | head -60'`
  and the kernel's parameter contract in `sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh` (around
  line 974 and the function signature) to see which parameter is checked for 128 elements (it is the
  o_norm weight; make sure you are not passing dt_bias or conv_bias there) and the exact expected shapes of
  every tensor. Do not guess; quote the signature in a comment.
- The driver that exercises the production path end-to-end for reference:
  `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer_decode.py`
  (search `_prepare_fused_decode`, `KDAState`).

## Deliverable
Fixed runner; `uv run --no-sync pytest tests/test_kimi_k3_profiling.py -q` green; commit; report the exact
argument list you now pass (name, shape, dtype) next to the kernel signature.
