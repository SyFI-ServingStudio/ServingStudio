# MLA trial 7: agent iterations

## iter_01
### hypothesis.md

# Hypothesis

VibeSim selected `unified.mla.moe.mxfp4_fused_moe` as the largest candidate: at
the B=128/L=8192 shape its R0 time was 527.255 us, with a 43.2% necessary
share, while the profiler showed the two routed MXFP4 BMMs at 284.017 us and
142.982 us. The source path is `KimiK3MoE._forward_fused` in
`python/sglang/srt/models/kimi_k3.py`.

The fused-front path produces `shared_output` and `latent` independently after
the same front GEMM. The old TP1 path ran `_forward_shared` and
`_forward_routed` serially before the existing flat collective. The change
launches only the shared branch on the MoE's dedicated side stream, keeps the
routed branch and all routing/expert operations on the current stream, and
joins before the collective. Each branch's arithmetic and the final
addition/normalization order are unchanged, so the layer output, KV write, and
post-step state should remain within the required tolerance.
