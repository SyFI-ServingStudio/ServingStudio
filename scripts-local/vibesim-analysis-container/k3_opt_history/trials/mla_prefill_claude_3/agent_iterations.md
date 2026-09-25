# MLA_PREFILL_CLAUDE trial 3: agent iterations

## iter_00
### hypothesis.md

# iter_00 hypothesis
Baseline only; no hypothesis tested. Prior levers already in the seed tree: mla_prefill_claude_1 (fp8 K/V pack, attn_res fused add, kv_indices grid split).

### analysis.md

# iter_00 -- baseline (seed tree, no source change)

Plain-flag latency_us (seed tree): pf49152 11265 / 16384,pf 8298 / 4x4096,pf 7761
(capture run: 11246 / 8320 / 7778). profile.json = profile_B1_L16384pf49152.json.

## Kernel table, 1,16384,pf49152 (us/step)
| kernel | us |
|---|---|
| cute-dsl prefix FMHA (BlackwellFusedMultiHeadAttentionForward, non-persistent) | 3300-3444 |
| merged front fp32 GEMM (nvjet tss) | 2283 |
| 6x nvjet tst GEMMs | 2073 |
| MoE bmm1 (mxfp4) | 874 |
| causal FMHA (trtllm-gen fp8 cubin, persistent) | 696 |
| attn_res x2 | 365 |
| MoE bmm2 | 328 |
| g_proj nvjet | 240 |
| situ_and_mul | 231 |
| add3 | 133 |
| _v1_flat pack x2 | 102 |
| finalize | 69 |
| fp8 quant | 48 |

Timeline: the GPU is idle ~46 us per step, almost all of it from a host sync
(`.item()` DtoH) right before the causal FMHA. The shared/routed MoE streams barely
overlap because both are compute-bound.

## VibeSim (prediction .:k3_mla_prefill; opt.json / analyze_*.json)
Sum over the 3 cases 47.0 ms, optimality 0.085. R6 (necessary_share) is null everywhere.
| node | R0 ms | R5 ms |
|---|---|---|
| merged_front_prefill | 16.8 | 5.0 |
| attention + output gate | 7.3 | 1.14 |
| mxfp4_fused_moe_prefill | 8.58 | 2.89 |
| shared_down | 6.44 | 1.92 |
| latent_up | 3.54 | 1.12 |
| mla_prefill_attention_prefix | 3.53 | 1.37 |
| mla_prefill_attention_causal | 1.54 | 0.515 |
attn_res is at R0/R5 = 1.0 (nothing left there).

## Prefix FMHA diagnosis
Grid = 64 q-tiles (256 rows/CTA) x 12 heads = 768 CTAs, 1 CTA/SM on 148 SMs, i.e.
5.19 waves; the last wave holds only 28 CTAs, so ~13% of the kernel is tail with
120 SMs idle. Microbenchmarks (tools/mb_fmha*.py): persistent / TMA-store knobs gave
3180-3256 us (noise); ex2-emulation count 0/20/32/48/64 gave 3233/3204/3093/3218/3376 us
(not MUFU-bound, non-exact) -> dropped. Overlapping the causal pass on a side stream
with the prefix pass (tools/mb_overlap.py): serial 4013-4207 us vs overlapped
3681-3771 us, bit-exact -> iter_01.

### result.json

```
{"iter":0,"change":"none (seed baseline)","latency_us":{"1,16384,pf49152":11265,"1,16384,pf":8298,"4,4096,pf":7761},"check_pass":{"1,16384,pf49152":true,"1,16384,pf":true,"4,4096,pf":true}}
```

(diff.patch: 0 lines, files: )

## iter_01
### hypothesis.md

# iter_01 hypothesis -- overlap the causal FMHA with the prefix FMHA's last wave

Node: unified.mla.attention.mla_prefill_attention_{prefix,causal} (+ the host sync before causal).

1. The non-persistent cute-dsl prefix FMHA runs 768 CTAs at 1 CTA/SM on 148 SMs (5.19 waves).
   Its last wave holds 28 CTAs, so ~120 SMs sit idle for ~1/6 of the kernel (~500 us).
   If the prefix grid is launched FIRST and the causal pass (persistent trtllm-gen cubin,
   ~700 us, independent inputs) is queued right behind it on the layer's alt stream,
   the causal CTAs backfill those idle SMs. Both passes are the same kernels with the
   same inputs, and merge_state gets the same arguments, so the output is bit-identical.
   Microbenchmark (tools/mb_overlap.py): serial 4013-4207 us vs overlapped 3681-3771 us,
   bit-exact. Expected step gain: 250-450 us at 1,16384,pf49152 (only point with a prefix).
2. Enabler: flashinfer's trtllm_ragged_attention_deepseek, without host lengths,
   probes for empty rows on the GPU with `.any().item()`. That blocks the host until the
   prefix kernel drains, so the causal launch can never land inside the prefix tail
   (confirmed in trace: causal started 2.4 ms after the prefix ended). The backend
   already has extend_seq_lens_cpu. Passing q/kv_seq_lens_cpu makes the check host-only;
   with every row active flashinfer launches the identical kernel. This also removes
   the ~46 us idle gap seen in iter_00 on all three points.

Safety: overlap only when the prefix actually ran on cute-dsl (no shared
workspace_buffer with the trtllm-gen causal kernel); if cute-dsl falls back at runtime,
the causal stream waits for the prefix pass. Only single-prefix-chunk batches take the path.

Prior trials:
- Builds on mla_prefill_claude_1 (fp8 K/V pack, iter_01) -- the fp8 prefix path this reorders.
- Builds on A4 / mla_24 (alt-stream overlap of shared vs routed MoE): same alt-stream pattern,
  but here one branch is tail-bound rather than compute-bound, which is why it pays off.
- Builds on kda_prefill_claude_1 iter_02 (drop a prefill host sync by reading the host copy
  of the lengths; bit-exact).
- Related to mla_b512_claude_2's tail-split idea (decode): that trial attacked wave
  quantization by splitting work; this one fills the tail with independent work instead.
- No prior trial overlapped the causal and prefix FMHAs.

(diff.patch: 200 lines, files: variants/new/python/sglang/srt/layers/attention/trtllm_mla_backend.py, variants/new/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py)
