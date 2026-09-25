# MLA trial 22: agent iterations

## iter_01
### hypothesis.md

# Hypothesis

VibeSim identified `unified.mla.attention.mla_decode_attention` as the largest
primary-point actionable node and reported a cached TRT-LLM alternative. The
installed FlashInfer API also exposes a dedicated `use_fp16_softmax` TRT-LLM
MLA cubin for K3's 576-wide query / 512-wide value shape.

The change is limited to FP8-KV TRT-LLM K3 decode at `max_seq_len >= 65536`,
covering B=1/L=1,048,576 and B=16/L=65,536 while leaving B=128/L=8,192 on
the baseline path. It changes only the attention accumulator implementation;
KV-cache append, routing, projections, MoE, and post-step state writes are
unchanged. The replay CHECK is the numerical and state proof for the accepted
tolerance.

### analysis.md



### result.json

```

```

## iter_02
### hypothesis.md

# Hypothesis

The profiler shows the required uniform decode points launching the
`fmhaSm100fKernel_*VarSeq*` TRT-LLM MLA kernel. The driver creates one common
context length for every request in each point, and FlashInfer's fixed-length
mode selects its persistent scheduler (`is_var_seq=False`), which should
reduce scheduler/launch overhead without changing the attention equation,
KV-cache append, or any post-step state.

This first test intentionally uses the direct fixed-mode selector to measure
the kernel choice and numerical behavior on the exact workload. A successful
result will be followed by a source guard based on uniform CPU sequence
metadata so production batches with genuinely different lengths retain the
variable-length path.

### analysis.md



### result.json

```

```

## iter_03
### hypothesis.md

# Hypothesis

VibeSim reports a cached alternative for the `mla_decode_attention` leaf:
`sglang_cutedsl_mla` and `sglang_trtllm_mla`. The current checkout's ordinary
decode path selects TRT-LLM Gen, so this iteration directly measures the other
cached B200 implementation. The two implementations consume the same packed
FP8 KV pages and query, and only the attention kernel changes; KV writes,
projections, MoE routing/activation, and all post-step state are unchanged.

The global switch is an exploratory measurement. If it wins only for a subset
of the three shapes, the final source will select CuTeDSL only for that shape
and retain TRT-LLM Gen elsewhere.

### analysis.md



### result.json

```

```

## iter_04
### hypothesis.md

# Hypothesis

The initial kernel table measured the fused FP8 KV append/query-concat launch
as `set_mla_kv_concat_q_fp8_kernel<8>` at roughly 5.7 us for B=1 and B=16.
VibeSim's `mla_cache_append` ladder has a very small necessary share, so this
is launch/scheduling overhead rather than useful arithmetic. The 4-warp
specialization uses the same loads, stores, conversion, and PDL ordering with
half as many resident threads for the tiny grids; B=128 keeps the 8-warp
choice. No attention values or state layout change.

### analysis.md



### result.json

```

```

## iter_05
### hypothesis.md

# Hypothesis

VibeSim kept the MLA attention node as the largest actionable primary-point
operator after the TRT-LLM backend choice. The actual K3 source shows that
the q-B projection shape `(2304, 1536)` is eligible for the existing
low-latency fused-A GEMM, but the shape allowlist excludes it. The preceding
K3 fused A projection already launches `fused_a_gemm_kernel`, while q-B is
observed as generic NVJet/CUDA GEMM work.

Adding only this exact K3 shape to the existing guard changes the GEMM
implementation, not its BF16 inputs/weights, FP32 accumulation contract, or
output layout. The attention cache append, attention kernel, MoE path, and
post-step state are unchanged; replay CHECK must establish numerical/state
equivalence.

### analysis.md



### result.json

```

```

## iter_06
### hypothesis.md

## Hypothesis

The measured B=1 long-context critical leaf is the TRT-LLM FP8 MLA FMHA kernel at about 130 us. The installed native FlashMLA implementation supports the same 12-head, 576-dimensional query and paged FP8 KV layout and uses the same absorbed MLA dot-product and value dimensions. Route ordinary non-DCP decode through it while retaining the existing TRT-LLM/CuTeDSL path for DCP. The output remains the same BF16 attention result and the KV write is unchanged; the golden replay will verify both output and post-step state.

## iter_07
### hypothesis.md

## Hypothesis

The profile launches an FP8-Q/FP8-KV MLA kernel after a separate query quantization step. FlashInfer's SM100 cubin table contains a native BF16-Q/FP8-KV Kimi-K3 kernel for the same 576 query and 512 value dimensions. Keep the query in BF16 for this exact K3 decode shape, while converting and storing K/V exactly as before. The attention result may differ only by the removal of query quantization; replay against the original golden must confirm the driver's 2% numerical tolerance and unchanged KV/post-step state.

## iter_08
### hypothesis.md

## Hypothesis

The B=1 profile's 38 us merged MoE front is the FP32-output TGV GEMM. A direct GPU benchmark of the existing exact-accumulation tactics for `(M,N,K)=(1,15984,7168)` measured tactic 18 at about 35 us versus the current heuristic's tactic 12 at about 36-37 us. Select tactic 18 only for that exact decode shape. It changes CTA tiling and scheduling, not the FP32 accumulation or output layout, so the router and post-step state should remain unchanged. B=16 continues through the previous selector.
