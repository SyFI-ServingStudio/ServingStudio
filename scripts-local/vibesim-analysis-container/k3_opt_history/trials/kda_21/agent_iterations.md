# KDA trial 21: agent iterations

## iter_02
### hypothesis.md

The fixed workload has 112 local experts and top-2 routing.  The existing
route+quant handoff is only attempted inside the large 896-expert/top-16
router branch, so this shape launches the Triton router and MXFP8 quantizer
separately.  The per-kernel profile shows those preparation launches on the
layer path, while VibeSim identifies the routed MXFP4 leaf as the dominant
operator.

Extend the handoff with a small CUDA route+quant specialization for N=112,
K=2 and invoke it before the generic small-router fallback.  The kernel keeps
fp32 sigmoid scores, correction-bias ranking, fp32 renormalized weights, and
the existing QuantTrait MXFP8 conversion; it only combines the launches.  The
original path remains the fallback for every other shape, and B=1/32/128 all
use the same guarded specialization.  State is not touched, and the packed
routing values are produced from the same bf16-rounded weights consumed by
the existing FlashInfer handoff.
