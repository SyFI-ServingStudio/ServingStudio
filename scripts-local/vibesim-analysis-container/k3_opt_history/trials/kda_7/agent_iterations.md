# KDA trial 7: agent iterations

## agent log.md

iter_00 baseline: VibeSim p_38ccb83a6c4e44298f45b2251bd4cf54 ranked mxfp4_fused_moe first; clean graph 556.512/310.720/179.616 us (B128/B32/B1), golden captured.
iter_01 rejected: K3 BF16-activation MXFP4 experiment had no TRT-LLM SiTU tactic at B=1; source rolled back and smoke passed.
iter_02 rejected: FlashInfer recurrent_kda matched output/state but regressed graph latency to 591.168/312.704/179.616 us; source rolled back.
iter_03 rejected: disabling TRT-LLM PDL was bit-exact but regressed graph latency to 565.920/318.784/185.792 us; source rolled back.

## iter_00
### hypothesis.md

Baseline only. No source change was made in iteration 00.

### analysis.md

Iteration 00 baseline

VibeSim build: simulate?prediction=.:before
Prediction id: p_38ccb83a6c4e44298f45b2251bd4cf54

Analysis interpretation from workspace-info:
- R0 is measured leaf GPU time.
- R5 is the hardware-limit estimate.
- R6 is the segmented necessary-work estimate.
- R7 is the scope-fused necessary-work estimate.
- A large R0/R5 with nonzero R6 indicates an inefficient implementation of necessary work;
  a large R0-R6 gap is batching/launch/scheduling headroom.

The iteration analysis predicted 542 us for B=128, 303 us for B=32, and 161 us for B=1.
Operator analysis total kernel time was 1.006 ms in the fixed prediction. The operator
breakdown put mxfp4_fused_moe first at 542.877 us / 53.96%, followed by
qkvbfg_a_proj at 144.538 us / 14.37% and merged_front at 120.183 us / 11.95%.

Optimality scope=iter ranked these leaves:
- unified.kda.moe.mxfp4_fused_moe: R0=542.878 us, R5=9.836 us, R6=231.980 us,
  R7=231.980 us, R0/R5=55.19, necessary_share=0.4273.
- unified.kda.attention.qkvbfg_a_proj: R0=144.538 us, R5=41.950 us,
  R6=11.310 us, necessary_share=0.0782.
- unified.kda.attention.kda_recurrent_decode: R0=42.495 us, R5=8.161 us,
  R6=15.912 us, necessary_share=0.3744.

The per-leaf kernel drill-down reported no cached alternative for mxfp4_fused_moe,
while the smaller GEMM leaves had cached alternatives. The measured B=128 table
confirmed the selected MoE leaf is the pair of TRT-LLM BMMs at 292.261 us and
147.143 us. Other relevant launches were the router, activation quantizer, and
finalize kernels. The source mapping is Mxfp4MoEMethod._apply_sm100_trtllm_gen()
in python/sglang/srt/layers/quantization/mxfp4.py, calling the TRT-LLM runner in
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py.

Clean graph baseline (plain fixed command): B=128 556.512 us; B=32 310.720 us;
B=1 179.616 us. The diagnostic profile values are intentionally not used as the
timing reference.

### result.json

```
{
  "iteration": 0,
  "mode": "baseline",
  "latency_us": {
    "128,8192": 556.5119981765747,
    "32,8192": 310.7199966907501,
    "1,8192": 179.61600422859192
  },
  "checks": "golden captured; replay check pending after an edit"
}
```

## iter_01
### hypothesis.md

Target: unified.kda.moe.mxfp4_fused_moe.

VibeSim ranked this leaf first at R0=542.878 us with necessary_share=0.4273 and
R0/R5=55.19. The real B=128 profile showed the TRT-LLM MXFP4 BMM pair consuming
439.403 us, plus separate activation quantization work before it.

Change Mxfp4MoEMethod so only the K3 SM100 per-GPU expert shape (112 local
experts, hidden 3584, intermediate 3072) selects the existing "bf16" activation
mode. This preserves the same MXFP4 packed weights, scales, routing, top-k,
SiTU activation, output buffer, and expert accumulation. It removes only the
MXFP8 activation quantization and supplies the same BF16 routed input to the
TRT-LLM MXFP4 operator. KDA recurrence and its BF16 state are not on this path.

The golden replay is the acceptance test because BF16 activation mode can change
the MoE output within the allowed tolerance; any output/state failure rejects
this hypothesis even if the BMM timing improves.

Result: rejected before replay. The B=1 smoke reached the TRT-LLM runner and
failed with no tactic for MXFP4 weights x BF16 activations for the routed SiTU
configuration. The source change was removed; the post-rollback smoke passed.

### result.json

```
{
  "iteration": 1,
  "status": "rejected",
  "reason": "TRT-LLM SM100 has no MXFP4-weight x BF16-activation SiTU tactic for the required shape",
  "smoke_after_rollback": {
    "point": "1,8192",
    "latency_us": 199.8399943113327,
    "latency_mode": "graph",
    "pass": true
  },
  "replay": "not run because the candidate failed the smoke path"
}
```

(diff.patch: 0 lines, files: )

## iter_02
### hypothesis.md

Target: unified.kda.attention.kda_recurrent_decode.

VibeSim ranked the recurrent leaf at R0=42.495 us, R5=8.161 us, R6=15.912 us,
and necessary_share=0.3744, with no cached alternative. The profile showed the
Triton fused recurrent kernel at 29.270 us (B=128), plus separate convolution
and gated output norm launches.

The source already contains FlashInfer's SM100 ``recurrent_kda`` wrapper. It
operates directly on the BF16 ``[slots, heads, 128, 128]`` state pool and
implements the same per-K decay, sigmoid beta, Q/K L2 normalization, state
update, and K3 lower-bound gate. The direct layer harness leaves the generic
decode setting at Triton, so this iteration makes the KDA dispatcher select
FlashInfer opportunistically on SM100 and keeps Triton on unavailable builds.
Only the recurrent implementation changes; convolution, projections, output
norm, MoE routing/weights, and state tensor ownership stay the same. The
golden replay must verify both output and post-step state on all points.

Result: rejected. Replay correctness passed at all points, including
``state_ok: true``, but graph latency changed to 591.168/312.704/179.616 us
for B=128/B=32/B=1. The primary point regressed by 6.2%, so the dispatcher
change was removed.

### result.json

```
{
  "iteration": 2,
  "status": "rejected",
  "latency_us": {
    "128,8192": 591.1679863929749,
    "32,8192": 312.7039968967438,
    "1,8192": 179.61600422859192
  },
  "checks": {
    "128,8192": {"max_rel_err": 0.015277690864691969, "state_ok": true, "pass": true},
    "32,8192": {"max_rel_err": 0.015151421120877407, "state_ok": true, "pass": true},
    "1,8192": {"max_rel_err": 0.010416592593119337, "state_ok": true, "pass": true}
  },
  "reason": "FlashInfer recurrent_kda is correct but slower at the scored batch size"
}
```

(diff.patch: 0 lines, files: )

## iter_03
### hypothesis.md

Target: unified.kda.moe.mxfp4_fused_moe scheduling.

VibeSim's optimality ladder showed a large R0-to-R6 scheduling/batching gap for
the dominant MoE leaf, and the actual TRT-LLM BMM names contain ``schPd``. This
iteration disables only the PDL launch attribute passed to the TRT-LLM MoE
operator. It does not change any inputs, weights, tactic dimensions, routing,
activation, arithmetic, output buffer, or state update, so output and state
should be identical. The hypothesis is that ordinary graph launch ordering is
lower overhead for the static B=128 replay. The all-point golden replay is the
acceptance test.

Result: rejected. The PDL-off replay was bit-exact with ``state_ok: true`` but
measured 565.920/318.784/185.792 us for B=128/B=32/B=1, all slower than the
clean baseline. The source was restored.

### result.json

```
{
  "iteration": 3,
  "status": "rejected",
  "latency_us": {
    "128,8192": 565.9199953079224,
    "32,8192": 318.7839984893799,
    "1,8192": 185.7919991016388
  },
  "checks": {"all_points": {"max_rel_err": 0.0, "state_ok": true, "pass": true}},
  "reason": "PDL-off scheduling regressed every graph replay point"
}
```

(diff.patch: 0 lines, files: )

## iter_04
### hypothesis.md

Target: unified.kda.moe.merged_front and its handoff into mxfp4_fused_moe.

The merged front is the next VibeSim-ranked leaf after the dominant MoE BMM:
R0=120.183 us, R5=91.108 us, R6=30.048 us, necessary_share=0.2500. The source
currently emits the entire [gate_up, router_logits, routed_input] front as FP32
to preserve exact router logits. For K3's SM100 MXFP4 path, the activation and
expert inputs are consumed as BF16/MXFP8 and the router's BF16-to-FP32 handling
is already explicit. Select BF16 front output only for that exact path, which
reduces front output traffic and avoids carrying FP32 routed activations into
the MoE handoff. Weights, routing algorithm, expert arithmetic, KDA state, and
output buffers are otherwise unchanged; replay checks whether top-k and output
remain within tolerance.

(diff.patch: 25 lines, files: python/sglang/srt/models/kimi_k3.py)

## iter_05
### hypothesis.md

Target: unified.kda.moe.mxfp4_fused_moe, the dominant VibeSim leaf.

The B=128 MXFP4/TRT-LLM expert node has R0=542.878 us, R5=9.836 us, R6=231.980 us,
and necessary_share=0.4273. Its current tactic is tuned with max_num_tokens=128,
exactly the live token count. Retuning this exact K3 shape with a 256-token ceiling
may select a better SM100 tile/scheduling tactic for the large expert workload.
The B=32 and B=1 ceilings remain unchanged. This affects only FlashInfer's kernel
selection metadata; tensor values, routing, weights, state updates, and output
buffers are unchanged, so replay must remain numerically identical.
