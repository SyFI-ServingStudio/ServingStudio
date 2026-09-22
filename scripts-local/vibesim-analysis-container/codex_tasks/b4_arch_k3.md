# Task: the Kimi-K3 simulator arch recipe `kimi_k3_sglang` (branch `kimi-k3`)

VibeSim repository, git branch `kimi-k3` (do not switch). CPU only (`CUDA_VISIBLE_DEVICES`
empty). Use `uv run ...` for Python and `cargo` for Rust (`just test-cpu` = the CPU test
gate). Follow the repo's skills and say in your report which steps of each you executed:
`skills/top-add-new-arch/SKILL.md` (Phases 2-4; Phase 1's per-op decision table is given
below), `skills/impl-compose-op/SKILL.md`, `skills/impl-compose-worklet/SKILL.md`,
`skills/impl-compose-arch/SKILL.md`, `skills/impl-wire-new-arch/SKILL.md`,
`skills/impl-add-model-work-label/SKILL.md` (only its location-map part — the `model.work`
K3 label itself already exists on this branch: `model/work/models/kimi_k3.py`, read it and
keep its semantic names). Comments in English. Commit on `kimi-k3` in logical steps; every
commit message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
Stop and write a clear question into your report (do not guess silently) whenever a design
decision is not covered here.

## What exists on this branch that you must build on
- `model/config/kimi_k3.json` — verbatim HF config (LM under `text_config`).
- `model/work/models/kimi_k3.py` (+ `attention/kda.py`, `attention/mla.py`, MoE changes,
  MXFP4 quant) — the necessary-work label; its `LayerStack` tags are the semantic prefixes
  your location map must consume exactly once each.
- Profiling kinds/backends for the K3 kernels (see `git log` on this branch and
  `profiling/kernels/{kda_recurrent_decode,mla_decode_attention,mxfp4_fused_moe}.py`, the
  `sglang_triton`/`sglang_k3_absorb`/`sglang_cutedsl_mla`/`sglang_trtllm_mla` backends, env
  `sglang_k3_env`) and their Rust twins under `simulator/src/timing/kernels/`.
- Measured per-kernel tables from the real layer (what actually launches per decode step):
  `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/*.json`.

## Templates
Hybrid recipe: `simulator/src/arch/qwen36_local.rs` + `simulator/src/worklet/qwen36_gdn_local.rs`
+ `simulator/src/op/ssm/gdn_decode.rs`. DP/EP shape + per-layer recipe table:
`simulator/src/arch/deepseek_v4_vllm.rs` (`LAYER_RECIPES`, `normalize_input` with
`groups.len()==EP`, `gpus_per_replica/num_attn_dp_groups`). sglang-on-B200 backend dictionary:
`simulator/src/arch/glm52_sglang_nvfp4_tp_dsa_moe.rs:85-106`. MoE worklet with EP split:
`simulator/src/worklet/nvfp4_moe_local.rs` (`split_for_ep`). Absorbed-MLA worklet (minus the
DSA indexer): `simulator/src/worklet/sglang_glm52_dsa_attn_local.rs`.

## The model (decode-first; prefill worklets may be minimal but must exist and be honest)
93 layers: layer 0 dense MLP (intermediate 33792, situ act), layers 1..92 MoE. Attention per
`text_config.linear_attn_config`: `full_attn_layers` (24, 1-based [4,8,...,92,93]) = MLA,
`kda_layers` (69) = KDA — use the explicit lists from the config, NOT a fixed cadence.
Per-GPU production shape on B200 (the sglang cookbook recipe): **attention TP8** (12 of 96
heads per GPU; MLA latent KV replicated across the 8 TP ranks unless DCP), **EP8** for the
routed experts (112 of 896 local), **PP2** across the two 8-GPU nodes (each stage ~46
layers). Model `Parallel { attn_tp_size, ep_size, pp_size, dcp_size(=1 for v1), gpu_name }`
with TWO topologies that must both build: (a) production `attn_tp=8, ep=8, pp=2`;
(b) **single-rank alignment topology** `attn_tp=1 (12 heads via a `heads_per_rank` override),
ep=1 with `local_experts=112`, pp=1` — this is what our single-GPU driver measures, so a
layer-level prediction can be aligned against it. Also support `sim_kda_layers` /
`sim_mla_layers` overrides (predict ONE KDA or ONE MLA layer + nothing else; qwen36's
`sim_num_layers` pattern, `qwen36_local.rs:323-334`) — the alignment probe is one layer.
Communication in v1: attention-TP all-reduce of o_proj and the MoE EP all-to-all are
modelled as ZERO-time leaves (named, so they can be filled later) — say so in the code.

### Per-op decision table (Phase 1 output — fixed)
KDA layer decode (12 heads/rank): `single_gemm` fused qkvbfg projection 7168 -> (3*12*128 + 2*12*128 + 12 [+pad])
(the fused `qkvbfg_a_proj` sglang launches with use_full_rank_gate) | `gdn_causal_conv_decode:sglang_triton`
(channels 3*12*128=4608, k=4) | `kda_recurrent_decode:sglang_fused` (the fused KDA decode kernel
sglang runs at 12 heads: conv+recurrence+gated-norm; fall back to `sglang_triton` recurrent +
`gdn_gated_rms_norm:sglang_triton` if the fused kind was registered as separate leaves — match
what the branch's kinds provide) | recurrent state read+write per request 12x128x128 x state
bytes (bf16 in the cookbook recipe) — fold into the recurrent kernel's bytes as the kind
defines | `single_gemm` o_proj 12*128 -> 7168 | zero-time attn all-reduce leaf.
MLA layer decode: `single_gemm:sglang_fused_a_auto` fused_qkv_a 7168 -> 2112 | `rms_norm` q_a
(1536) and kv_a (512) | `single_gemm` q_b 1536 -> 12*192 | `batched_gemm:sglang_k3_absorb`
q_nope@w_kc (12 x [m,128]x[128,512]) | `mla_cache_append:sglang_cuda` (576, fp8 in the recipe) |
`mla_decode_attention:sglang_cutedsl_mla` (12 heads, kv fp8_e4m3, page 64) | `batched_gemm`
@w_vc (12 x [m,512]x[512,128]) | `single_gemm` output gate 7168 -> 12*128 + `elementwise`
sigmoid-mul | `single_gemm` o_proj 12*128 -> 7168 | zero-time attn all-reduce leaf.
MoE (every layer 1..92): `single_gemm` merged front [7168 -> 12288 (2 shared gate_up) + 896
(router) + 3584 (latent down)] — ONE GEMM, as sglang's `_merge_front_weights` fuses them
(router logits fp32 via `gemm_fp32_output:sglang_router_auto` only if the branch models the
router separately — pick one and state it) | routing: folded into `mxfp4_fused_moe` |
`elementwise` situ on the shared gate_up | `single_gemm` shared down 6144 -> 7168 |
`mxfp4_fused_moe:flashinfer_trtllm_sm100_sglang` (hidden 3584, inter 3072, 896/112 experts,
top-16) | `rms_norm` routed norm (3584) | `single_gemm` latent up 3584 -> 7168 |
`elementwise` add3 (routed + shared + residual) | zero-time EP a2a leaf.
Layer glue: `residual_rms_norm` input/post-attention norms (7168). Model: embedding gather
(`elementwise`), final `rms_norm`, lm_head `single_gemm` 7168 -> 163840 (per PP stage only
where they live).

## Deliverables
1. Worklets `kimi_k3_kda_local`, `kimi_k3_mla_local`, `kimi_k3_moe_local`, `kimi_k3_dense_local`
   (+ head/embedding reuse), arch `kimi_k3_sglang` with the recipe table and both topologies,
   arms in `arch/config.rs` (`IterArchSel::KimiK3Sglang`, ProviderSchema), `arch/mod.rs`,
   `arch/build.rs`, `deployment/unified.rs`.
2. Presets: `presets/predict_kimi_k3_b200_sglang_tp8ep8pp2.json` (+ cases: decode B in
   {1,32,128} at kv 8192, B=1 at 1048576, B=16 at 65536; the number of groups must equal
   `num_attn_dp_groups()` of the topology — read `simulator/src/timing_predict.rs:171-249`) and
   `presets/predict_kimi_k3_b200_rank1_layer_{kda,mla}.json` (single-rank, single-layer).
3. Location map `model/work/location_maps/kimi_k3_sglang_unified.json` (every non-communication
   leaf listed, every `model.work` semantic consumed once — `analyzer/rust/src/optimality/location.rs`
   rules; run whatever validator the skill names).
4. Tests in the style of `qwen36_local.rs:1041-1615` (slot/leaf-name freeze, resolved shapes,
   normalize_input, KV/state bytes: MLA 576 x elem_bytes x 24 layers per token; KDA state
   12x128x128 x elem + conv (k-1)x4608 x 2 per request per KDA layer) and `just test-cpu` green.
5. `uv run python -m launcher timing-predict presets/predict_kimi_k3_b200_rank1_layer_kda.json`
   must reach the kernel-cache lookup stage and fail ONLY with cache misses (rows come later
   on GPU) — paste the miss list in your report. If it fails earlier, fix it.

## Report
Files, test results, the timing-predict miss list, and every open design question.
