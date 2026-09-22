# Task: add the Kimi-K3 "necessary work" label to `model/work` (VibeSim repo, branch `kimi-k3`)

You are in the VibeSim repository (this directory), on git branch `kimi-k3`. Do NOT switch
branches. Do NOT use any GPU (`CUDA_VISIBLE_DEVICES` is empty on purpose; everything here is
CPU-only Python). Run Python through `uv run ...` (the repo's `.venv`). Follow the repo's own
skill for this job: read `skills/impl-add-model-work-label/SKILL.md` first and follow it,
including its rules about deriving the architecture from the TRUE model code/config (not from
a shrunk experiment config) and adding hand-derived goldens. Write code comments in English.

## Goal
Register **Kimi-K3** (`moonshotai/Kimi-K3`) in the independent FLOP/byte accountant so that
```
uv run python -m model.work model/config/kimi_k3.json --decode 128x8192 --gpu B200 --json
uv run python -m model.work model/config/kimi_k3.json --decode 1x1048576 --gpu B200 --json
uv run python -m model.work model/config/kimi_k3.json --prefill 8192@0 --gpu B200 --json
```
all run green, and `uv run pytest tests/test_model_work.py -q` passes with new K3 goldens.

## Inputs you have
- `model/config/kimi_k3.json` — the verbatim HF `config.json` of moonshotai/Kimi-K3 (already
  placed; do not "simplify" it). Note it is a VL-style wrapper: the LM config is under
  `text_config`; `architectures` = `["KimiLinearForCausalLM"]`, `model_type` = `kimi_linear`.
  Register BOTH `KimiLinearForCausalLM` and `KimiK3ForConditionalGeneration` (sglang's entry
  class names) in `model/work/registry.py`.
- The sglang v0.5.20 model code for K3 is readable at
  `/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/k3_src_v0520/`
  (`models_kimi_k3.py` = `sglang/srt/models/kimi_k3.py`, `configs_kimi_linear.py`,
  `deepseek_v2.py` for the MLA absorbed decode path, `layers_attention_linear_kda_backend.py`
  and `_attn_dir/linear/kernels/kda_triton.py` for the KDA decode chain). Inventory the matmuls
  and state traffic from THAT code, not from memory of similar models.

## Architecture facts to encode (verify each against the sources above)
- 93 decoder layers. Layer 0 (index 0) is a DENSE MLP layer (`first_k_dense_replace=1`,
  `intermediate_size=33792`, SiTU/gated activation); layers 1..92 are MoE (`moe_layer_freq=1`).
- Attention is HYBRID per `text_config.linear_attn_config`: `full_attn_layers` (24 layers,
  1-based indices [4,8,...,92,93]) are dense **MLA**; `kda_layers` (69) are **KDA** (Kimi Delta
  Attention, a gated-deltanet linear attention). Use the explicit lists, not a fixed cadence.
- **KDA layer** (`KimiK3DeltaAttention`): 96 heads x head_dim 128 (`linear_attn_config`),
  projections from hidden 7168: q/k/v (3 x 96*128 = 36864 total), beta `b_proj` (96), and with
  `use_full_rank_gate: true` the forget gate `f` and output gate `g` are FULL-RANK
  7168 -> 96*128 each (no low-rank 128 bottleneck); short causal conv1d (kernel 4) over the
  q/k/v channels with a per-request conv state of (kernel-1) x 36864; the recurrent
  delta-rule state is per request 96 x 128 x 128 (fp32 by default; bf16 with
  `--mamba-ssm-dtype bfloat16`) read AND written every decode step; `FusedRMSNormGated`
  output norm (per head, 128); `o_proj` 96*128 -> 7168. Decode FLOPs of the recurrent step
  are O(heads*d_k*d_v) per token. Prefill uses the chunked delta-rule path.
- **MLA layer** (`KimiK3MLAAttention`, DeepSeek-V3 style, absorbed decode): 96 heads,
  `q_lora_rank 1536`, `kv_lora_rank 512`, `qk_nope_head_dim 128`, `qk_rope_head_dim 64`,
  `v_head_dim 128`. Decode: `fused_qkv_a_proj_with_mqa` 7168 -> 1536+576, `q_a_layernorm`
  (1536) and `kv_a_layernorm` (512), `q_b_proj` 1536 -> 96*192, absorb `q_nope @ w_kc`
  (per head [128 x 512]), attention over the latent KV cache of 576 (=512+64) per token per
  layer (bf16 by default; fp8_e4m3 with `--kv-cache-dtype fp8_e4m3`), `attn_out @ w_vc` (per
  head [512 x 128]), `o_proj` 96*128 -> 7168, and with `mla_use_output_gate: true` an extra
  gate projection 7168 -> 96*128 + sigmoid multiply on the o_proj input. KV cache bytes per
  token = 576 x elem_bytes for each of the 24 MLA layers only.
- **MoE (layers 1..92)**: router gate 7168 -> 896 (bf16 in / fp32 logits), grouped sigmoid
  top-16 with `e_score_correction_bias` (`noaux_tc`), `routed_scaling_factor 1.0`;
  **latent MoE**: `routed_expert_down_proj` 7168 -> 3584 (`routed_expert_hidden_size`),
  routed experts operate in the 3584 space: gate_up 3584 -> 2x3072, down 3072 -> 3584, then
  `routed_expert_norm` RMSNorm(3584) and `routed_expert_up_proj` 3584 -> 7168; 2 shared
  experts in FULL hidden space: gate_up 7168 -> 2x(2*3072)=12288 total... verify the exact
  shared-expert intermediate from the code (`KimiK3MLP` with `num_shared_experts=2` and
  `moe_intermediate_size=3072`), down -> 7168. Routed experts are **MXFP4** (4-bit e2m1
  weights, group 32, one uint8 E8M0 scale per 32 weights: 0.5 B/weight + 1/32 B/weight of
  scale) per `quantization_config` (`format: mxfp4-pack-quantized`, targets Linear; in the
  shipped checkpoint only the routed experts are quantized — attention, shared experts,
  dense MLP, embeddings and lm_head stay bf16). `model/work/quantization.py` currently
  supports only fp8/modelopt-NVFP4: add MXFP4 (`compressed-tensors`-style config) with
  `converted_prefixes` limited to the routed experts.
- Norms: `input_layernorm`, `post_attention_layernorm` (fused residual add) per layer, final
  norm; vocab 163840, untied embeddings + lm_head; `num_nextn_predict_layers = 0` (no MTP).

## How to build it
- Template files: `model/work/models/qwen3_6_moe.py` (hybrid `LayerStack`s of a linear
  attention + MoE), `model/work/attention/linear.py` (GatedDeltaNet — KDA is its cousin; add
  a KDA spec with the full-rank gates and beta), `model/work/attention/glm52_dsa.py` lines
  50-89 (absorbed-MLA matmul groups — reuse WITHOUT the DSA indexer, i.e. a plain `mla.py`),
  `model/work/ffn/moe.py` (needs an `expert_input_dim`/latent size distinct from hidden, the
  latent down/up projections, the routed norm, and 2 shared experts), `model/work/ffn/dense.py`
  for layer 0. Tag the stacks `dense`, `kda`, `mla` so location maps can address them later.
- Add `model/work/models/kimi_k3.py`, register in `model/work/registry.py`, extend
  `quantization.py`, add goldens in `tests/test_model_work.py` (hand-derive the per-layer
  parameter counts and decode bytes for one KDA layer, one MLA layer and the MoE block at
  B=128, L=8192 and show the arithmetic in comments; total params should come out near the
  official 2.8T).
- A location map is NOT part of this task (the simulator arch does not exist yet).

## Deliverable
Commit your work on branch `kimi-k3` in one or a few commits with clear messages, each ending
with the trailer line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
Finish with a short report: files added/changed, the three CLI outputs above (summarized),
and any architecture question you could not resolve from the sources (do not guess silently
— state it).
