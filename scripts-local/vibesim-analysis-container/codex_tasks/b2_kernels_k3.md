# Task: register the Kimi-K3 decode kernels in VibeSim's profiling layer (branch `kimi-k3`)

You are in the VibeSim repository, on git branch `kimi-k3`. Do NOT switch branches. You have
NO GPU here (`CUDA_VISIBLE_DEVICES` is empty): write and unit-test the code CPU-side; the
GPU measurement runs are done by the operator afterwards with the exact commands you leave in
your report. Use `uv run ...`. Follow the repo's skills, in this order, and quote in your
report which steps you took from each: `skills/top-add-kernel/SKILL.md`,
`skills/orchestrator-add-kernel-to-python-profile/SKILL.md`, `skills/impl-register-kernel/SKILL.md`,
`skills/impl-wire-kernel-to-rust/SKILL.md`. Read `profiling/README.md` (kinds, backends,
envs, `kernel-profile` CLI, the `UNIQUE(gpu_name, backend, <args>)` contract) before coding.
Comments in English. Commit on `kimi-k3` with messages ending in
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Context
Kimi-K3 (moonshotai/Kimi-K3) on sglang v0.5.20, NVIDIA B200, production per-GPU shape =
attention TP8 rank (12 heads) + EP8 rank (112 mxfp4 routed experts). The decode step of one
decoder layer launches ~24 kernels (per-kernel tables measured by our driver are at
`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/*.json`
— read them: `kda_prod_B{32,128}_L8192.json`, `mla_prod_B1_L1048576.json`, `mla_prod_B128_L8192.json`;
fields: kernel name, count/step, total_us/step). sglang source for the call chains:
`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/k3_src_v0520/`
(`models_kimi_k3.py`, `layers_attention_linear_kda_backend.py`, `_attn_dir/linear/kernels/kda_triton.py`,
`_attn_dir/cutedsl_mla_backend.py`, `_attn_dir/trtllm_mla_backend.py`, `deepseek_v2.py`); the full
package lives inside the docker image `lmsysorg/sglang:v0.5.20` at
`/sgl-workspace/sglang/python/sglang` (you may `docker run --rm --entrypoint bash lmsysorg/sglang:v0.5.20 -c 'cat ...'`
to read files; never pass `--gpus`).

## Part 1 — a profiling environment for sglang v0.5.20 (B1)
Add a docker-backed `ProfileEnv` named `sglang_k3_env` in `profiling/exec/env.py` (see how
`vllm_env` is defined and how `profiling/exec/local.py` builds the container command and
bind-mounts `profiling/ launcher/ gpu/` read-only), image `lmsysorg/sglang:v0.5.20`, with
`/raid/yilegu/flashinfer_cache` mounted at `/root/.cache/flashinfer` (offline cubins), env
`SGLANG_OPT_FUSED_KDA_VERIFY=0 HF_HUB_OFFLINE=1`, `--shm-size 32g`. The worker entrypoint
must be `python -m profiling.exec.local_worker` inside that image (check that the image's
python has `torch`, `triton`, `flashinfer`, `sglang` importable — it does — and whether
`profiling/`'s own imports (e.g. pydantic/typer) are satisfied; if not, document the minimal
extra `pip install` and make the env install them on first use or via a tiny derived
Dockerfile `profiling/container/Dockerfile.sglang_k3`).

## Part 2 — kinds and backends (B2). Decisions are FIXED (do not re-litigate):
NEW kinds (different arg semantics from anything existing):
1. `kda_recurrent_decode` — sglang's `fused_sigmoid_gating_delta_rule_update` (Triton, in
   `sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py`), as called from
   `TritonKDAKernel.decode` (`kda_triton.py`) with `is_kda=True, lower_bound=-5.0,
   use_qk_l2norm_in_kernel=True`. Args: `num_heads, head_k_dim, head_v_dim, dtype (activations),
   state_dtype (fp32|bf16), lower_bound`; sweep axis `batch_size`. Backend `sglang_triton`
   (env `sglang_k3_env`), plus a `torch` semantic reference backend (multi-launch, CPU-testable
   shapes) as the skill requires. ALSO the fused-decode variant sglang actually runs at 12 heads:
   `kda_fused_decode` (tvm_ffi JIT CUDA in `sglang/kernels/ops/attention/kda_fused_decode`,
   engaged when `_prepare_fused_decode` succeeds) — expose it as backend `sglang_fused` of the
   SAME kind if its inputs map onto the same args (conv+recurrent+gated-norm fused), otherwise
   as a second kind `kda_fused_decode` — state which and why.
2. `mla_decode_attention` — dense absorbed-MLA decode over a paged latent KV (576 = 512 lora +
   64 rope per token). Args: `num_heads, kv_lora_rank, rope_dim, q_dtype, kv_dtype (bf16|fp8_e4m3),
   page_size`; sweep axes `batch_size, kv_len`. Backends `sglang_cutedsl_mla`
   (`CuteDslMLABackend.forward_decode` path / the `cute_dsl_mla_decode` flashinfer op it calls)
   and `sglang_trtllm_mla` (`TRTLLMMLABackend`, `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`
   or equivalent), and `sglang_triton` (`decode_attention_fwd` split-KV). Note trtllm-gen
   rejects 64 < heads < 128 and cute-dsl needs page_size 64 — encode as `BackendSupport`/runner
   validation, not silent fallbacks.
3. `mxfp4_fused_moe` — flashinfer `trtllm_fp4_block_scale_moe` with MXFP4 weights (uint8 e2m1
   packed, ue8m0 scales, group 32), SiTU activation with `gemm1_alpha=4.0, gemm1_clamp_limit=25.0`,
   biases, DeepSeek-V3 grouped-sigmoid top-k routing (n_group=1, topk_group=1, top_k=16,
   `routed_scaling_factor=1.0`), K3 shapes: `hidden_size=3584` (latent), `intermediate_size=3072`,
   `num_experts=896, num_local_experts=112, top_k=16`. Reuse the weight preparation sglang's
   `Mxfp4MoEMethod.process_weights_after_loading` performs (see how our driver fills random
   valid fp4 weights in `kimi_single_layer_decode.py::_fill_mxfp4_experts`). Args mirror
   `nvfp4_fused_moe` where sensible (`hidden_size, intermediate_size, num_experts,
   num_local_experts, top_k, input_dtype, weight_format="mxfp4_e2m1_ue8m0", group_size=32,
   routing_method="deepseek_v3_sigmoid", activation="situ", layerwise_global_ppm/
   folded_rank_position` if the repo's MoE kinds use them); sweep axis `num_tokens`.
NEW backends on EXISTING kinds:
4. `gdn_causal_conv_decode:sglang_triton` (`sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_update`,
   channels 36864 for 96 heads / 4608 for 12 heads, kernel 4, silu) and
   `gdn_gated_rms_norm:sglang_triton` (`sglang.kernels.ops.attention.fla.fused_norm_gate` FusedRMSNormGated,
   hidden=128, sigmoid gate). The existing `vllm_triton` backends are H200-gated
   (`profiling/runners/attention/_gdn_common.py`); add B200 to the sglang backends' `supports`.
5. `batched_gemm:sglang_k3_absorb` — the two absorb bmm's `q_nope @ w_kc` (num_batches=heads,
   k=128, n=512) and `attn_out @ w_vc` (k=512, n=128) for heads in {12, 96}; the existing
   runner pins GLM shapes (`profiling/runners/gemm/batched_gemm.py`).
Reuse as-is (no code): `single_gemm:sglang_bf16_auto/sglang_fused_a_auto`, `gemm_fp32_output:sglang_router_auto`,
`rms_norm`, `residual_rms_norm`, `mla_cache_append:sglang_cuda`, `elementwise`.

For every new kind: Python `profiling/kernels/<kind>.py` (KIND string, frozen Args, `register(...)`
per backend with `subprocess_env="sglang_k3_env"`), runner(s) under `profiling/runners/...`, the
barrel import in `profiling/kernels/__init__.py`, the Rust twin `simulator/src/timing/kernels/<kind>.rs`
(`KernelConfig`/`SweepCoords`/`KernelSpec` with the SAME KIND string, `register_kernel!`), and
the unit tests the skills demand (`cargo test -p simulator --lib` for the Rust twin — run it;
`uv run pytest` for the Python registry/schema tests — run them; CPU only).

## Do NOT
- write to `profiling/profile.db` (the shared DB) — no measurements here at all;
- change existing kinds' arg schemas or existing backends' behaviour;
- use a GPU.

## Report
End with (a) files added/changed, (b) test results, (c) the exact `uv run python -m launcher
kernel-profile run <kind> --backend <b> --spec '{...}' --db /tmp/k3_smoke.db --gpu-name "NVIDIA B200"`
smoke commands for each new kind/backend at ONE representative K3 shape, (d) anything you could
not resolve from the sources (say so explicitly rather than guessing).
