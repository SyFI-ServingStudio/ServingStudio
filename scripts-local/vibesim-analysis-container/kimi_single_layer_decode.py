#!/usr/bin/env python3
"""
kimi_single_layer_decode.py  (v2 -- Kimi-K3 / KDA)

Synthetic single-layer DECODE-latency driver for ONE Kimi-K3 decoder layer at a
KDA (Kimi Delta Attention) layer index, run on ONE B200. K3 is merged into
sglang stable v0.5.20 (class paths: sglang.srt.models.kimi_k3.{KimiK3DecoderLayer,
KimiK3DeltaAttention, KimiK3MLAAttention, KimiK3MoE}).

We construct exactly ONE `KimiK3DecoderLayer(config, layer_idx=5)`:
  * config.is_kda_layer(5) is True  -> layer.self_attn is KimiK3DeltaAttention
    (KDA linear/mamba-style recurrent attention), NOT KimiK3MLAAttention.
  * layer_idx 5 >= first_k_dense_replace(1) and %moe_layer_freq==0 -> layer.mlp
    is KimiK3MoE (latent MoE, situ->silu here, EP-rank-shrunk expert count).

EP-RANK modeling (target: 16xB200, EP=16): the routed-expert footprint on ONE
rank is n_routed_experts/EP. K3 has 896 routed experts; at EP=16 each rank holds
896/16 = 56. We build the layer with num_experts = --experts (default 56), keep
top-k = 16 and +2 shared experts. --ep is metadata only (records the modeled EP).

KDA STATE: KimiK3DeltaAttention uses a FIXED-SIZE recurrent state, NOT an
MLATokenToKVPool. Per request/slot:
  * conv state  (3, 36864)      bf16   [ (short_conv_kernel-1), proj+2*proj_k ]
  * temporal    (96, 128, 128)  fp32   [ num_heads, head_dim, head_dim ]
Both are independent of sequence length -> KDA decode cost is context-flat.
We stub the HybridReqToTokenPool / ModelRunner surface that KDAAttnBackend and
its forward_decode touch, wire a decode ForwardBatch, and CUDA-event time the
step (warmup 20, 100 iters).

Faithfulness caveats (see final report / --json-out):
  * bf16 random init, quant_config=None -> no MXFP4 routed experts (real K3
    ships MXFP4). MoE GEMM shapes are faithful; per-element cost is bf16 not fp4.
  * hidden_act forced to "silu" (K3 uses "situ"); same GEMM shapes.
  * single-GPU EP-rank approximation: no a2a / all-reduce cross-rank traffic.
  * KDA state is randomly initialised (not a real prefill), which is fine for
    latency (kernel shapes are state-size-driven, not value-driven).
"""
import argparse
import contextlib
import json
import os
import traceback
from types import SimpleNamespace

import torch

HIDDEN_SIZE = 7168  # K3 hidden dim


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def build_config_dir(experts, ep, num_layers=8, out_dir="/tmp/k3_cfg",
                     moe_backend="bf16", attn_type="kda", target_layer=5,
                     attn_heads=96, mamba_ssm_dtype="float32"):
    """Write a Kimi-K3 config.json (model_type=kimi_linear) with the routed-expert
    count shrunk to the per-EP-rank footprint. Returns (out_dir, cfg_dict).

    moe_backend != "bf16" (flashinfer_mxfp4 / deep_gemm) switches hidden_act to
    K3's real "situ" activation (the flashinfer_mxfp4 route/quant/finalize tuning
    at kimi_k3.py:501-514/559-562/1139-1166 is gated on hidden_act=="situ") and
    seeds the situ activation constants (beta=4.0/linear_beta=25.0, K3's real
    values) that KimiK3MoE forwards to the experts as gemm1_alpha/clamp_limit.
    The bf16 path is unchanged (silu, no situ betas).

    attn_type controls whether `target_layer` is a KDA linear-attn layer or a
    full-MLA (dense attention) layer. is_kda_layer(idx) == (idx+1) in kda_layers.
      * "kda": EVERY layer KDA (kda_layers = 1..num_layers, full_attn=[]), so the
        default target_layer=5 (-> 6) is a KDA layer. UNCHANGED legacy behavior.
      * "mla": target_layer is a FULL-attn (MLA) layer -- put (target_layer+1) in
        full_attn_layers and NOT in kda_layers so is_kda_layer(target)=False, and
        expose the attention heads THIS RANK owns.

    attn_heads = heads per GPU for the profiled layer type. 96 = the full head
    set (DP attention, attn_tp=1: what the first sweeps modeled). 12 = K3's
    heads/GPU under the cookbook's B200 recipes (attention TP8: PP2xTP8 and
    PP2xDCPEP8), which is also the ONLY shape the Blackwell MLA decode kernels
    accept (trtllm-gen rejects 64 < heads < 128; job 803) and the shape the fused
    KDA decode kernel is compiled for (_prepare_fused_decode: 12 heads).
    Building at tp=1 with attn_heads=12 reproduces every per-rank GEMM/kernel
    shape of a TP8 rank except the o_proj all-reduce.
    """
    mxfp4 = moe_backend != "bf16"
    if attn_type == "mla":
        # target_layer is MLA (full attention); all others KDA.
        full_attn_layers = [target_layer + 1]                       # 1-based
        kda_layers = [i for i in range(1, num_layers + 1)
                      if i != target_layer + 1]
        num_attention_heads = attn_heads   # MLA heads on this rank
        kda_heads = 96                     # unused by an MLA layer
    else:
        full_attn_layers = []
        kda_layers = list(range(1, num_layers + 1))   # every layer KDA
        num_attention_heads = 64   # MLA heads unused by a KDA layer
        kda_heads = attn_heads     # KDA heads on this rank
    cfg = {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "vocab_size": 163840,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,  # 96 for MLA, 64 for KDA-only
        "moe_intermediate_size": 3072,
        "num_experts": experts,             # EP-rank-shrunk (896/EP; 56 @ EP16)
        "num_experts_per_token": 16,        # K3 top-k
        "num_shared_experts": 2,
        "intermediate_size": 33792,         # dense layer-0 MLP (real HF value)
        "routed_scaling_factor": 1.0,       # real HF value (was 2.5 in the first sweeps)
        "mla_use_output_gate": True,        # real HF value: MLA o_proj output gate
        "mla_use_nope": True,               # real HF value (MLA skips rope on the decode path)
        "moe_renormalize": True,            # real HF value: renormalize top-k weights
        "num_key_value_heads": 96,          # real HF value
        "attn_res_block_size": 12,          # real HF value: attention-residual stream block
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "use_grouped_topk": True,
        "num_expert_group": 1,              # shrunk expert set -> single group
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "moe_router_activation_func": "sigmoid",
        # situ activation constants (K3's real values); only consumed on the
        # mxfp4/situ path. Harmless on the bf16/silu path.
        "activation_situ_beta": 4.0 if mxfp4 else None,
        "activation_situ_linear_beta": 25.0 if mxfp4 else None,
        "q_lora_rank": 1536,                # MLA dims (only full-attn layers use)
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "routed_expert_hidden_size": 3584,  # latent MoE space
        "latent_moe_use_norm": True,        # real HF value: RMSNorm(3584) on the routed latent
                                            # (sglang default False -> would silently drop it)
        "hidden_act": "situ" if mxfp4 else "silu",  # K3 uses "situ"; silu=same shapes on bf16
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 1048576,
        # KDA recurrent-state dtype (configs/mamba_utils.py reads config.mamba_ssm_dtype;
        # the cookbook B200 recipes run --mamba-ssm-dtype bfloat16 = half the state bytes).
        "mamba_ssm_dtype": mamba_ssm_dtype,
        "linear_attn_config": {
            "num_heads": kda_heads,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
            "gate_lower_bound": -5.0,
            # Real HF config (moonshotai/Kimi-K3 text_config.linear_attn_config):
            # full-rank f/g gates (7168->heads*128 each) -> the fused qkvbfg
            # projection + _merge_bfa_weights paths engage as in production.
            "use_full_rank_gate": True,
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"))
    log(f"[cfg] wrote {out_dir}/config.json  experts={experts} (EP={ep}) "
        f"top_k=16 shared=2 layers={num_layers}  moe_backend={moe_backend} "
        f"hidden_act={cfg['hidden_act']}  attn_type={attn_type} "
        f"target_layer={target_layer} mla_heads={num_attention_heads} kda_heads={kda_heads} "
        f"kda_layers={kda_layers} full_attn={full_attn_layers}")
    return out_dir, cfg


def bootstrap(cfg_dir, moe_backend="bf16", attn_type="kda", target_layer=5,
              attention_backend="triton", page_size=1, kv_cache_dtype="auto"):
    """Publish ServerArgs (role=scheduler), build ModelConfig, init distributed +
    model-parallel + dp-attention (tp=1, world=1, EP off).

    moe_backend != "bf16" sets ServerArgs.moe_runner_backend and calls
    initialize_moe_config(sa) so get_moe_runner_backend().is_flashinfer_mxfp4()
    (etc.) is true BEFORE the layer is constructed -- otherwise the backend
    stays AUTO and the experts fall back to the bf16 triton fused-MoE."""
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    torch.cuda.set_device(0)

    sa_kwargs = dict(
        model_path=cfg_dir,
        tp_size=1,
        dtype="bfloat16",
        device="cuda",
        trust_remote_code=True,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        enable_dp_attention=False,
        page_size=page_size,
        kv_cache_dtype=kv_cache_dtype,
    )
    if attn_type == "mla":
        # The MLA layer dispatches on get_attn_backend().decode_attention_backend_str
        # (None here) or the server-args default. Pin the requested backend so the
        # hand-built backend in MLAState and the layer's dispatch agree. Production
        # K3 on Blackwell auto-resolves trtllm_mla (cutedsl_mla for decode under
        # DCP); triton is the dependency-free default used for the first sweeps.
        sa_kwargs["attention_backend"] = attention_backend
        sa_kwargs["decode_attention_backend"] = attention_backend
    if moe_backend != "bf16":
        sa_kwargs["moe_runner_backend"] = moe_backend
    sa = ServerArgs(**sa_kwargs)
    set_global_server_args_for_scheduler(sa)  # publish(role="scheduler")

    if moe_backend != "bf16":
        from sglang.srt.layers.moe.utils import (
            initialize_moe_config,
            get_moe_runner_backend,
        )
        # Reads the just-published global ServerArgs (no positional arg).
        initialize_moe_config()
        # flashinfer mxfp4 precision -> "default" enables the SM100 trtllm-gen
        # branch in process_weights_after_loading / the situ front tuning.
        try:
            from sglang.srt.runtime_context import get_exec
            if getattr(get_exec().moe, "flashinfer_mxfp4_moe_precision", None) is None:
                get_exec().moe.flashinfer_mxfp4_moe_precision = "default"
            log(f"[moe] flashinfer_mxfp4_moe_precision="
                f"{get_exec().moe.flashinfer_mxfp4_moe_precision}")
        except Exception as e:
            log(f"[moe] could not set flashinfer_mxfp4_moe_precision: {e}")
        rb = get_moe_runner_backend()
        log(f"[moe] runner_backend={rb} is_flashinfer_mxfp4="
            f"{rb.is_flashinfer_mxfp4()} is_deep_gemm="
            f"{getattr(rb, 'is_deep_gemm', lambda: '?')()}")

    mc = ModelConfig(model_path=cfg_dir, dtype="bfloat16", trust_remote_code=True)
    cfg = mc.hf_config
    cfg.dtype = torch.bfloat16  # KimiK3DeltaAttention reads config.dtype

    port = 29500 + (os.getpid() % 2000)
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    initialize_dp_attention(sa, mc)

    from sglang.srt.runtime_context import get_parallel
    log(f"[boot] tp={get_parallel().tp_size} attn_tp={get_parallel().attn_tp_size} "
        f"world=1  is_kda_layer({target_layer})={cfg.is_kda_layer(target_layer)}  "
        f"attention_arch={getattr(mc, 'attention_arch', None)} dtype={cfg.dtype}")
    return sa, mc, cfg


# --------------------------------------------------------------------------- #
# Layer
# --------------------------------------------------------------------------- #
def build_layer(cfg, layer_idx=5, moe_backend="bf16", attn_type="kda", seed=0):
    """Build ONE KimiK3DecoderLayer; random-init bf16. attn_type "kda" builds at a
    KDA (linear-attn) layer index (self_attn=KimiK3DeltaAttention); "mla" builds at
    a full-attn index (self_attn=KimiK3MLAAttention, dense MLA).

    moe_backend != "bf16": pass a real Mxfp4Config so the experts get the
    Mxfp4MoEMethod (uint8 e2m1 weights + ue8m0 scales), fill the uint8 weight/
    scale buffers with random valid data, and run process_weights_after_loading
    (the SM100 trtllm / deep_gemm transform). With RANDOM fp4 weights the MoE
    output may be non-finite -- ACCEPTABLE; kernel timing is shape/dtype-driven.
    The KDA attention path and all bf16 sub-modules are untouched."""
    from sglang.srt.models.kimi_k3 import (
        KimiK3DecoderLayer, KimiK3DeltaAttention, KimiK3MLAAttention, KimiK3MoE,
    )
    if attn_type == "mla":
        assert not cfg.is_kda_layer(layer_idx), \
            f"layer {layer_idx} must be a FULL-attn (MLA) layer for attn_type=mla"
    else:
        assert cfg.is_kda_layer(layer_idx), f"layer {layer_idx} is not a KDA layer"
    quant_config = None
    if moe_backend != "bf16":
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
        quant_config = Mxfp4Config(is_checkpoint_mxfp4_serialized=True)
        log(f"[layer] mxfp4 quant_config={type(quant_config).__name__} "
            f"quant_format={getattr(quant_config, 'quant_format', None)}")
        # Mxfp4Config.get_quant_method only produces a method for the FusedMoE
        # experts; for the layer's LinearBase modules (KDA qkv/o proj, MoE gate,
        # latent down/up proj) it returns None, which trips the
        # `assert self.quant_method is not None` in LinearBase.__init__. Fall
        # those back to bf16 UnquantizedLinearMethod -> only the MoE experts run
        # fp4, attention/latent linears stay bf16 (the intended faithful model:
        # we replace ONLY the MoE, matching the bf16 attn baseline).
        from sglang.srt.layers.linear import LinearBase
        UnquantizedLinearMethod = None
        for _mod in ("sglang.srt.layers.quantization.unquant",
                     "sglang.srt.layers.quantization.base_config",
                     "sglang.srt.layers.linear"):
            try:
                import importlib
                UnquantizedLinearMethod = getattr(
                    importlib.import_module(_mod), "UnquantizedLinearMethod")
                log(f"[layer] UnquantizedLinearMethod from {_mod}")
                break
            except (ImportError, AttributeError):
                continue
        assert UnquantizedLinearMethod is not None, \
            "could not locate UnquantizedLinearMethod"
        _orig_gqm = Mxfp4Config.get_quant_method
        from sglang.srt.layers.radix_attention import RadixAttention

        def _gqm_linear_fallback(self, layer, prefix):
            # LinearBase modules (attn qkv/o proj, MoE gate, latent down/up) get a
            # bf16 fallback; only FusedMoE experts run fp4. RadixAttention (the MLA
            # attn_mqa) would otherwise raise "Mxfp4 attention layer is not
            # implemented" -> return None so it stays bf16 (create_weights skipped).
            if isinstance(layer, RadixAttention):
                return None
            m = _orig_gqm(self, layer, prefix)
            if m is None and isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return m

        Mxfp4Config.get_quant_method = _gqm_linear_fallback
        log("[layer] patched Mxfp4Config.get_quant_method: LinearBase/RadixAttention "
            "-> bf16 unquantized fallback")
    torch.manual_seed(seed)   # weights + mxfp4 fill are drawn from the global RNG
    # Production (KimiK3Model.__init__, kimi_k3.py:2816) hands every layer three
    # side streams: [0] MoE shared-expert overlap, [1] MLA alt, [2] KDA bfa /
    # MLA gate. The bfa/gate side-stream paths only fire under CUDA-graph
    # capture mode (get_is_capture_mode()), the MoE one whenever alt_stream is
    # set -- so pass them in both modes to match the production launch graph.
    alt_streams = [torch.cuda.Stream() for _ in range(3)]
    # Construct under default dtype bf16 like sglang's model loader does, so the
    # params the model declares fp32 on purpose (KDA A_log, dt_bias, the conv1d
    # weight: kimi_k3.py:1615/1623/1637) STAY fp32. A blanket .to(bf16) would
    # downcast them and _prepare_fused_decode would refuse the fused KDA decode
    # kernel (it checks those dtypes), silently forcing the unfused chain.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        layer = KimiK3DecoderLayer(config=cfg, layer_idx=layer_idx,
                                   quant_config=quant_config, alt_streams=alt_streams)
    finally:
        torch.set_default_dtype(prev_dtype)
    layer._driver_alt_streams = alt_streams

    is_kda = isinstance(layer.self_attn, KimiK3DeltaAttention)
    is_mla = isinstance(layer.self_attn, KimiK3MLAAttention)
    is_moe = isinstance(layer.mlp, KimiK3MoE)
    log(f"[layer] KimiK3DecoderLayer(layer_idx={layer_idx}): "
        f"self_attn={type(layer.self_attn).__name__} (KDA={is_kda} MLA={is_mla})  "
        f"mlp={type(layer.mlp).__name__} (MoE={is_moe})")
    if attn_type == "mla":
        assert is_mla and not is_kda, "self_attn must be KimiK3MLAAttention (MLA)"
    else:
        assert is_kda and not is_mla, "self_attn must be KimiK3DeltaAttention (KDA)"
        assert layer.self_attn.attn.lower_bound == -5.0, \
            f"expected KDA gate_lower_bound=-5.0, got {layer.self_attn.attn.lower_bound}"
    assert is_moe, "mlp must be KimiK3MoE"

    layer = layer.cuda().eval()   # dtypes as constructed (bf16 default, fp32 where declared)
    n = 0
    n_fp32 = 0
    with torch.no_grad():
        for p in layer.parameters():
            if p.is_floating_point():
                p.normal_(0.0, 0.02)
                n += p.numel()
                if p.dtype == torch.float32:
                    n_fp32 += p.numel()
        for b in layer.buffers():
            if b.is_floating_point():
                b.normal_(0.0, 0.02)
    log(f"[layer] float params={n/1e9:.3f}B (fp32 kept: {n_fp32/1e6:.2f}M)")
    # RadixLinearAttention captured conv_weights = qkv_conv1d.weight.squeeze(1) as a
    # plain tensor VIEW at __init__ time (on CPU). nn.Module.cuda() swaps each
    # Parameter's .data in place, so Parameter refs (A_log, dt_bias) stay valid but
    # this stale view still points at the old CPU storage -> refresh it post-move.
    # qkv_conv1d.weight is float32 [proj,1,K]; the kernel wants a device tensor.
    if attn_type != "mla":
        da = layer.self_attn
        da.attn.conv_weights = da.qkv_conv1d.weight.squeeze(1)
        da.attn.bias = da.qkv_conv1d.bias  # None (bias=False), kept for parity
        log(f"[layer] random-init {n/1e9:.3f}B float params -> cuda/bf16/eval; "
            f"conv_weights refreshed dev={da.attn.conv_weights.device}")
    else:
        # MLA absorbed decode reads self_attn.w_kc / w_vc, which the full model
        # builds in KimiK3ForCausalLM.post_load_weights() by absorbing kv_b_proj.
        # We build ONE layer directly, so replicate that absorption here (bf16
        # random weights: _get_k3_dense_weight is just kv_b_proj.weight.data).
        sa2 = layer.self_attn
        kv_b_weight = sa2.kv_b_proj.weight.data
        w_kc, w_vc = kv_b_weight.unflatten(
            0, (-1, sa2.qk_nope_head_dim + sa2.v_head_dim)
        ).split([sa2.qk_nope_head_dim, sa2.v_head_dim], dim=1)
        sa2.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        sa2.w_vc = w_vc.contiguous().transpose(1, 2)
        log(f"[layer] random-init {n/1e9:.3f}B float params -> cuda/bf16/eval "
            f"(MLA: absorbed kv_b_proj -> w_kc{tuple(sa2.w_kc.shape)} "
            f"w_vc{tuple(sa2.w_vc.shape)})")

    if moe_backend != "bf16":
        _fill_mxfp4_experts(layer.mlp.experts, moe_backend)

    _post_load_merge(layer, attn_type)
    return layer


def _post_load_merge(layer, attn_type):
    """Replicate KimiK3LinearForCausalLM.post_load_weights (kimi_k3.py:3355-3373)
    for the single layer: merge the horizontally-fused decode weights and cast the
    router correction bias to fp32. Production runs this once after weight load,
    before CUDA-graph capture; without it the driver measures the UNFUSED MoE
    front (3 GEMV launches instead of 1) and a per-step bias upcast.

    Notes on what engages at this (DP-attention, attn_tp=1, 96-head) shape:
    - _merge_front_weights: needs latent MoE + shared experts + a2a backend
      "none" (true here, single GPU) -> merges shared gate_up + router gate +
      latent down_proj into one [H, gu+E+latent] GEMM and enables _forward_fused.
      Under a real EP a2a deployment only gate+down would merge (SGLANG_K3_FUSED_FRONT).
    - _merge_bfa_weights: only when use_full_rank_gate (K3: False) -> no-op.
    - _prepare_fused_decode: compiled for the TP8 12-head layout; at 96 heads it
      logs "disabled" and returns -> the unfused Triton KDA chain IS the
      production DP path, so nothing changes here."""
    mlp = layer.mlp
    with torch.no_grad():
        if hasattr(mlp, "_merge_front_weights"):
            mlp._merge_front_weights()
        bias = getattr(getattr(mlp, "gate", None), "e_score_correction_bias", None)
        if bias is not None and bias.dtype != torch.float32:
            bias.data = bias.data.to(torch.float32)
        if attn_type != "mla":
            sa = layer.self_attn
            if hasattr(sa, "_merge_bfa_weights"):
                sa._merge_bfa_weights()
            if hasattr(sa, "_prepare_fused_decode"):
                sa._prepare_fused_decode()
    front = getattr(mlp, "_front_w", None)
    log(f"[layer] post_load_merge: front_w={None if front is None else tuple(front.shape)} "
        f"fused_front_eligible={getattr(mlp, '_eligible_for_fused_front', None)} "
        f"bias_dtype={None if bias is None else bias.dtype}"
        + ("" if attn_type == "mla" else
           f" bfa_w={getattr(layer.self_attn, '_bfa_w', None) is not None}"
           f" kda_fused_decode_ready={getattr(layer.self_attn, '_kda_fused_decode_ready', None)}"))


def _fill_mxfp4_experts(experts, moe_backend):
    """Fill the FusedMoE experts' uint8 e2m1 weight buffers + ue8m0 scales with
    random-but-valid data, then run process_weights_after_loading (the real
    SM100 trtllm / deep_gemm weight transform). Values are junk (output may be
    non-finite) but the kernel path + shapes/dtypes are faithful."""
    # Enumerate params AND buffers; log every tensor so shapes are on record.
    named = list(experts.named_parameters()) + list(experts.named_buffers())
    log(f"[mxfp4] experts={type(experts).__name__} tensors:")
    with torch.no_grad():
        for nm, t in named:
            tag = ""
            if t.dtype == torch.uint8:
                if "scale" in nm:
                    t.data.fill_(127)          # ue8m0 exponent 127 -> scale 2^0
                    tag = " <- ue8m0 fill 127"
                else:
                    t.data.random_(0, 256)     # e2m1 packed bytes
                    tag = " <- e2m1 randint"
            elif t.is_floating_point() and "bias" in nm:
                t.data.normal_(0.0, 0.02)      # w13/w2 bias (already normal'd, redo)
                tag = " <- bias normal"
            log(f"[mxfp4]   {nm:32s} {str(tuple(t.shape)):22s} {t.dtype}{tag}")

    qm = experts.quant_method
    log(f"[mxfp4] quant_method={type(qm).__name__} "
        f"use_flashinfer={getattr(qm, 'use_flashinfer', '?')} "
        f"use_marlin={getattr(qm, 'use_marlin', '?')} "
        f"_fi_kernel={getattr(qm, '_fi_kernel', '?')} "
        f"precision={getattr(qm, 'flashinfer_mxfp4_moe_precision', '?')}")

    # Prefer FusedMoE.process_weights_after_loading if it exists, else the
    # quant_method's. Wrap + log the full traceback on failure.
    try:
        if hasattr(experts, "process_weights_after_loading"):
            experts.process_weights_after_loading()
            log("[mxfp4] experts.process_weights_after_loading() OK")
        else:
            qm.process_weights_after_loading(experts)
            log("[mxfp4] quant_method.process_weights_after_loading(experts) OK")
    except Exception as e:
        log(f"[mxfp4] process_weights_after_loading FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        raise


# --------------------------------------------------------------------------- #
# KDA fixed-size recurrent state + stubbed pool / model_runner / backend
# --------------------------------------------------------------------------- #
class KDAState:
    """Owns the fixed-size conv + temporal recurrent state and the stub surface
    that KDAAttnBackend / forward_decode read."""

    def __init__(self, cfg, sa, B, device="cuda", seed=0):
        from sglang.srt.layers.attention.linear.utils import (
            LinearAttnKernelBackend, LinearAttnBackends,
        )
        from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend

        self.B = B
        num_slots = B + 1  # +1 padding slot, real reqs at 0..B-1

        params = cfg.mamba2_cache_params
        conv_shape = tuple(params.shape.conv[0])       # (3, 36864)
        temporal_shape = tuple(params.shape.temporal)  # (96, 128, 128)
        conv_dtype = params.dtype.conv                 # bf16
        ssm_dtype = params.dtype.temporal              # fp32

        # Seeded, NON-zero-mean recurrent state: a step that skips the state read
        # (or its writeback) then changes the output / post-step state measurably,
        # which a zero-mean random state would hide behind averaging.
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 17)
        self.conv = (torch.randn((num_slots, *conv_shape), dtype=torch.bfloat16,
                                 device=device, generator=gen) * 0.02).to(conv_dtype)
        self.temporal = (torch.randn((num_slots, *temporal_shape), dtype=torch.float32,
                                     device=device, generator=gen) * 0.02 + 0.05
                         ).to(ssm_dtype)
        self.state_bytes = (self.conv.numel() * self.conv.element_size()
                            + self.temporal.numel() * self.temporal.element_size())
        # Golden support: initial copies so a correctness step can start from the
        # same state after warm-up/timing mutated the buffers in place.
        self._conv0 = self.conv.clone()
        self._temporal0 = self.temporal.clone()

        conv_buf = self.conv
        temporal_buf = self.temporal

        # layer cache view (single layer): conv is a list, conv[0] is the buffer.
        layer_cache = SimpleNamespace(
            conv=[conv_buf], temporal=temporal_buf,
            replayssm_d=None, replayssm_k=None, replayssm_g=None,
        )
        mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(conv=[conv_buf], temporal=temporal_buf),
            enable_linear_replayssm=False,
            replayssm_write_pos=None,
        )

        dev = device

        class _StubReqToTokenPool:
            size = num_slots

            def __init__(self):
                self.mamba_pool = mamba_pool

            def get_mamba_indices(self, req_pool_indices):
                # slot id == req id (identity mapping in this synthetic driver)
                return req_pool_indices.to(torch.int32)

            def translate_mamba_indices(self, idx):
                return idx  # identity for the non-unified pool

            def mamba2_layer_cache(self, layer_id):
                return layer_cache

        self.req_pool = _StubReqToTokenPool()

        model_runner = SimpleNamespace(
            server_args=sa,
            device=dev,
            is_draft_worker=False,
            req_to_token_pool=self.req_pool,
            token_to_kv_pool=None,
            model_config=SimpleNamespace(hf_text_config=cfg),
            linear_attn_backends=LinearAttnBackends(
                decode=LinearAttnKernelBackend.TRITON,
                prefill=LinearAttnKernelBackend.TRITON,
                verify=LinearAttnKernelBackend.TRITON,
            ),
        )
        self.kda = KDAAttnBackend(model_runner)
        log(f"[kda] KDAAttnBackend built; conv={tuple(self.conv.shape)}/{self.conv.dtype} "
            f"temporal={tuple(self.temporal.shape)}/{self.temporal.dtype} "
            f"state={self.state_bytes/1e6:.2f}MB/slot-set")

    def reset(self):
        with torch.no_grad():
            self.conv.copy_(self._conv0)
            self.temporal.copy_(self._temporal0)

    def state_after(self):
        """Post-step recurrent state of the real requests (slots 0..B-1)."""
        return {"conv": self.conv[:self.B].clone(),
                "temporal": self.temporal[:self.B].clone()}

    def make_forward_batch(self, cfg, seq_len):
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch, ForwardMode,
        )
        B = self.B
        dev = "cuda"
        req_pool_indices = torch.arange(B, device=dev, dtype=torch.int64)
        seq_lens = torch.full((B,), seq_len, device=dev, dtype=torch.int64)
        seq_lens_cpu = torch.full((B,), seq_len, dtype=torch.int64)
        positions = torch.full((B,), seq_len - 1, device=dev, dtype=torch.int64)
        out_cache_loc = req_pool_indices.clone()
        input_ids = torch.randint(0, cfg.vocab_size, (B,), device=dev, dtype=torch.int64)

        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=B,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(B * seq_len),
            out_cache_loc=out_cache_loc,
            positions=positions,
        )
        # KDA metadata plumbing: no radix mamba tracking, no spec.
        fb.req_to_token_pool = self.req_pool
        fb.token_to_kv_pool = None
        fb.mamba_track_indices = None
        fb.mamba_track_mask = None
        fb._original_batch_size = B
        fb.spec_info = None
        # Precompute KDA forward metadata (query_start_loc + mamba_cache_indices)
        self.kda.init_forward_metadata(fb)
        return fb

    def free(self):
        del self.conv, self.temporal, self.kda, self.req_pool
        torch.cuda.empty_cache()


class MLAState:
    """MLA (full dense-attention) analogue of KDAState. Owns a per-(B,L)
    MLATokenToKVPool (ONE combined latent+rope vector of dim 576/token, seeded
    random bf16), a ReqToTokenPool mapping each of B reqs to L contiguous token
    slots, a KVIndexTranslator, and the hand-driven TRITON MLA attention backend
    against a stubbed model_runner. Unlike KDA (fixed recurrent state, context-
    flat), MLA reads the WHOLE L-token latent KV every decode step -> cost grows
    with L.

    We drive the ABSORBED MLA decode path: for a full-attn layer the triton
    handler returns AttnForwardMethod.MLA, so DeepseekV2AttentionMLA.forward_absorb
    reads the latent (kv_lora_rank=512) + rope (64) per token via the triton
    decode kernel. Random KV/weights -> latency is faithful (shape/dtype-driven),
    values may be non-finite (reported)."""

    def __init__(self, cfg, mc, sa, layer_idx, B, L, device="cuda",
                 attention_backend="triton", page_size=1, kv_dtype=torch.bfloat16,
                 seed=0):
        from sglang.srt.mem_cache.memory_pool import (
            MLATokenToKVPool, ReqToTokenPool,
        )
        from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
        from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

        self.B, self.L = B, L
        self.attention_backend = attention_backend
        self.page_size = page_size
        assert L % page_size == 0, f"seq_len {L} must be a multiple of page_size {page_size}"
        n_tokens = B * L
        kv_lora_rank = cfg.kv_lora_rank            # 512
        qk_rope_head_dim = cfg.qk_rope_head_dim    # 64  -> kv_cache_dim = 576

        # MLA KV pool: buffer (size+page_size, 1, 576) for the single layer. The
        # first page (slots 0..page_size-1) is the pool's padding page, as in the
        # real allocator; req r's L tokens live at [page_size + r*L, +L) so pages
        # stay aligned for the paged backends (flashmla/cutedsl/trtllm: 64).
        self.kv_dtype = kv_dtype
        self.kv_pool = MLATokenToKVPool(
            size=n_tokens,
            page_size=page_size,
            dtype=kv_dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
            start_layer=layer_idx,
            end_layer=layer_idx + 1,
        )
        # get_value_buffer(start_layer).shape[-1] == kv_lora_rank; the triton
        # backend's mambaish (kimi-linear) branch asks the pool for v_head_dim.
        _sl = self.kv_pool.start_layer
        self.kv_pool.get_v_head_dim = (
            lambda: self.kv_pool.get_value_buffer(_sl).shape[-1]
        )
        # Seeded latent+rope context (finite). fp8 has no normal_(): fill via a bf16
        # staging tensor chunk-wise and cast. STRUCTURED: on top of the N(0,0.02)
        # background, 64 rows per request spread uniformly over [0, L) are scaled
        # x32 ("planted keys"). Softmax then concentrates on those rows, so an
        # attention that reads only part of the context (a cheating "speedup")
        # produces a measurably different output; uniform random KV would not.
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 23)
        self.slot0 = page_size
        with torch.no_grad():
            kb = self.kv_pool.kv_buffer[0]
            step = 1 << 20
            for s in range(0, kb.shape[0], step):
                chunk = kb[s:s + step]
                chunk.copy_(torch.randn(chunk.shape, device=device,
                                        dtype=torch.bfloat16, generator=gen) * 0.02)
            n_plant = min(64, L)
            offs = (torch.arange(n_plant, device=device) * (L // n_plant)
                    + torch.randint(0, max(1, L // n_plant), (n_plant,),
                                    device=device, generator=gen))
            for r in range(B):
                rows = self.slot0 + r * L + offs
                kb[rows] = (kb[rows].to(torch.bfloat16) * 32.0).to(kb.dtype)
            # Golden support: the step writes exactly one row per request
            # (out_cache_loc = last slot of the request's range).
            self.write_locs = (self.slot0 + torch.arange(B, device=device) * L
                               + (L - 1))
            self._kv0_rows = kb[self.write_locs].clone()
        kb = self.kv_pool.kv_buffer[0]
        self.kv_bytes = kb.numel() * kb.element_size()

        # req -> token slot map: req r (row r+1; row 0 padding) owns
        # [page_size + r*L, page_size + r*L + L).
        self.req_pool = ReqToTokenPool(
            size=B, max_context_len=L, device=device, enable_memory_saver=False,
        )
        rows = torch.arange(B, device=device, dtype=torch.int32).view(B, 1) * L
        cols = torch.arange(L, device=device, dtype=torch.int32).view(1, L)
        self.req_pool.req_to_token[1:B + 1, :L] = self.slot0 + rows + cols

        # KVIndexTranslator (non-unified passthrough: allocator=None -> gathers
        # straight from req_to_token; this is what forward_decode's
        # _fill_kv_indptr_and_indices calls in v0.5.20).
        self.translator = KVIndexTranslator(
            req_to_token=self.req_pool.req_to_token,
            token_to_kv_pool_allocator=None,
            token_to_kv_pool=self.kv_pool,
            page_size=page_size,
            device=device,
        )

        # Stub ModelRunner: the union of what the registered MLA-capable backends
        # read at construction (triton: server_args/device/pools/translator/page_size/
        # kv_cache_dtype/model_config; trtllm_mla/flashmla/cutedsl_mla additionally:
        # use_mla_backend, dtype, max_running_requests, model_config.{num_attention_
        # heads,kv_lora_rank,qk_*_head_dim,v_head_dim,scaling,context_len}).
        model_runner = SimpleNamespace(
            server_args=sa,
            device=device,
            gpu_id=0,
            is_draft_worker=False,
            use_mla_backend=True,
            dtype=torch.bfloat16,
            max_running_requests=B,
            req_to_token_pool=self.req_pool,
            token_to_kv_pool=self.kv_pool,
            token_to_kv_pool_allocator=None,
            kv_index_translator=self.translator,
            sliding_window_size=None,
            page_size=page_size,
            kv_cache_dtype=kv_dtype,
            model_config=mc,
        )
        # Same factory the real ModelRunner uses (attention_registry.py:43-48), so
        # the sweep over {triton, trtllm_mla, cutedsl_mla, flashmla, ...} exercises
        # each backend's own constructor + metadata path.
        create = ATTENTION_BACKENDS[attention_backend]
        self.backend = create(model_runner)
        log(f"[mla] {type(self.backend).__name__} built (attention_backend="
            f"{attention_backend} page_size={page_size} kv_dtype={kv_dtype}); "
            f"kv={self.kv_bytes/1e9:.3f}GB "
            f"kv_dim={self.kv_pool.kv_cache_dim} "
            f"num_head={getattr(self.backend, 'num_head', getattr(self.backend, 'num_q_heads', '?'))}")

    def reset(self):
        with torch.no_grad():
            self.kv_pool.kv_buffer[0][self.write_locs] = self._kv0_rows

    def state_after(self):
        """The KV rows the step wrote (one per request)."""
        return {"kv_rows": self.kv_pool.kv_buffer[0][self.write_locs].clone()}

    def make_forward_batch(self, cfg):
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch, ForwardMode,
        )
        B, L = self.B, self.L
        dev = "cuda"
        req_pool_indices = torch.arange(B, device=dev, dtype=torch.int64) + 1  # rows 1..B
        seq_lens = torch.full((B,), L, device=dev, dtype=torch.int64)
        seq_lens_cpu = torch.full((B,), L, dtype=torch.int64)
        positions = torch.full((B,), L - 1, device=dev, dtype=torch.int64)
        # current-token write slot = last slot of req r's contiguous range.
        out_cache_loc = self.slot0 + (req_pool_indices - 1) * L + (L - 1)
        input_ids = torch.randint(0, cfg.vocab_size, (B,), device=dev, dtype=torch.int64)

        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=B,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(B * L),
            out_cache_loc=out_cache_loc,
            positions=positions,
        )
        fb.req_to_token_pool = self.req_pool
        fb.token_to_kv_pool = self.kv_pool
        fb.attn_backend = self.backend
        fb.spec_info = None
        self.backend.init_forward_metadata(fb)
        return fb

    def free(self):
        del self.backend, self.kv_pool, self.req_pool, self.translator
        torch.cuda.empty_cache()


class KDADecodeShim:
    """ForwardContext.attn_backend shim: RadixLinearAttention.forward calls
    get_attn_backend().forward(layer=, forward_batch=, mixed_qkv=, a=, b=);
    always DECODE here, so dispatch straight to KDAAttnBackend.forward_decode."""

    def __init__(self, kda):
        self.kda = kda

    def forward(self, layer, forward_batch, mixed_qkv, a, b, **kw):
        return self.kda.forward_decode(
            layer=layer, forward_batch=forward_batch,
            mixed_qkv=mixed_qkv, a=a, b=b,
        )


@contextlib.contextmanager
def forward_ctx(backend):
    from sglang.srt.model_executor.forward_context import (
        ForwardContext, set_forward_context,
    )
    prev = set_forward_context(ForwardContext(attn_backend=backend))
    try:
        yield
    finally:
        set_forward_context(prev)


def make_zero_allocator():
    from sglang.srt.utils.common import BumpAllocator
    return BumpAllocator(buffer_size=64, dtype=torch.float32, device="cuda")


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _cuda_time(fn, iters):
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record(); fn(); e1[i].record()
    torch.cuda.synchronize()
    t = sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))
    return t[len(t) // 2] * 1000.0  # median us


def call_layer(layer, fb, positions, hidden, ar=None):
    """One decoder-layer decode step. With an AttnResState (production
    attn_res_block_size=12) the layer runs _forward_attn_residual: `residual` carries
    the pending prefix_sum and `attn_res` the snapshot bank; without it, the plain
    residual path."""
    za = make_zero_allocator()
    return layer.forward(
        positions=positions,
        hidden_states=hidden,
        forward_batch=fb,
        residual=None if ar is None else ar.prefix,
        attn_res=None if ar is None else ar.ar,
        zero_allocator=za,
    )


class AttnResState:
    """K3 attention-residual stream state for ONE layer (kimi_k3.py:2429-2445,
    2893-2900; layers/attn_residual.py). Production builds one AttnResidual per
    forward with block_num = ceil(93/12) = 8 bank rows of [T, H]; a layer at index
    i aggregates over prev_valid_blocks = ceil(i/12) banked rows (two score
    projections [H->1] + two RMSNorms + the aggregation kernel per layer) and
    write-layers (i % 12 == 0) snapshot a new row. We seed the bank and the pending
    prefix_sum and expose reset()/state_after() for the golden."""

    K3_NUM_LAYERS = 93

    def __init__(self, layer, B, seed, device="cuda"):
        from sglang.srt.layers.attn_residual import AttnResidual
        bs = int(layer.attn_res_block_size)
        self.block_num = -(-self.K3_NUM_LAYERS // bs)
        self.nvb0 = int(layer.prev_valid_blocks)
        self.is_write_layer = bool(layer.is_block_write_layer)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 31)
        hidden0 = torch.zeros(B, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        self.ar = AttnResidual(hidden0, self.block_num, block_residual=None)
        with torch.no_grad():
            bank = torch.randn(self.ar.block_residual.shape, device=device,
                               dtype=torch.bfloat16, generator=gen) * 0.02 + 0.01
            self.ar.block_residual.copy_(bank)
            self.prefix = torch.randn(B, HIDDEN_SIZE, device=device,
                                      dtype=torch.bfloat16, generator=gen) * 0.02
        self.ar.num_valid_blocks = self.nvb0
        self._bank0 = self.ar.block_residual.clone()
        self._prefix0 = self.prefix.clone()

    def reset(self):
        with torch.no_grad():
            self.ar.block_residual.copy_(self._bank0)
            self.prefix.copy_(self._prefix0)
        self.ar.num_valid_blocks = self.nvb0

    def state_after(self):
        # the bank row a write-layer snapshots (else nothing changes)
        if self.is_write_layer:
            return {"attn_res_row": self.ar.block_residual[:, self.nvb0, :].clone()}
        return {}


def graph_capture_and_time(layer, backend, backend_for_ctx, fb, positions, hidden,
                           iters, B, warmup=3, ar=None):
    """Capture ONE decode step of the layer in a CUDA graph and time replays.

    Mirrors DecodeCudaGraphRunner (model_executor/runner/decode_cuda_graph_runner.py):
      init_cuda_graph_state(max_bs, max_num_tokens)          # static per-bs buffers
      init_forward_metadata_out_graph(fb, in_capture=True)   # metadata -> static bufs
      [model_capture_mode] warmup on a side stream, on_after_cuda_graph_warmup hook,
      torch.cuda.graph(...): init_forward_metadata_in_graph(fb) + layer.forward(...)
    Replay = init_forward_metadata_out_graph(fb) [replay-prep, done ONCE PER STEP
    for all 93 layers in production, so it is reported separately and NOT folded
    into the per-layer number] + graph.replay().

    The fb buffers are the static ones the graph was captured on (seq_lens,
    positions, out_cache_loc, input_ids); the KDA recurrent state / MLA KV pool are
    read+written in place exactly as in eager mode."""
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    out = {"graph_ok": False}
    backend.init_cuda_graph_state(B, B)
    backend.init_forward_metadata_out_graph(fb, in_capture=True)

    def run_once():
        in_graph = getattr(backend, "init_forward_metadata_in_graph", None)
        if in_graph is not None:
            in_graph(fb)
        return call_layer(layer, fb, positions, hidden, ar)

    side = torch.cuda.Stream()
    with model_capture_mode(), forward_ctx(backend_for_ctx), torch.inference_mode():
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                run_once()
            hook = getattr(backend, "on_after_cuda_graph_warmup", None)
            if hook is not None:
                hook()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            static_out = run_once()
    torch.cuda.synchronize()

    # replay-prep (metadata refresh from the live fb) -- timed separately.
    backend.init_forward_metadata_out_graph(fb)
    torch.cuda.synchronize()
    out["us_meta_prep"] = _cuda_time(lambda: backend.init_forward_metadata_out_graph(fb),
                                     max(10, iters // 4))
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    out["us_step_graph"] = _cuda_time(g.replay, iters)
    hs = static_out[0] if isinstance(static_out, tuple) else static_out
    out["graph_finite"] = bool(torch.isfinite(hs).all().item())
    out["graph_ok"] = True
    out["_graph"] = g            # kept alive by the caller for the point's lifetime
    out["_static_out"] = hs
    return out


def _point_path(path, B, L):
    """Per-point file name: fill {B}/{L} if present, else insert _B{B}_L{L} before the ext."""
    if "{B}" in path or "{L}" in path:
        return path.replace("{B}", str(B)).replace("{L}", str(L))
    root, ext = os.path.splitext(path)
    return f"{root}_B{B}_L{L}{ext or '.pt'}"


def gen_hidden(seed, B, device="cuda"):
    """Seeded decode input (one token per request), independent of the global RNG."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1000 + B)
    return torch.randn(B, HIDDEN_SIZE, device=device, dtype=torch.bfloat16,
                       generator=gen) * 0.02


def _tensor_diff(a, b):
    a = a.float(); b = b.float()
    d = (a - b).abs()
    ref = b.abs().max().item() + 1e-6
    return {"max_abs_err": d.max().item(),
            "max_rel_err": d.max().item() / ref,
            "mean_rel_err": (d.mean() / (b.abs().mean() + 1e-6)).item(),
            "nan": bool(~torch.isfinite(a).all())}


def correctness_step(layer, st, fb, positions, hidden, seed, backend_for_ctx,
                     graph=None, static_out=None, ar=None):
    """One decode step from the GOLDEN initial state and input; returns
    (output[B,H] bf16, post-step state dict). Uses the captured graph when
    available (the judged metric is graph replay, so the checked numerics must be
    the graph's), else the eager path."""
    st.reset()
    if ar is not None:
        ar.reset()
    with torch.no_grad():
        hidden.copy_(gen_hidden(seed, hidden.shape[0]))
    torch.cuda.synchronize()
    if graph is not None:
        graph.replay()
        torch.cuda.synchronize()
        out = static_out.clone()
    else:
        with forward_ctx(backend_for_ctx), torch.inference_mode():
            o = call_layer(layer, fb, positions, hidden, ar)
        torch.cuda.synchronize()
        out = (o[0] if isinstance(o, tuple) else o).clone()
    state = st.state_after()
    if ar is not None:
        state.update(ar.state_after())
    return out, state


def profile_kernels(fn, n=10):
    """torch.profiler pass over n calls of fn -> per-kernel {name,count,total_us,avg_us}
    (works for graph replays: CUDA activities inside a graph are recorded)."""
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    agg = {}
    for ev in prof.events():
        if getattr(ev, "device_type", None) is None:
            continue
        if "CUDA" not in str(ev.device_type):
            continue
        dur = ev.time_range.elapsed_us() if hasattr(ev, "time_range") else getattr(ev, "cuda_time", 0.0)
        a = agg.setdefault(ev.name, {"name": ev.name, "count": 0, "total_us": 0.0})
        a["count"] += 1
        a["total_us"] += dur
    rows = []
    for a in agg.values():
        a["count"] = a["count"] / n
        a["total_us"] = a["total_us"] / n
        a["avg_us"] = a["total_us"] / max(a["count"], 1e-9)
        rows.append(a)
    rows.sort(key=lambda r: -r["total_us"])
    return {"kernels": rows, "total_kernel_us": sum(r["total_us"] for r in rows),
            "num_launches": sum(r["count"] for r in rows)}


def nvtx_align_steps(step_fn, out_dir, B, L, n_steps, latency_mode):
    """Layer-level alignment PROBE for VibeSim's nsys pipeline: wrap n_steps decode steps
    in the NVTX ranges its parser keys on (`sglang_iteration(N): forward`,
    alignment/nsys/parse.py) and append the canonical sglang_text records
    (`VibeSimAlignmentWorker` once, one `VibeSimAlignmentIteration` per step; field set
    per alignment/profiler/record_extraction.py) to <out_dir>/server.log. Run the whole
    driver under `nsys profile --cuda-graph-trace=node ...` to get the matching .nsys-rep.
    This is a single-layer debug probe, not a full-model alignment (no K3 checkpoint)."""
    os.makedirs(out_dir, exist_ok=True)
    logp = os.path.join(out_dir, "server.log")
    with open(logp, "a") as f:
        f.write("VibeSimAlignmentWorker " + json.dumps({
            "schema_version": 1, "input_adapter": "sglang_text", "pid": os.getpid(),
            "device_id": 0, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "tp_rank": 0, "dp_rank": 0, "pp_rank": 0, "tp_size": 1, "dp_size": 1,
            "probe": f"kimi_single_layer_decode B={B} L={L} mode={latency_mode}"}) + "\n")
        torch.cuda.synchronize()
        for i in range(n_steps):
            torch.cuda.nvtx.range_push(f"sglang_iteration({i}): forward")
            step_fn()
            torch.cuda.nvtx.range_pop()
            f.write("VibeSimAlignmentIteration " + json.dumps({
                "schema_version": 1, "input_adapter": "sglang_text", "iteration_index": i,
                "prefill_tokens": 0, "decode_requests": B, "decode_tokens_scheduled": B,
                "prefill_chunk_pairs": [], "decode_kv_lens": [L] * B,
                "latency_mode": latency_mode}) + "\n")
        torch.cuda.synchronize()
    log(f"[nvtx] {n_steps} steps under sglang_iteration(N) ranges; records -> {logp}")


def time_split(layer, fb, positions, hidden, iters, warmup, attn_type="kda"):
    """Separately time self_attn vs MoE mlp vs the two RMSNorms."""
    out = {}

    def attn_once():
        za = make_zero_allocator()
        if attn_type == "mla":
            # Reuse the layer's own attn wiring: it sets up the MLA communicator
            # AttentionInputs (prepare_qkv_latent) before dispatching to the
            # KimiK3MLAAttention forward, exactly as the full-layer path does.
            return layer._run_self_attn_inner(hidden, positions, fb, za)
        return layer.self_attn(
            hidden_states=hidden, positions=positions,
            forward_batch=fb, zero_allocator=za,
        )

    def mlp_once():
        return layer.mlp(hidden, forward_batch=fb)

    def norms_once():
        h = layer.input_layernorm(hidden)
        h2, _ = layer.post_attention_layernorm(hidden, hidden.clone())
        return h2

    for fn, key in ((attn_once, "attn"), (mlp_once, "moe"), (norms_once, "norms")):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            out[f"us_{key}"] = _cuda_time(fn, iters)
        except Exception as e:
            out[f"us_{key}_error"] = f"{type(e).__name__}: {e}"
            out[f"us_{key}_tb"] = traceback.format_exc()
    return out


def time_point(cfg, sa, layer, B, seq_len, iters, warmup, split=False,
               attn_type="kda", mc=None, layer_idx=5, attention_backend="triton",
               cuda_graph=False, page_size=1, kv_dtype=torch.bfloat16, seed=0,
               golden=None, profile_kernels_out=None, nvtx_align=None):
    """golden: None | {"mode": "capture"|"replay", "path": str, "rel_err_max": float}."""
    res = {"B": B, "seq_len": seq_len, "ok": False, "attn_type": attn_type, "seed": seed}
    st = None
    graph_keepalive = None
    try:
        if attn_type == "mla":
            st = MLAState(cfg, mc, sa, layer_idx, B, seq_len,
                          attention_backend=attention_backend, page_size=page_size,
                          kv_dtype=kv_dtype, seed=seed)
            res["page_size"] = page_size
            res["kv_dtype"] = str(kv_dtype)
            fb = st.make_forward_batch(cfg)
            backend_for_ctx = st.backend        # real MLA attention backend
            res["attention_backend"] = attention_backend
            res["attention_backend_class"] = type(st.backend).__name__
        else:
            st = KDAState(cfg, sa, B, seed=seed)
            fb = st.make_forward_batch(cfg, seq_len)
            backend_for_ctx = KDADecodeShim(st.kda)
        hidden = gen_hidden(seed, B)
        positions = fb.positions
        # Production K3 runs the attention-residual stream (attn_res_block_size=12);
        # build its per-layer state so the layer takes _forward_attn_residual.
        ar = AttnResState(layer, B, seed) if getattr(layer, "use_attn_residuals", False) else None
        if ar is not None:
            res["attn_res"] = {"block_num": ar.block_num, "valid_blocks": ar.nvb0,
                               "write_layer": ar.is_write_layer}

        with forward_ctx(backend_for_ctx), torch.inference_mode():
            for _ in range(warmup):
                out = call_layer(layer, fb, positions, hidden, ar)
            torch.cuda.synchronize()

            hs = out[0] if isinstance(out, tuple) else out
            res["out_shape"] = list(hs.shape)
            res["finite"] = bool(torch.isfinite(hs).all().item())

            res["us_step"] = _cuda_time(lambda: call_layer(layer, fb, positions, hidden, ar), iters)
            res["us_token"] = res["us_step"] / B
            if attn_type == "mla":
                res["kv_gb"] = st.kv_bytes / 1e9
                # per-layer MLA-KV read = B*L*(kv_lora_rank+qk_rope)*elem_bytes.
                res["mla_kv_read_bytes"] = int(B) * int(seq_len) * 576 * (
                    1 if kv_dtype != torch.bfloat16 else 2)
                res["floor_us"] = res["mla_kv_read_bytes"] / 7e12 * 1e6  # 7 TB/s B200
            else:
                res["state_mb"] = st.state_bytes / 1e6
            res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            res["ok"] = True

            if split:
                res.update(time_split(layer, fb, positions, hidden, iters, warmup,
                                      attn_type=attn_type))
                if attn_type == "mla" and "us_attn" in res and res["floor_us"] > 0:
                    res["attn_over_floor"] = res["us_attn"] / res["floor_us"]

        if cuda_graph:
            # After eager + split so their metadata/timing are untouched by the
            # static graph buffers. Failure is reported, never silently downgraded.
            try:
                gbackend = st.backend if attn_type == "mla" else st.kda
                g = graph_capture_and_time(layer, gbackend, backend_for_ctx, fb,
                                           positions, hidden, iters, B, ar=ar)
                graph_keepalive = (g.pop("_graph"), g.pop("_static_out"))
                res.update(g)
                res["graph_speedup"] = res["us_step"] / res["us_step_graph"]
                res["us_token_graph"] = res["us_step_graph"] / B
            except Exception as e:
                res["graph_ok"] = False
                res["graph_error"] = f"{type(e).__name__}: {e}"
                res["graph_traceback"] = traceback.format_exc()
                log(f"[graph] B={B} L={seq_len} CAPTURE FAILED: {res['graph_error']}")
                log(res["graph_traceback"])

        # The judged latency: graph replay when captured, else eager.
        res["latency_us"] = res["us_step_graph"] if res.get("graph_ok") else res["us_step"]
        res["latency_mode"] = "graph" if res.get("graph_ok") else "eager"

        if profile_kernels_out:
            graph = graph_keepalive[0] if graph_keepalive else None
            if graph is not None:
                fn = graph.replay
            else:
                def fn():
                    with forward_ctx(backend_for_ctx), torch.inference_mode():
                        call_layer(layer, fb, positions, hidden, ar)
            prof = profile_kernels(fn)
            prof.update({"point": [B, seq_len], "latency_us": res["latency_us"],
                         "latency_mode": res["latency_mode"], "attn_type": attn_type})
            res["kernel_profile"] = {k: prof[k] for k in ("total_kernel_us", "num_launches")}
            res["kernel_profile_top"] = prof["kernels"][:12]
            os.makedirs(os.path.dirname(os.path.abspath(profile_kernels_out)), exist_ok=True)
            ppath = _point_path(profile_kernels_out, B, seq_len)
            json.dump(prof, open(ppath, "w"), indent=1)
            log(f"[profile] {ppath}: {prof['num_launches']:.0f} launches/step, "
                f"kernel sum {prof['total_kernel_us']:.1f}us "
                f"(latency {res['latency_us']:.1f}us {res['latency_mode']})")

        if nvtx_align:
            graph = graph_keepalive[0] if graph_keepalive else None
            if graph is not None:
                step_fn = graph.replay
            else:
                def step_fn():
                    with forward_ctx(backend_for_ctx), torch.inference_mode():
                        call_layer(layer, fb, positions, hidden, ar)
            nvtx_align_steps(step_fn, nvtx_align["dir"], B, seq_len, nvtx_align["steps"],
                             res["latency_mode"])

        if golden is not None:
            graph, static_out = graph_keepalive if graph_keepalive else (None, None)
            out, state = correctness_step(layer, st, fb, positions, hidden, seed,
                                          backend_for_ctx, graph, static_out, ar=ar)
            gpath = _point_path(golden["path"], B, seq_len)
            if golden["mode"] == "capture":
                torch.save({"B": B, "seq_len": seq_len, "seed": seed, "attn_type": attn_type,
                            "latency_mode": res["latency_mode"], "out": out.cpu(),
                            "state": {k: v.cpu() for k, v in state.items()}}, gpath)
                res["golden_path"] = gpath
                log(f"saved golden -> {gpath}")
            else:
                ref = torch.load(gpath, map_location="cuda")
                assert ref["B"] == B and ref["seq_len"] == seq_len and ref["seed"] == seed, \
                    f"golden {gpath} is for B={ref['B']} L={ref['seq_len']} seed={ref['seed']}"
                chk = _tensor_diff(out, ref["out"].cuda())
                state_chk = {k: _tensor_diff(state[k], ref["state"][k].cuda())
                             for k in state}
                tol = golden["rel_err_max"]
                state_ok = all((not c["nan"]) and c["max_rel_err"] <= tol
                               for c in state_chk.values())
                chk.update({
                    "state": state_chk, "state_ok": state_ok, "rel_err_max": tol,
                    "pass": bool((not chk["nan"]) and chk["max_rel_err"] <= tol and state_ok),
                })
                res["correctness"] = chk
                log("CHECK " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                           for k, v in chk.items() if k != "state"}))

        extra = ""
        if res.get("graph_ok"):
            extra += (f"  GRAPH us/step={res['us_step_graph']:.1f} "
                      f"(x{res['graph_speedup']:.2f} vs eager, meta_prep={res['us_meta_prep']:.1f}us"
                      f" finite={res['graph_finite']})")
        if "us_attn" in res:
            extra += (f"  attn={res.get('us_attn', float('nan')):.1f}us"
                     f" moe={res.get('us_moe', float('nan')):.1f}us"
                     f" norms={res.get('us_norms', float('nan')):.1f}us")
            if "floor_us" in res:
                extra += (f"  floor={res['floor_us']:.1f}us"
                          f" a/floor={res.get('attn_over_floor', float('nan')):.2f}x")
        memstr = (f"kv={res['kv_gb']:.3f}GB" if attn_type == "mla"
                  else f"state={res.get('state_mb', 0):.1f}MB")
        log(f"[time] B={B:5d} L={seq_len:9d}  us/step={res['us_step']:9.2f}  "
            f"us/tok={res['us_token']:8.3f}  {memstr}  "
            f"peak={res['peak_gb']:.2f}GB  finite={res['finite']}  "
            f"shape={res['out_shape']}{extra}")
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        log(f"[time] B={B} L={seq_len} FAILED: {res['error']}")
        log(res["traceback"])
    finally:
        del graph_keepalive
        if st is not None:
            st.free()
        torch.cuda.reset_peak_memory_stats()
    return res


def default_ladder(attn_type="kda"):
    ML = 1024 * 1024
    # CUDA-graph capture batch sizes (what a graph would freeze): dense low, stepped to
    # the DP-feasible max ~48. L is NOT a graph shape (runtime seq_lens), so it's a
    # profiling axis only.
    GRAPH_BS = (1, 2, 4, 8, 16, 24, 32, 40, 48)
    if attn_type == "mla":
        # MLA is CONTEXT-dependent (grows with L, unlike flat KDA).
        return [
            ("a_smoke", [(1, 4096)]),
            # captured-batch sweep at the reference context L=8192.
            ("b_graphbs", [(b, 8192) for b in GRAPH_BS]),
            # context sweep (B=1): MLA-attn us/layer scaling with L.
            ("c_ctxsweep", [(1, L) for L in (8192, 65536, 262144, ML)]),
            # feasible batch at long context (DP memory edge): 64k~16, 256k~4.
            ("d_longctx", [(16, 65536), (4, 262144)]),
        ]
    return [
        ("a_smoke", [(1, 4096)]),
        # captured-batch sweep at L=8192.
        ("b_graphbs", [(b, 8192) for b in GRAPH_BS]),
        # KDA state is O(1) in L -> confirm context-flatness (1M vs 8k).
        ("c_ctxflat", [(1, ML)]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--experts", type=int, default=56,
                    help="routed experts on this rank (896/EP; 56 @ EP=16)")
    ap.add_argument("--ep", type=int, default=16, help="modeled EP degree (metadata)")
    ap.add_argument("--layer-idx", type=int, default=-1,
                    help="decoder layer index to build; default 5 (KDA) / 4 (MLA)")
    ap.add_argument("--attn-type", type=str, default="kda", choices=["kda", "mla"],
                    help="kda = linear-attn layer (context-flat, default); "
                         "mla = full dense-attention layer (context-dependent)")
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--moe-backend", type=str, default="bf16",
                    choices=["bf16", "flashinfer_mxfp4", "deep_gemm"],
                    help="MoE expert path: bf16 (default, unchanged) or a real "
                         "mxfp4 fp4 kernel backend")
    ap.add_argument("--attention-backend", type=str, default="triton",
                    help="MLA full-attention backend (attn_type=mla only): triton "
                         "(default, dependency-free) | trtllm_mla (K3's Blackwell "
                         "production auto-resolution) | cutedsl_mla (decode under "
                         "DCP) | flashmla | flashinfer | fa3")
    ap.add_argument("--attn-heads", type=int, default=96,
                    help="attention heads on THIS rank for the profiled layer type: "
                         "96 = full set (DP attention, attn_tp=1); 12 = per-GPU heads "
                         "under the cookbook B200 recipes (attention TP8), the only "
                         "shape the Blackwell MLA decode kernels + fused KDA decode accept")
    ap.add_argument("--page-size", type=int, default=-1,
                    help="MLA KV page size; -1 = auto (64 for flashmla/cutedsl_mla/"
                         "trtllm_mla, else 1)")
    ap.add_argument("--kv-cache-dtype", type=str, default="bf16",
                    choices=["bf16", "fp8_e4m3"],
                    help="MLA latent-KV dtype (cookbook B200 recipes: fp8_e4m3)")
    ap.add_argument("--mamba-ssm-dtype", type=str, default="float32",
                    choices=["float32", "bfloat16"],
                    help="KDA recurrent-state dtype (cookbook B200 recipes: bfloat16)")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="also capture the decode step in a CUDA graph and time "
                         "replays (us_step_graph); eager us_step is still reported")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for weights, state/KV, and the decode input")
    ap.add_argument("--capture", type=str, default="",
                    help="save the golden (output + post-step state) per point to this "
                         "path (multi-point: _B{B}_L{L} suffix or {B}/{L} placeholders)")
    ap.add_argument("--replay", type=str, default="",
                    help="compare this run's output/state against a golden saved by --capture")
    ap.add_argument("--rel-err-max", type=float, default=2e-2,
                    help="CHECK passes iff max|a-b|/max|b| <= this for output and state")
    ap.add_argument("--profile-kernels", type=str, default="",
                    help="write a per-kernel torch.profiler table (JSON) per point")
    ap.add_argument("--nvtx-align", type=str, default="",
                    help="dir: run --nvtx-align-steps steps under sglang_iteration(N) NVTX "
                         "ranges and append VibeSim alignment records to <dir>/server.log "
                         "(run the driver under nsys to produce the matching .nsys-rep)")
    ap.add_argument("--nvtx-align-steps", type=int, default=20)
    ap.add_argument("--point", type=str, default="", help="single point B,seq_len")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()
    if args.layer_idx < 0:
        args.layer_idx = 4 if args.attn_type == "mla" else 5
    if args.page_size < 0:
        args.page_size = (64 if args.attention_backend in
                          ("flashmla", "cutedsl_mla", "trtllm_mla") else 1)
    kv_dtype = torch.float8_e4m3fn if args.kv_cache_dtype == "fp8_e4m3" else torch.bfloat16
    sa_kv_dtype = "fp8_e4m3" if args.kv_cache_dtype == "fp8_e4m3" else "auto"

    log("=" * 78)
    log(f"kimi_single_layer_decode v3 (K3/{args.attn_type.upper()})  "
        f"experts={args.experts} EP={args.ep} "
        f"layer_idx={args.layer_idx} iters={args.iters} warmup={args.warmup} "
        f"split={args.split} moe_backend={args.moe_backend} "
        f"attention_backend={args.attention_backend if args.attn_type == 'mla' else 'n/a'} "
        f"page_size={args.page_size} attn_heads={args.attn_heads} cuda_graph={args.cuda_graph} "
        f"kv_cache_dtype={args.kv_cache_dtype} mamba_ssm_dtype={args.mamba_ssm_dtype}")
    log(f"torch={torch.__version__}  dev={torch.cuda.get_device_name(0)}")
    import sglang
    log(f"sglang={getattr(sglang, '__version__', '??')}")
    log("=" * 78)

    cfg_dir, cfg_dict = build_config_dir(args.experts, args.ep, args.num_layers,
                                         moe_backend=args.moe_backend,
                                         attn_type=args.attn_type,
                                         target_layer=args.layer_idx,
                                         attn_heads=args.attn_heads,
                                         mamba_ssm_dtype=args.mamba_ssm_dtype)
    sa, mc, cfg = bootstrap(cfg_dir, moe_backend=args.moe_backend,
                            attn_type=args.attn_type, target_layer=args.layer_idx,
                            attention_backend=args.attention_backend,
                            page_size=args.page_size, kv_cache_dtype=sa_kv_dtype)
    layer = build_layer(cfg, args.layer_idx, moe_backend=args.moe_backend,
                        attn_type=args.attn_type, seed=args.seed)
    assert not (args.capture and args.replay), "use --capture or --replay, not both"
    golden = None
    if args.capture:
        golden = {"mode": "capture", "path": args.capture, "rel_err_max": args.rel_err_max}
    elif args.replay:
        golden = {"mode": "replay", "path": args.replay, "rel_err_max": args.rel_err_max}

    # Record the resolved MoE expert method for the report.
    try:
        _qm = layer.mlp.experts.quant_method
        moe_method = {
            "class": type(_qm).__name__,
            "use_flashinfer": getattr(_qm, "use_flashinfer", None),
            "use_marlin": getattr(_qm, "use_marlin", None),
            "_fi_kernel": getattr(_qm, "_fi_kernel", None),
            "flashinfer_mxfp4_moe_precision": getattr(
                _qm, "flashinfer_mxfp4_moe_precision", None),
        }
    except Exception as e:
        moe_method = {"error": str(e)}
    log(f"[moe] resolved expert method: {moe_method}")

    import sglang as _sg
    results = {
        "sglang_version": getattr(_sg, "__version__", "??"),
        "moe_backend": args.moe_backend,
        "hidden_act": cfg_dict["hidden_act"],
        "moe_method": moe_method,
        "experts": args.experts, "ep": args.ep, "layer_idx": args.layer_idx,
        "attn_type": args.attn_type,
        "attention_backend": args.attention_backend if args.attn_type == "mla" else None,
        "page_size": args.page_size if args.attn_type == "mla" else None,
        "attn_heads": args.attn_heads,
        "cuda_graph": args.cuda_graph,
        "kv_cache_dtype": args.kv_cache_dtype if args.attn_type == "mla" else None,
        "mamba_ssm_dtype": args.mamba_ssm_dtype if args.attn_type == "kda" else None,
        "num_attention_heads": cfg_dict["num_attention_heads"],
        "config": {k: cfg_dict[k] for k in (
            "num_experts", "num_experts_per_token", "num_shared_experts",
            "moe_intermediate_size", "routed_expert_hidden_size", "hidden_size")},
        "mla": {"num_attention_heads": cfg_dict["num_attention_heads"],
                "kv_lora_rank": cfg_dict["kv_lora_rank"],
                "qk_nope_head_dim": cfg_dict["qk_nope_head_dim"],
                "qk_rope_head_dim": cfg_dict["qk_rope_head_dim"],
                "v_head_dim": cfg_dict["v_head_dim"],
                "q_lora_rank": cfg_dict["q_lora_rank"],
                "kv_cache_dim": cfg_dict["kv_lora_rank"] + cfg_dict["qk_rope_head_dim"]}
               if args.attn_type == "mla" else None,
        "kda": {"num_heads": args.attn_heads, "head_dim": 128, "short_conv_kernel": 4,
                "gate_lower_bound": -5.0} if args.attn_type == "kda" else None,
        "points": [],
    }

    if args.point:
        # one or more "B,L" points separated by ";" (e.g. "1,1048576;16,65536")
        pts = []
        for tok in args.point.split(";"):
            B, L = (int(x) for x in tok.split(","))
            pts.append((B, L))
        rungs = [("adhoc", pts)]
    else:
        rungs = default_ladder(attn_type=args.attn_type)
        if args.only:
            keep = set(args.only.split(","))
            rungs = [r for r in rungs if r[0] in keep]

    for name, points in rungs:
        log(f"\n---- rung {name} ----")
        for (B, L) in points:
            r = time_point(cfg, sa, layer, B, L, args.iters, args.warmup,
                           split=args.split, attn_type=args.attn_type, mc=mc,
                           layer_idx=args.layer_idx,
                           attention_backend=args.attention_backend,
                           cuda_graph=args.cuda_graph, page_size=args.page_size,
                           kv_dtype=kv_dtype, seed=args.seed, golden=golden,
                           profile_kernels_out=args.profile_kernels,
                           nvtx_align=({"dir": args.nvtx_align, "steps": args.nvtx_align_steps}
                                       if args.nvtx_align else None))
            r["rung"] = name
            results["points"].append(r)
            # Machine-readable per-point line (what the judge parses).
            summary = {k: r.get(k) for k in (
                "B", "seq_len", "ok", "latency_us", "latency_mode", "us_step",
                "us_step_graph", "us_attn", "us_moe", "us_norms", "finite",
                "graph_finite", "attention_backend", "attn_heads", "seed", "error")}
            summary["attn_heads"] = args.attn_heads
            summary["point"] = [r["B"], r["seq_len"]]
            if "correctness" in r:
                summary["correctness"] = {k: v for k, v in r["correctness"].items()
                                          if k != "state"}
            log("JSON " + json.dumps(summary))

    log("\n" + "=" * 78)
    memcol = "kv_GB" if args.attn_type == "mla" else "state_MB"
    log(f"{'rung':12s} {'B':>6s} {'seq_len':>9s} {'us/step':>10s} {'us/tok':>9s} "
        f"{memcol:>9s}  status")
    for r in results["points"]:
        if r.get("ok"):
            extra = ""
            if "us_attn" in r:
                extra = (f"  attn={r.get('us_attn', float('nan')):.1f}"
                         f" moe={r.get('us_moe', float('nan')):.1f}"
                         f" norms={r.get('us_norms', float('nan')):.1f}")
                if "floor_us" in r:
                    extra += (f" floor={r['floor_us']:.1f}"
                              f" a/floor={r.get('attn_over_floor', float('nan')):.2f}x")
            memval = r.get("kv_gb", r.get("state_mb", 0.0))
            if r.get("graph_ok"):
                extra += (f"  graph={r['us_step_graph']:.1f}us (x{r['graph_speedup']:.2f})"
                          f" meta={r['us_meta_prep']:.1f}")
            elif "graph_error" in r:
                extra += f"  graph=FAILED({r['graph_error'][:60]})"
            log(f"{r['rung']:12s} {r['B']:6d} {r['seq_len']:9d} {r['us_step']:10.2f} "
                f"{r['us_token']:9.3f} {memval:9.3f}  ok{extra}")
        else:
            log(f"{r['rung']:12s} {r['B']:6d} {r['seq_len']:9d} {'-':>10s} {'-':>9s} "
                f"{'-':>9s}  FAIL {r.get('error', '')}")
    log("=" * 78)

    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=2)
        log(f"[out] wrote {args.json_out}")


if __name__ == "__main__":
    main()
