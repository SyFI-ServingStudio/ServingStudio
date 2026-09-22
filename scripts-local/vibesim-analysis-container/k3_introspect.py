#!/usr/bin/env python3
"""Phase-1 gate: verify Kimi-K3 + KDA APIs present in the sglang image (CPU-only)."""
import importlib, inspect, traceback

def show(label, fn):
    try:
        print(f"[OK ] {label}: {fn()}")
    except Exception as e:
        print(f"[ERR] {label}: {type(e).__name__}: {e}")

import sglang
print("sglang.__version__ =", getattr(sglang, "__version__", "??"))
print("sglang file =", sglang.__file__)

# --- K3 model classes ---
try:
    m = importlib.import_module("sglang.srt.models.kimi_k3")
    for cls in ("KimiK3DecoderLayer", "KimiK3DeltaAttention", "KimiK3MLAAttention",
                "KimiK3MoE", "KimiK3MLP", "KimiK3ForCausalLM"):
        obj = getattr(m, cls, None)
        print(f"  kimi_k3.{cls}: {'FOUND' if obj else 'MISSING'}")
    # decoder layer __init__ signature
    dl = getattr(m, "KimiK3DecoderLayer", None)
    if dl:
        print("  KimiK3DecoderLayer.__init__:", str(inspect.signature(dl.__init__)))
        print("  KimiK3DecoderLayer.forward:", str(inspect.signature(dl.forward)))
    da = getattr(m, "KimiK3DeltaAttention", None)
    if da:
        print("  KimiK3DeltaAttention.forward:", str(inspect.signature(da.forward)))
except Exception:
    print("kimi_k3 import FAILED:\n", traceback.format_exc())

# --- config ---
def _cfg():
    from sglang.srt.configs.kimi_linear import KimiLinearConfig
    return KimiLinearConfig
show("configs.kimi_linear.KimiLinearConfig", _cfg)

def _cfg_model():
    from sglang.srt.configs import KimiLinearConfig
    return KimiLinearConfig
show("configs.KimiLinearConfig (top)", _cfg_model)

# check config auto-registration in ModelConfig / hf registry
def _reg():
    from sglang.srt.configs.model_config import ModelConfig
    return "ModelConfig import OK"
show("model_config.ModelConfig", _reg)

# --- mamba utils / state ---
def _mamba():
    from sglang.srt.configs.mamba_utils import KimiLinearStateShape, KimiLinearCacheParams
    return (KimiLinearStateShape, KimiLinearCacheParams)
show("configs.mamba_utils KimiLinearStateShape/CacheParams", _mamba)

# --- KDA / hybrid backend ---
def _kda():
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
    return KDAAttnBackend
show("attention.linear.kda_backend.KDAAttnBackend", _kda)

def _hybrid():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase, HybridLinearAttnBackend)
    return (MambaAttnBackendBase, HybridLinearAttnBackend)
show("hybrid_linear_attn_backend Mamba/Hybrid", _hybrid)

def _radix():
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
    return RadixLinearAttention
show("radix_linear_attention.RadixLinearAttention", _radix)

# --- memory pool ---
def _pool():
    from sglang.srt.mem_cache.memory_pool import MambaPool, HybridReqToTokenPool
    return (MambaPool, HybridReqToTokenPool)
show("memory_pool MambaPool/HybridReqToTokenPool", _pool)

# --- bootstrap APIs ---
def _sa():
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    return "ServerArgs + set_global_server_args_for_scheduler"
show("server_args setters", _sa)

def _dist():
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel)
    return "dist init fns"
show("distributed.parallel_state", _dist)

def _dp():
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    return "initialize_dp_attention"
show("dp_attention", _dp)

def _rc():
    from sglang.srt.runtime_context import get_parallel, get_server_args, get_spec
    return "get_parallel/get_server_args/get_spec"
show("runtime_context", _rc)

def _fc():
    from sglang.srt.model_executor.forward_context import (
        ForwardContext, set_forward_context, get_forward_context)
    flds = [f for f in ForwardContext.__dataclass_fields__] if hasattr(ForwardContext, "__dataclass_fields__") else "n/a"
    return f"ForwardContext fields={flds}"
show("forward_context", _fc)

def _fb():
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    return "ForwardBatch/ForwardMode"
show("forward_batch_info", _fb)

def _bump():
    try:
        from sglang.srt.utils.common import BumpAllocator
    except Exception:
        from sglang.srt.utils import BumpAllocator
    return BumpAllocator
show("BumpAllocator", _bump)

print("DONE")
