#!/usr/bin/env python3
"""
kimi_single_layer_decode.py  (v1)

Synthetic single-layer DECODE-latency driver for a Kimi-K2 / DeepSeek-V3 MoE
decoder layer, run against a synthesized long-context MLA KV cache on ONE B200.

We construct exactly ONE `DeepseekV2DecoderLayer` (representative MoE layer,
layer_id == first_k_dense_replace, so `.mlp` is `DeepseekV2MoE`) with RANDOM
bf16 weights (quant_config=None -- no fp8 for v1), synthesize an
`MLATokenToKVPool` + `ReqToTokenPool` that maps B requests to L contiguous
token slots each, build a bs=B DECODE `ForwardBatch`, drive the TRITON MLA
attention backend by hand, and CUDA-event time the layer's decode step.

Everything runs inside `lmsysorg/sglang:v0.5.16-runtime` with one GPU.

The heavy global state that sglang's model layers assume is bootstrapped
minimally here (single-process, world_size=1, tp=1, EP off):
  * ServerArgs published via runtime_context (get_server_args())
  * torch.distributed + model-parallel groups (init_distributed_environment /
    initialize_model_parallel / initialize_dp_attention)
  * a per-forward ForwardContext(attn_backend=...) published via
    set_forward_context so RadixAttention/get_attn_backend() resolve.

The attention backend normally reads a ModelRunner; we hand it a light stub
carrying exactly the attributes TritonAttnBackend.__init__ touches.
"""
import argparse
import contextlib
import json
import os
import sys
import time
import glob
import traceback

import torch


DSV3_SNAP_GLOB = "/raid/hf/hub/models--deepseek-ai--DeepSeek-V3/snapshots/*"
HIDDEN_SIZE = 7168  # DeepSeek-V3 / Kimi-K2 hidden dim


def log(*a):
    print(*a, flush=True)


def build_config_dir(kimi: bool, experts=None, config_json=None, out_dir="/tmp/kimi_cfg"):
    """Build the layer config. Base = a supplied --config-json (swappable, e.g. a
    future Kimi-K3 config) else the local DeepSeek-V3 config.json. Apply Kimi-K2
    deltas if --kimi, then an optional --experts override (EP-rank modeling: build
    only n_routed_experts/EP experts on this GPU). Drop fp8 quant (bf16 random init
    for v1) and the trust-remote-code auto_map (use sglang's built-in parser)."""
    if config_json:
        base = json.load(open(config_json))
        log(f"[cfg] base = {config_json} (supplied)")
    else:
        snap = sorted(glob.glob(DSV3_SNAP_GLOB))[0]
        base = json.load(open(os.path.join(snap, "config.json")))
        log(f"[cfg] base = {snap}/config.json")
    if kimi:
        # Kimi-K2 deltas relative to DeepSeek-V3
        base["n_routed_experts"] = 384       # 256 -> 384
        base["num_attention_heads"] = 64     # 128 -> 64
        base["num_key_value_heads"] = 64
        base["first_k_dense_replace"] = 1    # 3 -> 1
        log("[cfg] applied Kimi-K2 deltas: n_routed_experts=384, num_attention_heads=64, first_k_dense_replace=1")
    if experts is not None:
        # EP-rank modeling: this GPU holds only n_routed_experts experts (e.g.
        # 384/16=24 for EP=16). Faithful MoE-footprint-per-rank; also much lighter/faster.
        base["n_routed_experts"] = experts
        # keep top-k <= experts so routing is valid on the shrunken expert set
        base["num_experts_per_tok"] = min(base.get("num_experts_per_tok", 8), experts)
        # grouped-topk (noaux_tc) needs n_group | n_routed_experts and topk_group<=n_group
        base["n_group"] = 1
        base["topk_group"] = 1
        log(f"[cfg] EP-rank override: n_routed_experts={experts} "
            f"num_experts_per_tok={base['num_experts_per_tok']} n_group=1 topk_group=1")
    base.pop("quantization_config", None)    # bf16 random init, no fp8 for v1
    base.pop("auto_map", None)               # use built-in deepseek_v3 parser
    os.makedirs(out_dir, exist_ok=True)
    json.dump(base, open(os.path.join(out_dir, "config.json"), "w"))
    return out_dir, base


def bootstrap(cfg_dir):
    """Publish ServerArgs, build ModelConfig, init distributed + parallel + dp-attn."""
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    torch.cuda.set_device(0)

    sa = ServerArgs(
        model_path=cfg_dir,
        tp_size=1,
        dtype="bfloat16",
        device="cuda",
        attention_backend="triton",
        trust_remote_code=True,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        enable_dp_attention=False,
    )
    set_global_server_args_for_scheduler(sa)
    log(f"[boot] ServerArgs published; attention_backends={sa.get_attention_backends()}")

    mc = ModelConfig(model_path=cfg_dir, dtype="bfloat16", trust_remote_code=True)
    log(f"[boot] ModelConfig: attn_arch={mc.attention_arch} heads={mc.num_attention_heads} "
        f"kv_heads={mc.get_num_kv_heads(1)} v_head_dim={mc.v_head_dim} kv_lora={mc.kv_lora_rank} "
        f"qk_rope={mc.qk_rope_head_dim} ctx={mc.context_len} scaling={mc.scaling:.5f}")

    port = 29500 + (os.getpid() % 2000)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    initialize_dp_attention(sa, mc)
    log("[boot] distributed + model-parallel + dp-attention initialized (tp=1, world=1)")
    return sa, mc


def build_layer(mc, layer_id):
    """Construct ONE DeepseekV2DecoderLayer (random bf16), assert .mlp is MoE."""
    from sglang.srt.models.deepseek_v2 import (
        DeepseekV2DecoderLayer,
        DeepseekV2MoE,
    )
    torch.manual_seed(0)
    layer = DeepseekV2DecoderLayer(
        config=mc.hf_config,
        layer_id=layer_id,
        quant_config=None,
    )
    is_moe = isinstance(layer.mlp, DeepseekV2MoE)
    log(f"[layer] built DeepseekV2DecoderLayer(layer_id={layer_id}); "
        f"mlp type = {type(layer.mlp).__name__}  is_DeepseekV2MoE={is_moe}")
    assert is_moe, f"layer.mlp is {type(layer.mlp).__name__}, expected DeepseekV2MoE"

    layer = layer.to(torch.bfloat16).cuda().eval()
    # Random-init every param + float buffer so outputs are finite / non-degenerate.
    np = 0
    with torch.no_grad():
        for p in layer.parameters():
            if p.is_floating_point():
                p.normal_(0.0, 0.02)
                np += p.numel()
        for b in layer.buffers():
            if b.is_floating_point():
                b.normal_(0.0, 0.02)
    log(f"[layer] random-init {np/1e9:.3f} B float params; moved to cuda/bf16/eval")
    return layer


class KVHolder:
    """Owns the per-point MLA KV pool + req->token map + attn backend stub."""

    def __init__(self, mc, sa, layer_id, B, L, device="cuda"):
        from sglang.srt.mem_cache.memory_pool import (
            MLATokenToKVPool,
            ReqToTokenPool,
        )
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        self.B, self.L = B, L
        n_tokens = B * L
        # MLA KV pool: buffer shape (size+page_size, 1, kv_lora_rank+qk_rope) = (.,1,576)
        self.kv_pool = MLATokenToKVPool(
            size=n_tokens,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=mc.kv_lora_rank,          # 512
            qk_rope_head_dim=mc.qk_rope_head_dim,  # 64
            layer_num=1,
            device=device,
            enable_memory_saver=False,
            start_layer=layer_id,
            end_layer=layer_id + 1,
        )
        # Seeded random KV context (finite bf16).
        torch.manual_seed(0)
        with torch.no_grad():
            self.kv_pool.kv_buffer[0].normal_(0.0, 0.02)

        # req -> token slot map: req r owns contiguous slots [r*L, r*L+L)
        self.req_pool = ReqToTokenPool(
            size=B, max_context_len=L, device=device, enable_memory_saver=False
        )
        rows = torch.arange(B, device=device, dtype=torch.int32).view(B, 1) * L
        cols = torch.arange(L, device=device, dtype=torch.int32).view(1, L)
        # req_to_token has a +1 padding row at index 0; real reqs live at 1..B.
        self.req_pool.req_to_token[1 : B + 1, :L] = rows + cols

        # KV bytes for the single layer.
        kb = self.kv_pool.kv_buffer[0]
        self.kv_bytes = kb.numel() * kb.element_size()

        # Light ModelRunner stub carrying exactly what TritonAttnBackend reads.
        class _MR:
            pass

        mr = _MR()
        mr.req_to_token_pool = self.req_pool
        mr.token_to_kv_pool = self.kv_pool
        mr.token_to_kv_pool_allocator = None      # translate_kv_loc -> None (skipped)
        mr.sliding_window_size = None
        mr.page_size = 1
        mr.server_args = sa
        mr.model_config = mc
        mr.device = device
        mr.gpu_id = 0
        mr.is_draft_worker = False
        self.backend = TritonAttnBackend(mr)
        log(f"[kv] B={B} L={L} tokens={n_tokens} kv_bytes={self.kv_bytes/1e9:.3f}GB "
            f"backend={type(self.backend).__name__} use_mla={getattr(self.backend,'use_mla',None)}")

    def make_forward_batch(self, mc):
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        B, L = self.B, self.L
        dev = "cuda"
        # req r's current token slot = last position slot = r*L + (L-1)
        req_pool_indices = (torch.arange(B, device=dev, dtype=torch.int64) + 1)  # rows 1..B
        seq_lens = torch.full((B,), L, device=dev, dtype=torch.int64)
        seq_lens_cpu = torch.full((B,), L, dtype=torch.int64)
        positions = torch.full((B,), L - 1, device=dev, dtype=torch.int64)
        out_cache_loc = (req_pool_indices - 1) * L + (L - 1)  # int64, current-token slot
        input_ids = torch.randint(0, mc.vocab_size, (B,), device=dev, dtype=torch.int64)

        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=B,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=B * L,
            out_cache_loc=out_cache_loc,
            positions=positions,
        )
        fb.req_to_token_pool = self.req_pool
        fb.token_to_kv_pool = self.kv_pool
        fb.attn_backend = self.backend
        self.backend.init_forward_metadata(fb)
        return fb

    def free(self):
        del self.backend, self.kv_pool, self.req_pool
        torch.cuda.empty_cache()


@contextlib.contextmanager
def forward_ctx(backend):
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        set_forward_context,
    )
    prev = set_forward_context(ForwardContext(attn_backend=backend))
    try:
        yield
    finally:
        set_forward_context(prev)


def make_zero_allocators():
    from sglang.srt.utils.common import BumpAllocator
    za = BumpAllocator(buffer_size=64, dtype=torch.float32, device="cuda")
    return za


def call_layer(layer, fb, positions, hidden_states):
    """Reset the fresh bump allocator each call and drive the layer forward."""
    za = make_zero_allocators()
    out = layer.forward(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=fb,
        residual=None,
        zero_allocator=za,
        gemm_output_zero_allocator=None,
    )
    return out


def time_point(layer, mc, sa, layer_id, B, L, iters, warmup, split=False):
    """Build KV+fb for (B,L) and time the decode step. Returns a result dict."""
    res = {"B": B, "L": L, "ok": False}
    holder = None
    try:
        holder = KVHolder(mc, sa, layer_id, B, L)
        fb = holder.make_forward_batch(mc)
        torch.manual_seed(0)
        hidden = torch.randn(B, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) * 0.02
        positions = fb.positions

        with forward_ctx(holder.backend), torch.inference_mode():
            # warmup
            for _ in range(warmup):
                out = call_layer(layer, fb, positions, hidden)
            torch.cuda.synchronize()

            # sanity: finite output
            hs = out[0] if isinstance(out, tuple) else out
            finite = bool(torch.isfinite(hs).all().item())
            res["out_shape"] = list(hs.shape)
            res["finite"] = finite

            # timed
            ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                ev0[i].record()
                call_layer(layer, fb, positions, hidden)
                ev1[i].record()
            torch.cuda.synchronize()
            times_ms = [ev0[i].elapsed_time(ev1[i]) for i in range(iters)]
            times_ms.sort()
            med_ms = times_ms[len(times_ms) // 2]
            mean_ms = sum(times_ms) / len(times_ms)

            res["us_step"] = med_ms * 1000.0
            res["us_step_mean"] = mean_ms * 1000.0
            res["us_token"] = (med_ms * 1000.0) / B
            res["kv_gb"] = holder.kv_bytes / 1e9
            res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            res["ok"] = True

            if split:
                res.update(time_split(layer, fb, positions, hidden, holder.backend, iters, warmup))

        log(f"[time] B={B:5d} L={L:8d}  us/step={res['us_step']:10.2f}  "
            f"us/tok={res['us_token']:8.3f}  kv={res['kv_gb']:.2f}GB  peak={res['peak_gb']:.2f}GB  "
            f"finite={res.get('finite')}  shape={res.get('out_shape')}")
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        log(f"[time] B={B} L={L} FAILED: {res['error']}")
        log(res["traceback"])
    finally:
        if holder is not None:
            holder.free()
        torch.cuda.reset_peak_memory_stats()
    return res


def time_split(layer, fb, positions, hidden, backend, iters, warmup):
    """Separately time self_attn vs mlp for a rough attn-vs-moe split.
    Re-drives the two sub-blocks with the same residual-stream input `hidden`."""
    out = {}
    # ---- attention only ----
    def attn_once():
        za = make_zero_allocators()
        return layer.self_attn(
            positions=positions,
            hidden_states=hidden,
            forward_batch=fb,
            zero_allocator=za,
            layer_scatter_modes=layer.layer_scatter_modes,
        )
    def mlp_once():
        return layer.mlp(hidden, fb, None)

    for fn, key in ((attn_once, "attn"), (mlp_once, "moe")):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                e0[i].record(); fn(); e1[i].record()
            torch.cuda.synchronize()
            t = sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))
            out[f"us_{key}"] = t[len(t)//2] * 1000.0
        except Exception as e:
            out[f"us_{key}_error"] = f"{type(e).__name__}: {e}"
    return out


def default_ladder():
    ML = 1024 * 1024  # 1,048,576
    ladder = []
    ladder.append(("a_smoke", [(1, 4096)]))
    ladder.append(("b_1M", [(1, ML)]))
    ladder.append(("c_sweep", [(b, 8192) for b in (128, 256, 512, 1024, 2048)]))
    ladder.append(("d_longctx", [(32, ML)]))
    return ladder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kimi", dest="kimi", action="store_true", default=True)
    ap.add_argument("--no-kimi", dest="kimi", action="store_false")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--split", action="store_true", help="also time attn vs moe split")
    ap.add_argument("--only", type=str, default="", help="run only these ladder rungs (comma list of a_smoke,b_1M,c_sweep,d_longctx)")
    ap.add_argument("--point", type=str, default="", help="single ad-hoc point B,L (overrides ladder)")
    ap.add_argument("--experts", type=int, default=None,
                    help="override n_routed_experts (EP-rank modeling, e.g. 24 for K2 384/EP=16)")
    ap.add_argument("--config-json", type=str, default="",
                    help="swappable base config.json (e.g. a future Kimi-K3 config)")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    log("=" * 78)
    log(f"kimi_single_layer_decode v1  kimi={args.kimi}  iters={args.iters} warmup={args.warmup} split={args.split}")
    log(f"torch={torch.__version__}  cuda_dev={torch.cuda.get_device_name(0)}")
    log("=" * 78)

    cfg_dir, cfg = build_config_dir(args.kimi, experts=args.experts,
                                    config_json=(args.config_json or None))
    layer_id = cfg["first_k_dense_replace"]
    sa, mc = bootstrap(cfg_dir)
    layer = build_layer(mc, layer_id)

    results = {"kimi": args.kimi, "layer_id": layer_id,
               "config": {k: cfg[k] for k in ("n_routed_experts", "num_attention_heads",
                          "first_k_dense_replace", "num_experts_per_tok", "moe_intermediate_size",
                          "hidden_size", "kv_lora_rank", "qk_rope_head_dim")},
               "points": []}

    if args.point:
        B, L = (int(x) for x in args.point.split(","))
        rungs = [("adhoc", [(B, L)])]
    else:
        rungs = default_ladder()
        if args.only:
            keep = set(args.only.split(","))
            rungs = [r for r in rungs if r[0] in keep]

    for name, points in rungs:
        log(f"\n---- rung {name} ----")
        for (B, L) in points:
            r = time_point(layer, mc, sa, layer_id, B, L, args.iters, args.warmup, split=args.split)
            r["rung"] = name
            results["points"].append(r)

    # summary table
    log("\n" + "=" * 78)
    log(f"{'rung':10s} {'B':>6s} {'L':>9s} {'us/step':>11s} {'us/tok':>9s} {'kv_GB':>7s} {'peak_GB':>8s}  status")
    for r in results["points"]:
        if r.get("ok"):
            extra = ""
            if "us_attn" in r or "us_moe" in r:
                extra = f"  attn={r.get('us_attn',float('nan')):.1f}us moe={r.get('us_moe',float('nan')):.1f}us"
            log(f"{r['rung']:10s} {r['B']:6d} {r['L']:9d} {r['us_step']:11.2f} {r['us_token']:9.3f} "
                f"{r['kv_gb']:7.2f} {r['peak_gb']:8.2f}  ok{extra}")
        else:
            log(f"{r['rung']:10s} {r['B']:6d} {r['L']:9d} {'-':>11s} {'-':>9s} {'-':>7s} {'-':>8s}  FAIL {r.get('error','')}")
    log("=" * 78)

    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=2)
        log(f"[out] wrote {args.json_out}")


if __name__ == "__main__":
    main()
