#!/usr/bin/env python3
"""Single-layer profiler for vLLM PR-28103 (Qwen3 qk_norm redundant-copy).

The PR removes `input.contiguous()` in the rms_norm custom-op wrapper. That copy
fires only on a NON-contiguous input -- exactly what Qwen3's per-head q_norm/k_norm
receive, because q/k come from a `.split()` view of the fused qkv projection
(row stride = full qkv width, not q_size) then `.view(T, n_heads, head_dim)`.

This profiler reconstructs that strided tensor faithfully (no faking: the stride is
produced by the same split+view the model uses) and CUDA-times RMSNorm on it. Run the
SAME script in the before-image and the after-image; the latency delta is purely the
eliminated copy -- a clean signal for the eval judge, unlike the <1% end-to-end delta
on B200.

Granularity:
  operator      -- time torch.ops._C.rms_norm directly (before wraps it in .contiguous())
  module        -- time vllm RMSNorm.forward_cuda on the strided q_by_head (default)
Usage: profile_qk_norm.py [--granularity module|operator] [--tokens 8,64,1024] [--iters 2000]
"""
import argparse, json, time
import torch

# Qwen3-0.6B attention shape
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 16, 8, 128
Q_SIZE, KV_SIZE = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
QKV_W = Q_SIZE + 2 * KV_SIZE  # 4096
EPS = 1e-6


def make_strided_qk(T, device, dtype):
    """Reproduce the model's non-contiguous q_by_head / k_by_head views."""
    qkv = torch.randn(T, QKV_W, device=device, dtype=dtype)
    q, k, v = qkv.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)   # views -> non-contiguous
    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // HEAD_DIM, HEAD_DIM)
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // HEAD_DIM, HEAD_DIM)
    return q_by_head, k_by_head


def cuda_time(fn, iters, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", default="module", choices=["module", "operator"])
    ap.add_argument("--tokens", default="8,64,256,1024")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--check", action="store_true",
                    help="also verify RMSNorm(strided) matches a torch reference (correctness gate)")
    args = ap.parse_args()

    import vllm  # noqa
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.layernorm import RMSNorm

    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    norm = RMSNorm(HEAD_DIM, eps=EPS).to(dev, dtype)  # q_norm/k_norm are RMSNorm(head_dim)
    weight = norm.weight.data

    def ref_rmsnorm(x):
        # fp32 reference RMSNorm over the last dim (what the kernel must reproduce)
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + EPS)).to(dtype) * weight

    if args.check:
        # correctness gate: RMSNorm on a STRIDED input must match the reference.
        # a broken "fix" (e.g. reading wrong strides) diverges or NaNs here.
        qh, _ = make_strided_qk(256, dev, dtype)
        got = norm(qh)
        exp = ref_rmsnorm(qh)
        max_abs = (got.float() - exp.float()).abs().max().item()
        nan = bool(torch.isnan(got).any().item())
        # bf16 RMSNorm tolerance: ~2^-7 relative; use a generous abs floor
        ok = (not nan) and (max_abs < 0.05)
        print(f"CORRECTNESS strided max_abs_err={max_abs:.4g} nan={nan} pass={ok}")
        print("CHECK " + json.dumps({"max_abs_err": max_abs, "nan": nan, "pass": ok}))

    results = {"vllm": vllm.__version__, "granularity": args.granularity, "tokens": {}}
    for T in [int(x) for x in args.tokens.split(",")]:
        q_by_head, k_by_head = make_strided_qk(T, dev, dtype)
        assert not q_by_head.is_contiguous(), "expected strided q_by_head"

        if args.granularity == "operator":
            out = torch.empty_like(q_by_head)
            def fn(x=q_by_head, o=out):
                # mirrors the before/after rms_norm wrapper body: before adds .contiguous()
                torch.ops._C.rms_norm(o, x, weight, EPS)
        else:  # module: the real RMSNorm.forward_cuda path the model calls
            def fn(x=q_by_head):
                norm(x)

        ms = cuda_time(fn, args.iters)
        # also time both q and k (the model does two norms per layer)
        results["tokens"][T] = {"ms_per_call": ms, "contiguous": q_by_head.is_contiguous()}
        print(f"T={T:5d}  {args.granularity}  {ms*1000:8.3f} us/call")

    print("JSON " + json.dumps(results))


if __name__ == "__main__":
    main()
