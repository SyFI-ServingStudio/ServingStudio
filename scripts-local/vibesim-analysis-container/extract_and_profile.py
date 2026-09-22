#!/usr/bin/env python3
"""General single-unit extractor + profiler for vLLM / torch models.

Model-agnostic replacement for the qk_norm-specific profiler. It does NOT encode any
operator's shapes: it *captures* the real inputs a target unit receives during a genuine
forward pass (exact strides/dtypes/contiguity the model produces), then replays just that
unit under CUDA timing. Because it profiles whatever target you name, it cannot hand the
agent the answer -- the agent must first identify the bottleneck (e.g. from the VibeSim
optimality ladder) and then point this tool at it.

vLLM note: vLLM V1 runs the model in a subprocess, so we reach it via
`llm.collective_rpc` (needs env VLLM_ALLOW_INSECURE_SERIALIZATION=1, set automatically).
Capture/time/replay all execute inside the worker where the model lives.

Target kinds (granularity is NOT limited to whole modules -- the per-case measured floor,
not the granularity, decides whether a target is a viable eval):
  module    -- dotted nn.Module path relative to the model root, e.g.
               `model.layers.0.self_attn.q_norm`. Also covers CROSS-MODULE fusions: name the
               enclosing parent module (e.g. `model.layers.0.self_attn`) and the PR's
               fused/unfused variants show up as that region's cost.
  function  -- a MICRO-OP: an importable callable named `pkg.mod:attr` (attr may be dotted,
               e.g. a `Class.method`), e.g. `sglang.srt.layers...fp8_kernel:per_token_group_quant_fp8`.
               We wrap it in the worker to capture its real inputs, then time the callable
               standalone. Catches call sites that invoke it module-qualified (`mod.f(...)`);
               a site that did `from mod import f` before we patched keeps the old binding.
  operator  -- an aten op name (torch framework only; via TorchDispatchMode).

Phases:
  --capture GOLDEN.pt   run forward, save target's (inputs, output) golden to GOLDEN.pt.
                        Use on the *before* code to record the reference output a fix must
                        preserve.
  --replay  GOLDEN.pt   load captured inputs, run the *current* impl, CUDA-time it, and
                        (if GOLDEN has an output) diff -> CHECK {max_abs_err,nan,pass}.
  (neither)             capture + time in one run (latency only; no golden correctness).
"""
import argparse, json, os
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
import torch


def detect_kind(spec):
    if "::" in spec or spec.startswith("aten") or spec.startswith("torch.ops"):
        return "operator"
    if ":" in spec:              # pkg.mod:callable  -> a function/micro-op target
        return "function"
    return "module"


def _split_callable(spec):
    # "pkg.mod:func" or "pkg.mod.attr:func" -> (import_path, attr_chain)
    mod, _, attr = spec.partition(":")
    return mod, attr


# --------------------------- worker-side (cloudpickled into the vLLM worker) -------------
def _resolve(root, path):
    obj = root
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


# _pack snapshots a tensor by cloning its ENTIRE underlying storage flat + recording the
# (shape, stride, storage_offset) view metadata. This is the only round-trip that survives
# save/load *with the exact strides intact* -- plain .cpu()/.clone() silently contiguize a
# strided view (e.g. a qkv .split() slice, stride 4096 -> 2048), which would erase the very
# non-contiguity that triggers a redundant copy. Cloning at capture time also makes the
# snapshot independent, so later in-place forward ops can't corrupt the golden.
def _pack(t):
    import torch
    if not torch.is_tensor(t):
        return t
    numel = t.untyped_storage().nbytes() // t.element_size()
    flat = torch.as_strided(t, (numel,), (1,), 0).detach().clone().cpu()
    return {"__t__": 1, "shape": tuple(t.shape), "stride": tuple(t.stride()),
            "off": int(t.storage_offset()), "flat": flat}


def _unpack(d, device="cuda"):
    import torch
    if not (isinstance(d, dict) and d.get("__t__")):
        return d
    flat = d["flat"].to(device)
    return torch.as_strided(flat, d["shape"], d["stride"], d["off"])


def _meta1(d):
    import torch
    return {"shape": list(d["shape"]), "stride": list(d["stride"]),
            "dtype": str(d["flat"].dtype),
            "contiguous": torch.as_strided(d["flat"], d["shape"], d["stride"], d["off"]
                                           ).is_contiguous()}


def _mk_install(path):
    def fn(worker):
        import torch
        mod = _resolve_w(worker.model_runner.model, path)
        mod._ep_cap = {}

        def hook(m, a, kw, out):
            if not mod._ep_cap:
                o = out[0] if isinstance(out, (tuple, list)) else out
                # pack (clone full storage) NOW so later in-place ops can't corrupt it
                mod._ep_cap["args"] = [_pack_w(x) for x in a]
                mod._ep_cap["kwargs"] = {k: _pack_w(v) for k, v in kw.items()}
                mod._ep_cap["out"] = _pack_w(o) if torch.is_tensor(o) else None
        mod._ep_h = mod.register_forward_hook(hook, with_kwargs=True)
        return True
    return fn


def _mk_time_golden(path, iters, want_golden):
    def fn(worker):
        import torch
        mod = _resolve_w(worker.model_runner.model, path)
        if hasattr(mod, "_ep_h"):
            mod._ep_h.remove()
        c = mod._ep_cap
        if not c:
            raise RuntimeError(f"target '{path}' never fired during the forward")
        a = tuple(_unpack_w(x) for x in c["args"])
        kw = {k: _unpack_w(v) for k, v in c["kwargs"].items()}
        meta = [_meta1_w(x) for x in c["args"] if isinstance(x, dict) and x.get("__t__")]
        for _ in range(50):
            mod(*a, **kw)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            mod(*a, **kw)
        e.record(); torch.cuda.synchronize()
        res = {"latency_us": s.elapsed_time(e) / iters * 1000, "input_meta": meta}
        if want_golden:
            res["golden"] = {"args": c["args"], "kwargs": c["kwargs"], "out": c["out"]}
        return res
    return fn


def _mk_replay(path, payload, iters, err_max):
    def fn(worker):
        import torch
        mod = _resolve_w(worker.model_runner.model, path)
        a = tuple(_unpack_w(x) for x in payload["args"])
        kw = {k: _unpack_w(v) for k, v in payload["kwargs"].items()}
        out = mod(*a, **kw)
        o = out[0] if isinstance(out, (tuple, list)) else out
        o = o.detach().clone() if torch.is_tensor(o) else o   # snapshot before warmup reuse
        for _ in range(50):
            mod(*a, **kw)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            mod(*a, **kw)
        e.record(); torch.cuda.synchronize()
        res = {"latency_us": s.elapsed_time(e) / iters * 1000}
        g = payload.get("out")
        if isinstance(g, dict) and g.get("__t__") and torch.is_tensor(o):
            gg = _unpack_w(g).float(); oo = o.float()
            err = (oo - gg).abs().max().item(); nan = bool(torch.isnan(oo).any())
            res["correctness"] = {"max_abs_err": err, "nan": nan,
                                  "pass": (not nan) and err < err_max}
        return res
    return fn


# ---- function/micro-op targets (wrap an importable callable, time it standalone) --------
# Unlike a module target (which persists as an attribute on the model tree), a function target
# is a free callable. We resolve `pkg.mod:attr` to (holder, leaf, orig), monkeypatch the leaf
# on its holder to a capturing wrapper, and stash state on `worker` (the worker object persists
# across collective_rpc calls in the same process, so ep_install -> generate -> ep_time share it).
def _resolve_fn_w(spec):
    import importlib
    modpath, attr = spec.split(":")
    holder = importlib.import_module(modpath)
    parts = attr.split(".")
    for p in parts[:-1]:
        holder = getattr(holder, p)
    leaf = parts[-1]
    return holder, leaf, getattr(holder, leaf)


def _mk_install_fn(spec):
    def fn(worker):
        import torch
        holder, leaf, orig = _resolve_fn_w(spec)
        cap = {}

        def wrapper(*a, **kw):
            out = orig(*a, **kw)
            if not cap:
                o = out[0] if isinstance(out, (tuple, list)) else out
                cap["args"] = [_pack_w(x) for x in a]
                cap["kwargs"] = {k: _pack_w(v) for k, v in kw.items()}
                cap["out"] = _pack_w(o) if torch.is_tensor(o) else None
            return out
        setattr(holder, leaf, wrapper)
        worker._ep_fn = {"holder": holder, "leaf": leaf, "orig": orig, "cap": cap}
        return True
    return fn


def _mk_time_golden_fn(spec, iters, want_golden):
    def fn(worker):
        import torch
        st = worker._ep_fn
        setattr(st["holder"], st["leaf"], st["orig"])   # unpatch: time the real callable
        c = st["cap"]
        if not c:
            raise RuntimeError(f"function '{spec}' never fired during the forward")
        call = st["orig"]
        a = tuple(_unpack_w(x) for x in c["args"])
        kw = {k: _unpack_w(v) for k, v in c["kwargs"].items()}
        meta = [_meta1_w(x) for x in c["args"] if isinstance(x, dict) and x.get("__t__")]
        for _ in range(50):
            call(*a, **kw)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            call(*a, **kw)
        e.record(); torch.cuda.synchronize()
        res = {"latency_us": s.elapsed_time(e) / iters * 1000, "input_meta": meta}
        if want_golden:
            res["golden"] = {"args": c["args"], "kwargs": c["kwargs"], "out": c["out"]}
        return res
    return fn


def _mk_replay_fn(spec, payload, iters, err_max):
    def fn(worker):
        import torch
        _, _, call = _resolve_fn_w(spec)          # fresh import = the CURRENT (agent's) impl
        a = tuple(_unpack_w(x) for x in payload["args"])
        kw = {k: _unpack_w(v) for k, v in payload["kwargs"].items()}
        out = call(*a, **kw)
        o = out[0] if isinstance(out, (tuple, list)) else out
        o = o.detach().clone() if torch.is_tensor(o) else o
        for _ in range(50):
            call(*a, **kw)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            call(*a, **kw)
        e.record(); torch.cuda.synchronize()
        res = {"latency_us": s.elapsed_time(e) / iters * 1000}
        g = payload.get("out")
        if isinstance(g, dict) and g.get("__t__") and torch.is_tensor(o):
            gg = _unpack_w(g).float(); oo = o.float()
            err = (oo - gg).abs().max().item(); nan = bool(torch.isnan(oo).any())
            res["correctness"] = {"max_abs_err": err, "nan": nan,
                                  "pass": (not nan) and err < err_max}
        return res
    return fn


# These helpers are injected into each worker fn's globals below (self-contained).
def _resolve_w(root, path):
    obj = root
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


for _f in (_mk_install, _mk_time_golden, _mk_replay,
           _mk_install_fn, _mk_time_golden_fn, _mk_replay_fn):
    g = _f.__globals__
    g.setdefault("_resolve_w", _resolve_w)
    g.setdefault("_resolve_fn_w", _resolve_fn_w)
    g.setdefault("_pack_w", _pack)
    g.setdefault("_unpack_w", _unpack)
    g.setdefault("_meta1_w", _meta1)


# --------------------------- drivers -----------------------------------------------------
def run_vllm(args, kind):
    if kind == "operator":
        raise SystemExit("vLLM path supports module or function targets "
                         "(operator capture needs in-process dispatch; use --framework torch).")
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, enforce_eager=True, dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=args.gpu_mem, served_model_name="m")
    path = args.target
    mk_install, mk_time, mk_replay = (
        (_mk_install_fn, _mk_time_golden_fn, _mk_replay_fn) if kind == "function"
        else (_mk_install, _mk_time_golden, _mk_replay))

    if args.replay:
        payload = torch.load(args.replay, weights_only=False)
        return llm.collective_rpc(mk_replay(path, payload, args.iters, args.err_max))[0]

    llm.collective_rpc(mk_install(path))
    llm.generate({"prompt_token_ids": list(range(1, max(2, args.tokens) + 1))},
                 SamplingParams(max_tokens=4, temperature=0.0), use_tqdm=False)
    res = llm.collective_rpc(mk_time(path, args.iters, bool(args.capture)))[0]
    if args.capture:
        torch.save(res.pop("golden"), args.capture)
    return res


def run_sglang(args, kind):
    """sglang path. The model lives in the scheduler SUBPROCESS, reachable only via
    Engine.collective_rpc(method_name, **kwargs) -> getattr(scheduler, name)(**kwargs). So,
    unlike vLLM, we cannot ship a closure: the methods ep_install/ep_time/ep_replay must
    already exist on the Scheduler (baked into the case image via sglang_profile_patch.attach;
    see build_case scripts). Results come back through a file, not the RPC return channel."""
    if kind == "operator":
        raise SystemExit("sglang path supports module or function targets, not operator.")
    from sglang.srt.entrypoints.engine import Engine
    RES = "/tmp/_ep_result.json"
    if os.path.exists(RES):
        os.remove(RES)
    ekw = dict(model_path=args.model, dtype="bfloat16", tp_size=1,
               mem_fraction_static=args.gpu_mem, disable_cuda_graph=True,
               trust_remote_code=True)
    if args.sglang_quant:
        ekw["quantization"] = args.sglang_quant
    eng = Engine(**ekw)
    try:
        if args.replay:
            eng.collective_rpc("ep_replay", target=args.target, golden_path=args.replay,
                               out_path=RES, iters=args.iters, err_max=args.err_max)
        else:
            eng.collective_rpc("ep_install", target=args.target)
            eng.generate(input_ids=[list(range(1, max(2, args.tokens) + 1))],
                         sampling_params={"max_new_tokens": 4, "temperature": 0.0})
            eng.collective_rpc("ep_time", target=args.target, out_path=RES,
                               iters=args.iters, want_golden=bool(args.capture),
                               golden_path=(args.capture or ""))
    finally:
        eng.shutdown()
    if not os.path.exists(RES):
        raise RuntimeError("sglang profiling produced no result file -- the scheduler RPC "
                           "likely raised (check that sglang_profile_patch is attached and the "
                           "target module path is valid)")
    with open(RES) as f:
        return json.load(f)


def run_torch(args, kind):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).cuda().eval()
    ids = torch.arange(1, max(2, args.tokens) + 1, device="cuda").unsqueeze(0)
    cap = {}

    if kind == "module":
        mod = _resolve(model, args.target)

        def hook(m, a, kw, out):
            if not cap:
                o = out[0] if isinstance(out, (tuple, list)) else out
                cap["args"] = tuple(x.detach() if torch.is_tensor(x) else x for x in a)
                cap["kwargs"] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kw.items()}
                cap["out"] = o.detach() if torch.is_tensor(o) else None
        h = mod.register_forward_hook(hook, with_kwargs=True)
        with torch.no_grad():
            model(ids)
        h.remove()
        call = lambda: mod(*cap["args"], **cap["kwargs"])
    else:
        from torch.utils._python_dispatch import TorchDispatchMode
        name = args.target.replace("aten::", "aten.").replace("torch.ops.", "")

        class Grab(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, a=(), kwargs=None):
                out = func(*a, **(kwargs or {}))
                if not cap and name in str(func):
                    cap["args"] = tuple(x.detach() if torch.is_tensor(x) else x for x in a)
                    cap["kwargs"] = {k: (v.detach() if torch.is_tensor(v) else v)
                                     for k, v in (kwargs or {}).items()}
                    cap["out"] = out.detach() if torch.is_tensor(out) else None
                    cap["func"] = func
                return out
        with torch.no_grad(), Grab():
            model(ids)
        call = lambda: cap["func"](*cap["args"], **cap["kwargs"])

    if not cap:
        raise RuntimeError(f"target '{args.target}' never fired during the forward")
    meta = [{"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype),
             "contiguous": x.is_contiguous()} for x in cap["args"] if torch.is_tensor(x)]
    for _ in range(50):
        call()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.iters):
        call()
    e.record(); torch.cuda.synchronize()
    return {"latency_us": s.elapsed_time(e) / args.iters * 1000, "input_meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--target-kind", default="auto",
                    choices=["auto", "module", "function", "operator"])
    ap.add_argument("--framework", default="vllm", choices=["vllm", "torch", "sglang"])
    ap.add_argument("--sglang-quant", default=None,
                    help="sglang --quantization (e.g. fp8 for dynamic fp8 of a bf16 model)")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--err-max", type=float, default=0.05)
    args = ap.parse_args()

    kind = detect_kind(args.target) if args.target_kind == "auto" else args.target_kind
    driver = {"vllm": run_vllm, "sglang": run_sglang, "torch": run_torch}[args.framework]
    res = driver(args, kind)
    res.update(target=args.target, kind=kind, framework=args.framework, tokens=args.tokens)

    tag = "REPLAY" if args.replay else "CAPTURE"
    print(f"{tag} target={args.target} kind={kind} latency={res['latency_us']:.3f}us/call"
          + (f" inputs={res['input_meta']}" if "input_meta" in res else ""))
    if "correctness" in res:
        print("CHECK " + json.dumps(res["correctness"]))
    if args.capture:
        print(f"saved golden -> {args.capture}")
    print("JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
