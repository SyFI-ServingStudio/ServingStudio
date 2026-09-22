"""In-worker profiling methods for sglang, attached to the Scheduler class.

sglang's offline Engine reaches the model only through `Engine.collective_rpc(method: str,
**kwargs)`, which the scheduler subprocess dispatches as `getattr(self, method)(**kwargs)`
(see Scheduler.handle_rpc_request). Unlike vLLM's `collective_rpc(callable)` we therefore
cannot ship a closure -- the method must already exist on the Scheduler instance. This module
attaches three such methods and is imported for its side effect (baked into the case image so
it runs inside the scheduler subprocess).

The RPC return channel only carries (success, message_str), so results (latency, correctness,
golden) are written to `out_path` on the container filesystem; the driver reads them back.

Capture/pack/replay logic mirrors extract_and_profile.py's vLLM worker fns exactly: _pack
snapshots the full underlying storage flat + (shape,stride,offset) so strided views survive
save/load, and clones at capture time so later in-place forward ops can't corrupt the golden.
"""
import json
import torch


def _resolve(root, path):
    obj = root
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _pack(t):
    if not torch.is_tensor(t):
        return t
    numel = t.untyped_storage().nbytes() // t.element_size()
    flat = torch.as_strided(t, (numel,), (1,), 0).detach().clone().cpu()
    return {"__t__": 1, "shape": tuple(t.shape), "stride": tuple(t.stride()),
            "off": int(t.storage_offset()), "flat": flat}


def _unpack(d, device="cuda"):
    if not (isinstance(d, dict) and d.get("__t__")):
        return d
    flat = d["flat"].to(device)
    return torch.as_strided(flat, d["shape"], d["stride"], d["off"])


def _meta1(d):
    return {"shape": list(d["shape"]), "stride": list(d["stride"]),
            "dtype": str(d["flat"].dtype),
            "contiguous": torch.as_strided(d["flat"], d["shape"], d["stride"], d["off"]
                                           ).is_contiguous()}


def _resolve_fn(spec):
    # "pkg.mod:attr" (attr may be dotted, e.g. Class.method) -> (holder, leaf, orig callable)
    import importlib
    modpath, attr = spec.split(":")
    holder = importlib.import_module(modpath)
    parts = attr.split(".")
    for p in parts[:-1]:
        holder = getattr(holder, p)
    leaf = parts[-1]
    return holder, leaf, getattr(holder, leaf)


def _model(self):
    # tp_worker.model_runner.model is the model root inside the scheduler (see weight_updater).
    for attr in ("tp_worker", "draft_worker"):
        w = getattr(self, attr, None)
        mr = getattr(w, "model_runner", None) if w is not None else None
        if mr is not None and getattr(mr, "model", None) is not None:
            return mr.model
    raise RuntimeError("could not locate model_runner.model on the scheduler")


def ep_install(self, target):
    if ":" in target:                       # function/micro-op target
        import sys
        holder, leaf, orig = _resolve_fn(target)
        cap = {}

        def wrapper(*a, **kw):
            out = orig(*a, **kw)
            if not cap:
                o = out[0] if isinstance(out, (tuple, list)) else out
                cap["args"] = [_pack(x) for x in a]
                cap["kwargs"] = {k: _pack(v) for k, v in kw.items()}
                cap["out"] = _pack(o) if torch.is_tensor(o) else None
            return out
        # Rebind EVERY alias of `orig` across all loaded modules by IDENTITY, not just the
        # defining module. Call sites that did `from mod import fn` (or `... as alias`) hold
        # their own reference in their module namespace; patching only `holder.leaf` would miss
        # them. Scanning sys.modules for `is orig` catches them all (and any renamed alias).
        sites = []
        for m in list(sys.modules.values()):
            if m is None:
                continue
            try:
                d = vars(m)
            except TypeError:
                continue
            for an, av in list(d.items()):
                if av is orig:
                    setattr(m, an, wrapper)
                    sites.append((m, an))
        if not sites:                        # fallback: at least patch the defining module
            setattr(holder, leaf, wrapper); sites.append((holder, leaf))
        self._ep_fn = {"orig": orig, "cap": cap, "sites": sites}
        return
    mod = _resolve(_model(self), target)
    mod._ep_cap = {}

    def hook(m, a, kw, out):
        if not mod._ep_cap:
            o = out[0] if isinstance(out, (tuple, list)) else out
            mod._ep_cap["args"] = [_pack(x) for x in a]
            mod._ep_cap["kwargs"] = {k: _pack(v) for k, v in kw.items()}
            mod._ep_cap["out"] = _pack(o) if torch.is_tensor(o) else None
    mod._ep_h = mod.register_forward_hook(hook, with_kwargs=True)


def _time(call, iters):
    for _ in range(50):
        call()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        call()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


def ep_time(self, target, out_path, iters=3000, want_golden=False, golden_path=""):
    if ":" in target:                       # function/micro-op target
        st = self._ep_fn
        for m, an in st["sites"]:            # unpatch every alias: time the real callable
            setattr(m, an, st["orig"])
        c = st["cap"]
        if not c:
            raise RuntimeError(f"function '{target}' never fired during the forward")
        call = st["orig"]
        a = tuple(_unpack(x) for x in c["args"])
        kw = {k: _unpack(v) for k, v in c["kwargs"].items()}
        meta = [_meta1(x) for x in c["args"] if isinstance(x, dict) and x.get("__t__")]
        lat = _time(lambda: call(*a, **kw), int(iters))
        res = {"latency_us": lat, "input_meta": meta}
        if want_golden and golden_path:
            torch.save({"args": c["args"], "kwargs": c["kwargs"], "out": c["out"]}, golden_path)
            res["golden_path"] = golden_path
        with open(out_path, "w") as f:
            json.dump(res, f)
        return
    mod = _resolve(_model(self), target)
    if hasattr(mod, "_ep_h"):
        mod._ep_h.remove()
    c = getattr(mod, "_ep_cap", None)
    if not c:
        raise RuntimeError(f"target '{target}' never fired during the forward")
    a = tuple(_unpack(x) for x in c["args"])
    kw = {k: _unpack(v) for k, v in c["kwargs"].items()}
    meta = [_meta1(x) for x in c["args"] if isinstance(x, dict) and x.get("__t__")]
    lat = _time(lambda: mod(*a, **kw), int(iters))
    res = {"latency_us": lat, "input_meta": meta}
    if want_golden and golden_path:
        torch.save({"args": c["args"], "kwargs": c["kwargs"], "out": c["out"]}, golden_path)
        res["golden_path"] = golden_path
    with open(out_path, "w") as f:
        json.dump(res, f)


def ep_replay(self, target, golden_path, out_path, iters=3000, err_max=0.05):
    payload = torch.load(golden_path, weights_only=False)
    if ":" in target:                       # function/micro-op target -> current (agent's) impl
        _, _, call = _resolve_fn(target)
        mod = None
    else:
        mod = _resolve(_model(self), target)
        call = mod
    a = tuple(_unpack(x) for x in payload["args"])
    kw = {k: _unpack(v) for k, v in payload["kwargs"].items()}
    out = call(*a, **kw)
    o = out[0] if isinstance(out, (tuple, list)) else out
    o = o.detach().clone() if torch.is_tensor(o) else o
    lat = _time(lambda: call(*a, **kw), int(iters))
    res = {"latency_us": lat}
    g = payload.get("out")
    if isinstance(g, dict) and g.get("__t__") and torch.is_tensor(o):
        gg = _unpack(g).float(); oo = o.float()
        err = (oo - gg).abs().max().item(); nan = bool(torch.isnan(oo).any())
        res["correctness"] = {"max_abs_err": err, "nan": nan,
                              "pass": (not nan) and err < float(err_max)}
    with open(out_path, "w") as f:
        json.dump(res, f)


def ep_scan_install(self, needle="per_token_group_quant"):
    """Diagnostic: wrap EVERY callable across sys.modules whose attribute name contains `needle`
    (comma-separated OK) and record which ones actually fire during the subsequent forward, with
    the first call's input shapes. Answers 'does ANY group-quant kernel fire on this model?'
    without guessing the exact callable name."""
    import sys
    needles = [n.strip().lower() for n in needle.split(",") if n.strip()]
    fired = {}                       # "module.attr" -> {"calls": n, "shapes": [...]}
    sites = []                       # (module, attr, orig) for restore

    def mk(name, orig):
        def w(*a, **kw):
            r = fired.setdefault(name, {"calls": 0, "shapes": None})
            r["calls"] += 1
            if r["shapes"] is None:
                r["shapes"] = [list(x.shape) for x in a if torch.is_tensor(x)]
            return orig(*a, **kw)
        return w

    seen = set()
    for m in list(sys.modules.values()):
        if m is None:
            continue
        try:
            d = vars(m)
        except TypeError:
            continue
        for an, av in list(d.items()):
            if (callable(av) and not isinstance(av, type)
                    and any(nd in an.lower() for nd in needles) and id(av) not in seen):
                nm = f"{getattr(m,'__name__','?')}.{an}"
                setattr(m, an, mk(nm, av)); sites.append((m, an, av)); seen.add(id(av))
    self._ep_scan = {"fired": fired, "sites": sites}


def ep_scan_report(self, out_path):
    sc = getattr(self, "_ep_scan", None) or {"fired": {}, "sites": []}
    for m, an, orig in sc["sites"]:
        setattr(m, an, orig)
    with open(out_path, "w") as f:
        json.dump({"patched": len(sc["sites"]), "fired": sc["fired"]}, f, indent=2)


def attach(scheduler_cls):
    """Attach the three profiling methods onto the Scheduler class.

    Called from an appendix appended to sglang's scheduler.py at image-build time (where the
    Scheduler class is already defined), so this module never imports scheduler.py itself --
    no circular import. Importing this module has no side effect on its own.
    """
    scheduler_cls.ep_install = ep_install
    scheduler_cls.ep_time = ep_time
    scheduler_cls.ep_replay = ep_replay
    scheduler_cls.ep_scan_install = ep_scan_install
    scheduler_cls.ep_scan_report = ep_scan_report
