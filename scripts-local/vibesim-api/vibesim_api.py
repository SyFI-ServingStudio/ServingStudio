#!/usr/bin/env python3
"""vibesim_api v0 — the VibeSim API surface for an external optimizer agent.

Design doc: roofline_guided_agent_benchmark/docs/vibesim-api-design.md
AGENT-FACING calls (the optimizer agent sees ONLY the before-state workspace):

  simulate                 timing-predict from a sim config (cache-only, zero GPU)
  analyze                  timing breakdown: operator | run_summary | iteration
  get_optimality_analysis  scoped R0/R5/R6/R7 ladders ranked by headroom
  get_kernel_metrics       per-kernel drill-down + profiled_alternatives
  workspace_info           what the workspace offers: configs, gpu, profile coverage, runs

EVALUATOR-SIDE call (NOT exposed to the agent — the after state is ground
truth; the harness uses this to judge the agent's final candidate against the
recorded GT pair; the agent self-checks by re-running simulate + optimality
on its own edits):

  compare                  candidate-vs-baseline / candidate-vs-GT ladder + leaf-set delta

Zero-GPU guarantees baked in: every subprocess runs with CUDA_VISIBLE_DEVICES=""
and the timing-predict path errors on a profile.db cache miss (it never
JIT-profiles). All analyze verbs are pure log/JSON processing, and every
analyzer subprocess runs with the owning workspace repo as CWD (calling from
elsewhere silently degrades R5-R7 — verified 2026-09-13).

CLI:
  vibesim_api.py simulate  --workspace w_XXX --config sim-configs/before.json [--run-name NAME] [--analysis]
  vibesim_api.py analyze   --prediction w_XXX:RUN [--level operator|run_summary|iteration]
  vibesim_api.py optimality --prediction w_XXX:RUN [--scope iter] [--max-depth N]
  vibesim_api.py kernels   --prediction w_XXX:RUN --kernel-set name1,name2
  vibesim_api.py compare   --prediction-a w_XXX:RUN_A --prediction-b w_XXX:RUN_B [--scope iter]
  vibesim_api.py workspace-info --workspace w_XXX
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

WS_ROOT = Path(os.environ.get(
    "VIBESIM_WS_ROOT", "/raid/yilegu/roofline_guided_agent/VibeSimWorkspace"))
FALLBACK_ANALYZE = WS_ROOT / "agent-workspaces/w_45893bc43b51/repo/target/release/analyze"


# ---------------------------------------------------------------- resolution

def resolve_workspace(workspace: str) -> Path:
    """Workspace id (w_...) or explicit repo path -> repo Path."""
    p = Path(workspace)
    if p.is_dir() and (p / "simulator").exists():
        return p.resolve()
    cand = WS_ROOT / "agent-workspaces" / workspace / "repo"
    if cand.is_dir():
        return cand.resolve()
    raise SystemExit(f"workspace not found: {workspace}")


def resolve_prediction(prediction: str) -> tuple[Path, Path]:
    """'w_XXX:run_name' | log-dir path | prediction_id -> (repo, log_dir)."""
    if ":" in prediction and not prediction.startswith("/"):
        ws, run = prediction.split(":", 1)
        repo = resolve_workspace(ws)
        log_dir = repo / "logs" / run
        if not log_dir.is_dir():
            raise SystemExit(f"run not found: {log_dir}")
        return repo, log_dir
    p = Path(prediction)
    if p.is_dir():
        log_dir = p.resolve()
        # owning repo = nearest ancestor containing simulator/
        for anc in log_dir.parents:
            if (anc / "simulator").is_dir():
                return anc, log_dir
        raise SystemExit(f"cannot find owning repo of {log_dir}")
    if prediction.startswith("p_"):
        hits = []
        for meta in WS_ROOT.glob("agent-workspaces/*/repo/logs/*/prediction.meta.json"):
            try:
                if json.loads(meta.read_text()).get("prediction_id") == prediction:
                    hits.append(meta.parent)
            except Exception:
                continue
        if len(hits) == 1:
            return resolve_prediction(str(hits[0]))
        raise SystemExit(f"prediction_id {prediction}: {len(hits)} matches")
    raise SystemExit(f"cannot resolve prediction handle: {prediction}")


def analyze_bin(repo: Path) -> Path:
    own = repo / "target/release/analyze"
    for cand in (own, FALLBACK_ANALYZE):
        if cand.exists():
            try:
                h = subprocess.run([str(cand), "--help"], capture_output=True,
                                   text=True, timeout=30).stdout
                if "optimality-scoped" in h:
                    return cand
            except Exception:
                continue
    raise SystemExit("no analyze binary with optimality-scoped available")


def _env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""          # zero-GPU, belt and suspenders
    env.setdefault("TMPDIR", "/raid/yilegu/tmp")
    home = os.path.expanduser("~")
    env["PATH"] = f"{home}/.cargo/bin:{home}/.local/bin:" + env.get("PATH", "")
    return env


def _run(cmd: list[str], cwd: Path, timeout: int = 1800) -> tuple[int, str, str]:
    r = subprocess.run(cmd, cwd=str(cwd), env=_env(), capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


# ---------------------------------------------------------------- 1. simulate

def simulate(workspace: str, sim_config: str | dict, run_name: str | None = None,
             analysis: bool = False, strict_cache_only: bool = True) -> dict:
    repo = resolve_workspace(workspace)
    run_name = run_name or f"api_predict_{time.strftime('%Y%m%d_%H%M%S')}"

    # materialize the config with log_dir pinned to the run name
    if isinstance(sim_config, dict):
        cfg = dict(sim_config)
        src_dir = repo
    else:
        cfg_path = (repo / sim_config) if not str(sim_config).startswith("/") \
            else Path(sim_config)
        cfg = json.loads(cfg_path.read_text())
        src_dir = cfg_path.parent
    cfg["log_dir"] = f"logs/{run_name}"
    cfg_dir = repo / "sim-configs"
    cfg_dir.mkdir(exist_ok=True)
    # cases_file resolves relative to the config's own directory
    if "cases_file" in cfg and not str(cfg["cases_file"]).startswith("/"):
        cases_src = (src_dir / cfg["cases_file"]).resolve()
        (cfg_dir / f"{run_name}.cases.json").write_text(cases_src.read_text())
        cfg["cases_file"] = f"{run_name}.cases.json"
    run_cfg = cfg_dir / f"{run_name}.json"
    run_cfg.write_text(json.dumps(cfg, indent=2))

    # timing-predict is the JIT-free path: a profile.db miss ERRORS, never
    # touches a GPU. strict_cache_only documents intent; there is no lax mode.
    if not strict_cache_only:
        raise SystemExit("v0 supports strict_cache_only=true only (zero-GPU policy)")
    rc, out, err = _run(["uv", "run", "python", "-m", "launcher", "timing-predict",
                         str(run_cfg.relative_to(repo)), "--no-analyze"], repo)
    log_dir = repo / "logs" / run_name
    if rc != 0:
        missing = [l for l in (out + err).splitlines()
                   if re.search(r"cache|missing|profile", l, re.I)][:20]
        return {"ok": False, "error": "timing-predict failed", "rc": rc,
                "cache_miss_candidates": missing,
                "stderr_tail": err.splitlines()[-15:]}
    meta = json.loads((log_dir / "prediction.meta.json").read_text())
    result = {"ok": True, "prediction_id": meta.get("prediction_id"),
              "prediction": f"{repo.parents[0].name}:{run_name}"
              if repo.name == "repo" else str(log_dir),
              "run_name": run_name, "log_dir": str(log_dir), "meta": meta}
    if analysis:
        ab = analyze_bin(repo)
        rc2, out2, err2 = _run([str(ab), "run", str(log_dir)], repo)
        result["analysis_rc"] = rc2
        result["reports"] = sorted(p.name for p in (log_dir / "reports").glob("*.json"))
    return result


# ------------------------------------------------------- scoped-ladder helpers

def _scoped(repo: Path, log_dir: Path, path: str) -> dict | None:
    """Run optimality-scoped for one path; None if the path doesn't exist."""
    ab = analyze_bin(repo)
    rc, out, err = _run([str(ab), "optimality-scoped", str(log_dir),
                         "--path", path], repo, timeout=300)
    if rc != 0:
        return None
    # find the report the command just wrote (slug contains a selector hash)
    reports = sorted((log_dir / "reports").glob("optimality_scoped_*.json"),
                     key=lambda p: p.stat().st_mtime)
    for rp in reversed(reports):
        d = json.loads(rp.read_text())
        if d.get("selection", {}).get("canonical_path") == path or \
           d.get("selection", {}).get("selector", {}).get("path") == path:
            return d
    return None


def _rung(d: dict, key: str):
    v = d.get("rungs", {}).get(key)
    return v.get("value_gpu_seconds") if isinstance(v, dict) else v


def _walk(repo: Path, log_dir: Path, root: str, max_depth: int,
          cache: dict) -> list[dict]:
    """Probe-walk the CostTree: children are consecutive indices 0..N."""
    if root in cache:
        node = cache[root]
    else:
        node = _scoped(repo, log_dir, root)
        cache[root] = node
    if node is None:
        return []
    rows = [node]
    if max_depth <= 0:
        return rows
    leaves = node.get("selection", {}).get("descendant_leaves", [])
    if len(leaves) <= 1:
        return rows                                  # leaf: nothing below
    i = 0
    while True:
        child = _scoped(repo, log_dir, f"{root}/{i}")
        if child is None:
            break
        cache[f"{root}/{i}"] = child
        rows += [child] + _walk(repo, log_dir, f"{root}/{i}", max_depth - 1,
                                cache)[1:]
        i += 1
    return rows


def _row(d: dict) -> dict:
    sel = d.get("selection", {})
    label = sel.get("node_label") or ""
    r0, r5 = _rung(d, "r0_measured"), _rung(d, "r5_hardware_limit")
    r6, r7 = _rung(d, "r6_segmented_necessary"), _rung(d, "r7_scope_fused_necessary")
    leaves = sel.get("descendant_leaves", [])
    return {
        "path": sel.get("canonical_path"),
        "node": label.split(" (")[0].split(" [")[0] if label
                else (leaves[0] if len(leaves) == 1 else f"node@{sel.get('canonical_path')}"),
        "num_leaves": len(leaves),
        "r0_measured": r0, "r5_hardware_limit": r5,
        "r6_segmented_necessary": r6, "r7_scope_fused_necessary": r7,
        "headroom_r0_minus_r5": (r0 - r5) if None not in (r0, r5) else None,
        "excess_over_necessary_r5_minus_r6": (r5 - r6) if None not in (r5, r6) else None,
        "r0_over_r5": (r0 / r5) if r5 else None,
        "necessary_share_r6_over_r0": (r6 / r0) if (r0 and r6 is not None) else None,
        "descendant_leaves": leaves,
    }


# ---------------------------------------------------------------- 2. analyze

def analyze(prediction: str, level: str = "operator") -> dict:
    repo, log_dir = resolve_prediction(prediction)
    ab = analyze_bin(repo)
    reports = log_dir / "reports"

    if level == "run_summary":
        f = reports / "optimality_report.json"
        if not f.exists():
            _run([str(ab), "run", str(log_dir), "optimality"], repo)
        d = json.loads(f.read_text())
        cl = d.get("cluster", {})
        return {"level": "run_summary", "log_dir": str(log_dir),
                "unit": d.get("unit"),
                "r0_real_gpu_s": cl.get("real", cl.get("busy")),
                "r5_hardware_limit_gpu_s": cl.get("hardware_limit"),
                "r6_hardware_necessary_gpu_s": cl.get("hardware_necessary"),
                "optimality_ratio": d.get("optimality_ratio"),
                "necessary_ratio": d.get("necessary_ratio"),
                "gap_buckets_gpu_s": {k: v.get("gpu_seconds", v) if isinstance(v, dict) else v
                                      for k, v in cl.get("buckets", {}).items()},
                "worst_batching_kernels": d.get("worst_batching_kernels")}

    if level == "operator":
        f = reports / "kernel_time_share_report.json"
        if not f.exists():
            _run([str(ab), "run", str(log_dir), "kernel-time-share"], repo)
        d = json.loads(f.read_text())
        segs = sorted(d["totals"]["overall"]["segments"],
                      key=lambda s: -s["share_pct"])
        # manifest carries kind + configured backends per leaf
        cfg = {}
        for mf in (log_dir / "raw/cost_manifest").glob("worker_*.json"):
            for sec in json.loads(mf.read_text()).get("sections", []):
                for slot in sec.get("slots", []):
                    cfg[slot["name"]] = slot.get("kernel_config", {}).get("backends")
        return {"level": "operator", "log_dir": str(log_dir),
                "total_kernel_time_ms": d["totals"]["overall"]["kernel_time_ms"],
                "operators": [{"name": s["position"], "kind": s["kind"],
                               "kernel_time_ms": s["kernel_time_ms"],
                               "share_pct": s["share_pct"],
                               "backends": cfg.get(s["position"])} for s in segs]}

    if level == "iteration":
        _run([str(ab), "gen-iter-breakdown", str(log_dir), "--no-color"], repo)
        txt = (reports / "iter_breakdown.ans").read_text()
        return {"level": "iteration", "log_dir": str(log_dir),
                "tree_text": txt}

    raise SystemExit(f"unknown level {level}")


# ------------------------------------------------- 3. get_optimality_analysis

def get_optimality_analysis(prediction: str, scope: str = "iter",
                            max_depth: int = 99,
                            kernel_set: list[str] | None = None) -> dict:
    repo, log_dir = resolve_prediction(prediction)
    cache: dict = {}
    if kernel_set:
        # leaf-name mode: walk once, keep matching leaves
        nodes = _walk(repo, log_dir, scope, max_depth, cache)
        rows = [_row(d) for d in nodes]
        rows = [r for r in rows if r["num_leaves"] == 1 and
                any(r["descendant_leaves"][0] == k or
                    r["descendant_leaves"][0].endswith(k) for k in kernel_set)]
    else:
        nodes = _walk(repo, log_dir, scope, max_depth, cache)
        rows = [_row(d) for d in nodes]
    rows.sort(key=lambda r: -(r["headroom_r0_minus_r5"] or 0))
    return {"log_dir": str(log_dir), "scope": scope,
            "sorted_by": "headroom_r0_minus_r5 desc",
            "reading_guide": {
                "r6_approx_r5": "work is necessary; gap above R5 is batching/launch/scheduling — kernel swaps won't move it",
                "r6_approx_zero": "operator does work the model semantics don't require — unnecessary-work bottleneck",
                "r0_over_r5_large_r6_nonzero": "inefficient implementation of necessary work — kernel-efficiency bottleneck"},
            "nodes": rows}


# ---------------------------------------------------- 4. get_kernel_metrics

_SHAPE_SKIP = {"id", "gpu_name", "backend", "dtype", "profiler_git_hash",
               "profiler_run_at", "cuda_version", "driver_version",
               "backend_version", "verified", "time_ms", "tflops",
               "memory_bandwidth_gbps", "energy_j", "is_outlier",
               "retry_count", "outlier_reason", "created_at", "m"}


def _profiled_alternatives(repo: Path, kind: str, gpu_name: str | None) -> list[str] | None:
    db = repo / "profiling/profile.db"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if kind not in tables:
            return None
        q = f"SELECT DISTINCT backend FROM {kind}"
        args: tuple = ()
        if gpu_name:
            q += " WHERE gpu_name LIKE ?"
            args = (f"%{gpu_name.split()[-1]}%",)
        return sorted(r[0] for r in con.execute(q, args))
    except Exception:
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass


def get_kernel_metrics(prediction: str, kernel_set: list[str]) -> dict:
    repo, log_dir = resolve_prediction(prediction)
    # manifest: kind + config per leaf
    slots = {}
    gpu_name = None
    for mf in (log_dir / "raw/cost_manifest").glob("worker_*.json"):
        for sec in json.loads(mf.read_text()).get("sections", []):
            for slot in sec.get("slots", []):
                slots[slot["name"]] = slot
                gpu_name = gpu_name or slot.get("kernel_config", {}).get("gpu_name")
    # scoped ladder per requested leaf
    opt = get_optimality_analysis(prediction, scope="iter", max_depth=99,
                                  kernel_set=kernel_set)
    ladders = {r["descendant_leaves"][0]: r for r in opt["nodes"]}
    out = []
    for k in kernel_set:
        name = next((n for n in slots if n == k or n.endswith(k)), k)
        slot = slots.get(name, {})
        kind = slot.get("kind")
        row = ladders.get(name, {})
        cfgd = slot.get("kernel_config", {})
        shape = {kk: (vv.get("value") if isinstance(vv, dict) else vv)
                 for kk, vv in cfgd.items()
                 if kk not in ("backends", "gpu_name")}
        alts = _profiled_alternatives(repo, kind, gpu_name) if kind else None
        cur = (cfgd.get("backends") or [None])[0]
        out.append({"name": name, "path": row.get("path"), "kind": kind,
                    "backend": cur, "shape": shape,
                    "r0_gpu_s": row.get("r0_measured"),
                    "r5_gpu_s": row.get("r5_hardware_limit"),
                    "r6_gpu_s": row.get("r6_segmented_necessary"),
                    "profiled_alternatives": alts,
                    "has_cached_alternative":
                        bool(alts and any(a != cur for a in alts))})
    return {"log_dir": str(log_dir), "gpu": gpu_name, "kernels": out}


# ---------------------------------------------------------------- 5. compare

def compare(prediction_a: str, prediction_b: str, scope: str = "iter",
            max_depth: int = 99) -> dict:
    a = get_optimality_analysis(prediction_a, scope, max_depth)
    b = get_optimality_analysis(prediction_b, scope, max_depth)
    by_a = {r["node"]: r for r in a["nodes"]}
    by_b = {r["node"]: r for r in b["nodes"]}
    leaves_a = {l for r in a["nodes"] if r["num_leaves"] == 1
                for l in r["descendant_leaves"]}
    leaves_b = {l for r in b["nodes"] if r["num_leaves"] == 1
                for l in r["descendant_leaves"]}
    deltas = []
    for name in sorted(set(by_a) | set(by_b)):
        ra, rb = by_a.get(name), by_b.get(name)
        row = {"node": name,
               "in_a": ra is not None, "in_b": rb is not None}
        for key in ("r0_measured", "r5_hardware_limit", "r6_segmented_necessary"):
            va = ra and ra.get(key)
            vb = rb and rb.get(key)
            row[key] = {"a": va, "b": vb,
                        "delta": (vb - va) if None not in (va, vb) else None,
                        "rel_pct": (100.0 * (vb - va) / va)
                        if va and vb is not None else None}
        deltas.append(row)
    deltas.sort(key=lambda r: -abs(r["r0_measured"]["delta"] or 0))
    ra0 = by_a.get(scope.split("/")[-1]) or (a["nodes"][0] if a["nodes"] else None)
    rb0 = by_b.get(scope.split("/")[-1]) or (b["nodes"][0] if b["nodes"] else None)
    return {"a": a["log_dir"], "b": b["log_dir"], "scope": scope,
            "leaves_removed_in_b": sorted(leaves_a - leaves_b),
            "leaves_added_in_b": sorted(leaves_b - leaves_a),
            "root_r0_a": ra0 and ra0["r0_measured"],
            "root_r0_b": rb0 and rb0["r0_measured"],
            "root_r0_rel_pct": (100.0 * (rb0["r0_measured"] - ra0["r0_measured"])
                                / ra0["r0_measured"])
            if ra0 and rb0 and ra0["r0_measured"] else None,
            "node_deltas": deltas}


# ---------------------------------------------------------- 6. workspace_info

def workspace_info(workspace: str) -> dict:
    repo = resolve_workspace(workspace)
    info: dict = {"repo": str(repo)}
    spec = repo / "gpu/spec.json"
    if spec.exists():
        try:
            g = json.loads(spec.read_text())
            info["gpu_spec"] = list(g)[:8] if isinstance(g, dict) else "present"
        except Exception:
            info["gpu_spec"] = "present"
    info["model_configs"] = sorted(p.name for p in (repo / "model/config").glob("*.json"))[:40] \
        if (repo / "model/config").is_dir() else []
    info["arch_recipes"] = sorted(p.stem for p in (repo / "simulator/src/arch").glob("*.rs")
                                  if p.stem not in ("mod",)) \
        if (repo / "simulator/src/arch").is_dir() else []
    db = repo / "profiling/profile.db"
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cov = {}
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            if t.startswith("_"):
                continue
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                bks = [r[0] for r in con.execute(
                    f"SELECT DISTINCT backend FROM {t} LIMIT 12")]
                cov[t] = {"rows": n, "backends": bks}
            except Exception:
                continue
        con.close()
        info["profile_db"] = {"path": str(db), "kernel_kinds": cov}
    runs = []
    for meta in sorted((repo / "logs").glob("*/prediction.meta.json")):
        m = json.loads(meta.read_text())
        runs.append({"run_name": meta.parent.name,
                     "prediction_id": m.get("prediction_id"),
                     "arch_type": m.get("arch_type"), "gpu": m.get("gpu")})
    info["timing_predictions"] = runs
    return info


# --------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="verb", required=True)

    s = sub.add_parser("simulate")
    s.add_argument("--workspace", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--run-name")
    s.add_argument("--analysis", action="store_true")

    a = sub.add_parser("analyze")
    a.add_argument("--prediction", required=True)
    a.add_argument("--level", default="operator",
                   choices=["operator", "run_summary", "iteration"])

    o = sub.add_parser("optimality")
    o.add_argument("--prediction", required=True)
    o.add_argument("--scope", default="iter")
    o.add_argument("--max-depth", type=int, default=99)
    o.add_argument("--kernel-set")

    k = sub.add_parser("kernels")
    k.add_argument("--prediction", required=True)
    k.add_argument("--kernel-set", required=True)

    c = sub.add_parser("compare")
    c.add_argument("--prediction-a", required=True)
    c.add_argument("--prediction-b", required=True)
    c.add_argument("--scope", default="iter")
    c.add_argument("--max-depth", type=int, default=99)

    w = sub.add_parser("workspace-info")
    w.add_argument("--workspace", required=True)

    args = ap.parse_args()
    if args.verb == "simulate":
        out = simulate(args.workspace, args.config, args.run_name, args.analysis)
    elif args.verb == "analyze":
        out = analyze(args.prediction, args.level)
    elif args.verb == "optimality":
        ks = args.kernel_set.split(",") if args.kernel_set else None
        out = get_optimality_analysis(args.prediction, args.scope,
                                      args.max_depth, ks)
    elif args.verb == "kernels":
        out = get_kernel_metrics(args.prediction, args.kernel_set.split(","))
    elif args.verb == "compare":
        out = compare(args.prediction_a, args.prediction_b, args.scope,
                      args.max_depth)
    elif args.verb == "workspace-info":
        out = workspace_info(args.workspace)
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
