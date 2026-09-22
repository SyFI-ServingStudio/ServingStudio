#!/usr/bin/env python3
"""Objective, cache-only sim gate for the VibeSim optimization eval.

Given (a) an issue spec (YAML) and (b) a path to an *optimality_scoped* report
JSON produced by `analyze optimality-scoped <logdir> --label <gt_scope.label>`
for the agent's re-simulated output, decide PASS/FAIL deterministically:

  structural : the GT redundant leaves (forbidden_leaf_regex) are gone AND the
               required_leaves are still present.
  metric     : r0_measured / r5_hardware_limit each land within rel_to_after of
               the recorded GT "after" value AND >= min_drop_from_before below
               the recorded "before" value.
  necessary  : r6_segmented_necessary is ~unchanged (necessary_unchanged_rel).

No GPU, no network. Pure JSON + spec arithmetic.
"""
from __future__ import annotations
import json, re, sys, argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def _load_spec(p: Path) -> dict:
    text = p.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    raise SystemExit("PyYAML not available; run under `uv run` or `pip install pyyaml`")


def _rung(report: dict, key: str):
    r = report.get("rungs", {}).get(key)
    if isinstance(r, dict):
        return r.get("value_gpu_seconds")
    return r


def evaluate(spec: dict, report: dict) -> dict:
    g = spec["objective_gate"]
    sel = report.get("selection", {})
    leaves = sel.get("descendant_leaves", [])
    checks = []

    # --- structural ---
    if g.get("forbidden_leaf_regex"):
        forbidden = re.compile(g["forbidden_leaf_regex"])
        offending = [l for l in leaves if forbidden.search(l)]
        struct_ok = not offending
        checks.append(("structural.no_forbidden_leaves", struct_ok,
                       f"forbidden matches={offending} in {leaves}"))
    req = g.get("required_leaves", [])
    req_ok = all(any(rl == l or l.endswith(rl) for l in leaves) for rl in req)
    checks.append(("structural.required_leaves_present", req_ok,
                   f"required={req} leaves={leaves}"))
    # optional exact leaf-set equality (forbids adding leaves too)
    if g.get("expected_leaves_exact"):
        exp = set(g["expected_leaves_exact"])
        exact_ok = set(leaves) == exp
        checks.append(("structural.leaves_exact", exact_ok,
                       f"expected={sorted(exp)} got={sorted(leaves)}"))

    # --- metric ---
    tol = g["tolerance"]
    rel_after = float(tol["rel_to_after"])
    min_drop = float(tol["min_drop_from_before"])
    before, after = g["rungs"]["before"], g["rungs"]["after"]
    metric_rungs = g.get("metric_rungs", ["r0_measured", "r5_hardware_limit"])
    for key in metric_rungs:
        val = _rung(report, key)
        if val is None:
            checks.append((f"metric.{key}.present", False, "rung missing from report"))
            continue
        a, b = float(after[key]), float(before[key])
        near_after = abs(val - a) / a <= rel_after if a else (val == 0)
        below_before = val <= b * (1.0 - min_drop)
        checks.append((f"metric.{key}.near_after", near_after,
                       f"val={val:.4e} after={a:.4e} rel={abs(val-a)/a:.3f}<= {rel_after}"))
        checks.append((f"metric.{key}.below_before", below_before,
                       f"val={val:.4e} <= before*{1-min_drop}={b*(1-min_drop):.4e}"))

    # --- necessary work unchanged ---
    nrel = float(g["necessary_unchanged_rel"])
    nval = _rung(report, "r6_segmented_necessary")
    if nval is not None:
        nb = float(before["r6_segmented_necessary"])
        n_ok = abs(nval - nb) / nb <= nrel if nb else True
        checks.append(("necessary.unchanged", n_ok,
                       f"val={nval:.4e} before={nb:.4e} rel<= {nrel}"))

    # --- optional: rungs that must NOT move (e.g. r5 for kernel-efficiency fixes) ---
    for key in g.get("unchanged_rungs", []):
        val = _rung(report, key)
        if val is None:
            checks.append((f"unchanged.{key}.present", False, "rung missing from report"))
            continue
        b = float(before[key])
        ok = abs(val - b) / b <= nrel if b else (val == 0)
        checks.append((f"unchanged.{key}", ok,
                       f"val={val:.4e} before={b:.4e} rel<= {nrel}"))

    passed = all(ok for _, ok, _ in checks)
    return {"pass": passed, "checks": [
        {"name": n, "pass": ok, "detail": d} for n, ok, d in checks]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("report", type=Path, help="optimality_scoped_*.json for agent output")
    args = ap.parse_args()
    spec = _load_spec(args.spec)
    report = json.loads(args.report.read_text())
    result = evaluate(spec, report)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
