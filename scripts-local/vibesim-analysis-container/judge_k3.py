#!/usr/bin/env python3
"""Open-ended judge for the Kimi-K3 single-decoder-layer case (no PR answer key).

The optimization UNIT is one KimiK3DecoderLayer (KDA or MLA variant) driven by
kimi_single_layer_decode.py (the "extractor" for this case) on ONE B200, in the
production TP8-rank shape.  The judged METRIC is the CUDA-graph replay time of the
decode step (`latency_us`, mode "graph").  Correctness = the step's output AND its
post-step state (KDA recurrent state / written MLA KV row) must match a golden
captured from the PRISTINE tree with the same seed (relative tolerance).

Integrity model (the agent never touches anything the judge measures with):
  * baseline + golden come from a fresh container of `before_image` (pristine sglang
    tree, judge-owned driver copy, flashinfer cache mounted READ-ONLY);
  * the agent's edited tree is reduced to its `.py` diff against the pristine tree,
    applied onto a fresh copy of the pristine tree (drops __pycache__/binaries/etc.),
    and that copy is bind-mounted READ-ONLY over the sglang package in a second
    fresh `before_image` container for the replay;
  * every point is measured `reps` times; the primary point must improve by
    max(min_improvement, 3*sigma_before/before); secondaries must not regress
    beyond max(0.02, 3*sigma/before); every point must pass CHECK.

Config keys (issue_k3_*.json): before_image, driver, driver_args, points{primary,
secondary}, reps, min_improvement, rel_err_max, seed, edit_tree{container_path},
mounts, shm_size.  Runner-provided args: --tree-dir (agent tree on host),
--pristine-dir (pristine tree on host).

Usage: judge_k3.py --agent-container NAME --config issue_k3_kda.json --tree-dir DIR
                   --pristine-dir DIR [--golden-dir DIR] [--out verdict.json]
"""
import argparse, hashlib, json, os, pathlib, re, shutil, statistics, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent


def sh(argv, timeout=3600, check=False):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    out = "\n".join(l for l in (p.stdout + "\n" + p.stderr).splitlines() if "Wrapper:" not in l)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(argv)}\n{out[-2000:]}")
    return out


def parse_json_lines(out):
    """Driver stdout -> {(B,L): summary dict} from the `JSON {...}` lines."""
    res = {}
    for line in out.splitlines():
        if line.startswith("JSON "):
            d = json.loads(line[5:])
            res[(int(d["B"]), int(d["seq_len"]))] = d
    return res


def points_of(cfg):
    prim = tuple(cfg["points"]["primary"])
    secs = [tuple(p) for p in cfg["points"].get("secondary", [])]
    return prim, secs


def point_arg(points):
    return ";".join(f"{b},{l}" for b, l in points)


def case_key(cfg, driver_path):
    h = hashlib.sha1()
    h.update(pathlib.Path(driver_path).read_bytes())
    h.update(json.dumps({k: cfg.get(k) for k in (
        "before_image", "driver_args", "points", "seed", "reps")}, sort_keys=True).encode())
    return h.hexdigest()[:12]


class Container:
    """A fresh `before_image` container with the judge's env; optional read-only tree mount."""

    def __init__(self, cfg, name, tree_mount=None):
        self.cfg, self.name = cfg, name
        gpu = os.environ.get("DOCKER_GPU_ARG") or f'"device={cfg.get("gpu", "3")}"'
        argv = ["docker", "run", "-d", "--name", name, "--gpus", gpu,
                "--shm-size", cfg.get("shm_size", "32g"),
                "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "HF_HUB_OFFLINE=1",
                "-e", "SGLANG_OPT_FUSED_KDA_VERIFY=0", "-e", "TOKENIZERS_PARALLELISM=false",
                "-v", "/raid/yilegu/flashinfer_cache:/root/.cache/flashinfer:ro"]
        for m in cfg.get("mounts", []):
            argv += ["-v", m]
        if tree_mount is not None:
            host, cpath = tree_mount
            argv += ["-v", f"{host}:{cpath}:ro"]
        argv += ["--workdir", "/tmp", cfg["before_image"], "sleep", "infinity"]
        sh(["docker", "rm", "-f", name])
        sh(argv, check=True)

    def cp_in(self, src, dst):
        sh(["docker", "cp", str(src), f"{self.name}:{dst}"], check=True)

    def cp_out(self, src, dst):
        sh(["docker", "cp", f"{self.name}:{src}", str(dst)], check=True)

    def exec(self, argv, timeout=3600):
        return sh(["docker", "exec", "--workdir", "/tmp", self.name] + argv, timeout=timeout)

    def rm(self):
        sh(["docker", "rm", "-f", self.name])


def run_driver(cont, cfg, driver_name, points, extra):
    argv = ["python3", f"/tmp/{driver_name}", "--point", point_arg(points),
            "--seed", str(cfg.get("seed", 0))] + cfg["driver_args"].split() + extra
    out = cont.exec(argv)
    res = parse_json_lines(out)
    missing = [p for p in points if p not in res]
    if missing:
        raise RuntimeError(f"driver produced no JSON line for {missing}:\n{out[-3000:]}")
    return res, out


def ensure_baseline(cfg, driver_path, golden_dir, points, key):
    """Golden + baseline latencies from the pristine image (cached by case key)."""
    gdir = pathlib.Path(golden_dir) / f"golden_k3_{key}"
    meta = gdir / "baseline.json"
    if meta.exists():
        return gdir, json.loads(meta.read_text())
    gdir.mkdir(parents=True, exist_ok=True)
    cont = Container(cfg, f"judge_k3_golden_{int(time.time())}")
    try:
        cont.cp_in(driver_path, f"/tmp/{driver_path.name}")
        cont.exec(["mkdir", "-p", "/tmp/golden"])
        lat = {p: [] for p in points}
        raw_tail = ""
        for rep in range(int(cfg.get("reps", 5))):
            extra = ["--capture", "/tmp/golden/golden.pt"] if rep == 0 else []
            res, out = run_driver(cont, cfg, driver_path.name, points, extra)
            raw_tail = out[-2000:]
            for p in points:
                if not res[p].get("ok"):
                    raise RuntimeError(f"baseline point {p} failed: {res[p].get('error')}\n{out[-2000:]}")
                lat[p].append(float(res[p]["latency_us"]))
            if rep == 0:
                if "saved golden" not in out:
                    raise RuntimeError(f"golden capture missing:\n{out[-2000:]}")
                for p in points:
                    cont.cp_out(f"/tmp/golden/golden_B{p[0]}_L{p[1]}.pt", gdir)
                mode = {p: res[p].get("latency_mode") for p in points}
        base = {"points": {f"{b},{l}": {"lat": v, "median": statistics.median(v),
                                          "sigma": statistics.pstdev(v) if len(v) > 1 else 0.0,
                                          "latency_mode": mode[(b, l)]}
                           for (b, l), v in lat.items()},
                "driver_args": cfg["driver_args"], "seed": cfg.get("seed", 0),
                "before_image": cfg["before_image"], "raw_tail": raw_tail}
        meta.write_text(json.dumps(base, indent=2))
        return gdir, base
    finally:
        cont.rm()


def build_judged_tree(pristine, agent_tree, out_dir):
    """Fresh pristine copy + the agent's `.py` changes only. Returns (dir, patch_text, files)."""
    pristine, agent_tree, out_dir = map(pathlib.Path, (pristine, agent_tree, out_dir))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(pristine, out_dir, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changed = []
    for f in agent_tree.rglob("*.py"):
        rel = f.relative_to(agent_tree)
        if "__pycache__" in rel.parts:
            continue
        pf = pristine / rel
        if not pf.exists() or pf.read_bytes() != f.read_bytes():
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out_dir / rel)
            changed.append(str(rel))
    # deletions of .py files by the agent are honoured too
    for pf in pristine.rglob("*.py"):
        rel = pf.relative_to(pristine)
        if "__pycache__" in rel.parts:
            continue
        if not (agent_tree / rel).exists() and (out_dir / rel).exists():
            (out_dir / rel).unlink()
            changed.append(f"-{rel}")
    patch = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc",
                            str(pristine), str(out_dir)], capture_output=True, text=True).stdout
    return out_dir, patch, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-container", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tree-dir", required=True, help="agent's edited sglang package (host path)")
    ap.add_argument("--pristine-dir", required=True, help="pristine sglang package (host path)")
    ap.add_argument("--golden-dir", default="/raid/yilegu/eval_goldens")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    driver_path = HERE / cfg.get("driver", "kimi_single_layer_decode.py")
    prim, secs = points_of(cfg)
    points = [prim] + secs
    key = case_key(cfg, driver_path)
    v = {"agent_container": args.agent_container, "case_key": key, "points": {}}

    # 1) pristine baseline + goldens
    gdir, base = ensure_baseline(cfg, driver_path, args.golden_dir, points, key)

    # 2) judged tree = pristine + agent .py diff
    judged_dir, patch, changed = build_judged_tree(
        args.pristine_dir, args.tree_dir, pathlib.Path(args.tree_dir).parent / (
            pathlib.Path(args.tree_dir).name + "_judged"))
    patch_path = judged_dir.parent / (pathlib.Path(args.tree_dir).name + "_judged.patch")
    patch_path.write_text(patch)
    v.update(changed_files=changed, diff_patch=str(patch_path), diff_lines=len(patch.splitlines()))

    # 3) replay on a fresh pristine container with the judged tree mounted read-only
    cont = Container(cfg, f"judge_k3_replay_{int(time.time())}",
                     tree_mount=(str(judged_dir), cfg["edit_tree"]["container_path"]))
    try:
        cont.cp_in(driver_path, f"/tmp/{driver_path.name}")
        cont.exec(["mkdir", "-p", "/tmp/golden"])
        for p in points:
            cont.cp_in(gdir / f"golden_B{p[0]}_L{p[1]}.pt", f"/tmp/golden/golden_B{p[0]}_L{p[1]}.pt")
        lat = {p: [] for p in points}
        correctness = {}
        for rep in range(int(cfg.get("reps", 5))):
            try:
                res, out = run_driver(cont, cfg, driver_path.name, points,
                                      ["--replay", "/tmp/golden/golden.pt",
                                       "--rel-err-max", str(cfg.get("rel_err_max", 2e-2))])
            except Exception as e:
                v.update(verdict="FAIL", reason=f"driver failed on agent tree: {e}"[:1500])
                return _emit(v, args.out)
            for p in points:
                if not res[p].get("ok"):
                    v.update(verdict="FAIL", reason=f"point {p} failed on agent tree: {res[p].get('error')}")
                    return _emit(v, args.out)
                lat[p].append(float(res[p]["latency_us"]))
                if rep == 0:
                    correctness[p] = res[p].get("correctness") or {"pass": False, "missing": True}
    finally:
        cont.rm()

    # 4) gates
    gates, per_point = {}, {}
    all_correct = True
    for p in points:
        b = base["points"][f"{p[0]},{p[1]}"]
        before, sigma = b["median"], b["sigma"]
        after = statistics.median(lat[p])
        improvement = (before - after) / before
        noise = 3.0 * sigma / before if before else 0.0
        c = correctness[p]
        ok_c = bool(c.get("pass"))
        all_correct = all_correct and ok_c
        per_point[f"{p[0]},{p[1]}"] = {
            "before_us": before, "before_sigma_us": sigma, "after_us": after,
            "after_lat": lat[p], "improvement": round(improvement, 4),
            "noise_frac_3sigma": round(noise, 4), "correctness": c,
            "latency_mode": b.get("latency_mode"), "primary": p == prim}
    pp = per_point[f"{prim[0]},{prim[1]}"]
    need = max(float(cfg.get("min_improvement", 0.05)), pp["noise_frac_3sigma"])
    gates["correctness"] = all_correct
    gates["latency_improved"] = pp["improvement"] >= need
    gates["no_secondary_regression"] = all(
        per_point[f"{p[0]},{p[1]}"]["improvement"] >= -max(0.02, per_point[f"{p[0]},{p[1]}"]["noise_frac_3sigma"])
        for p in secs)
    v.update(points=per_point, gates=gates, required_improvement=round(need, 4),
             before_us=pp["before_us"], latency_us=pp["after_us"],
             recovered_frac=pp["improvement"],  # = fractional improvement (open-ended case)
             correctness={"pass": all_correct}, golden_dir=str(gdir))
    v["verdict"] = "PASS" if all(gates.values()) else "FAIL"
    v["reason"] = ("all gates passed" if v["verdict"] == "PASS"
                   else "failed: " + ",".join(k for k, ok in gates.items() if not ok))
    _emit(v, args.out)


def _emit(v, out):
    s = json.dumps(v, indent=2)
    print(s)
    if out:
        pathlib.Path(out).write_text(s)
    sys.exit(0 if v.get("verdict") == "PASS" else 1)


if __name__ == "__main__":
    main()
