#!/usr/bin/env python3
"""General real-repro eval judge (host-side), model-agnostic.

Scores the agent's post-edit, recompiled tree by REPLAYING a golden captured from a
pristine before-image against the agent's container, using the model-agnostic
`extract_and_profile.py`. The target module path(s) are the eval's ANSWER KEY: they live
only here (and in the private issue config), never in the agent's task prompt.

Two gates, both measured on real hardware (GPU 3):
  1. correctness   -- replayed module output must still match the before-golden
                      (max_abs_err < err_max, no NaN). Catches a "fix" that drops work but
                      changes the result.
  2. latency       -- recovered = (before_us - measured_us) / (before_us - after_us);
                      PASS when recovered >= recover_pass, on the PRIMARY target (targets[0]).

The judge captures the golden itself on a fresh pristine before container (cached under
--golden-dir keyed by model+target), so the agent cannot tamper with the reference.

Config (JSON), e.g. issue_28103.json:
  {
    "model_snap": "/raid/.../snapshots/c1899...",
    "targets": ["model.layers.0.self_attn.q_norm", "model.layers.0.self_attn.k_norm"],
    "before_image": "rga-local/vllm-pr-28103:b200-before",
    "before_us": 13.20, "after_us": 7.52,
    "recover_pass": 0.70, "err_max": 0.05,
    "tokens": 8, "iters": 3000, "gpu": "3"
  }

Usage: judge_general.py --agent-container NAME --config issue_28103.json \
                        [--golden-dir DIR] [--out verdict.json]
"""
import argparse, json, subprocess, sys, re, pathlib, hashlib, time, os

HERE = pathlib.Path(__file__).resolve().parent
EXTRACTOR = HERE / "extract_and_profile.py"


def sh(argv, timeout=900):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return "\n".join(l for l in (p.stdout + "\n" + p.stderr).splitlines()
                     if "Wrapper:" not in l)


def parse_extractor(out):
    """Pull latency + optional CHECK dict from an extract_and_profile.py run."""
    lat, chk = None, None
    m = re.search(r'"latency_us":\s*([\d.]+)', out)
    if m:
        lat = float(m.group(1))
    m = re.search(r'CHECK (\{.*\})', out)
    if m:
        chk = json.loads(m.group(1))
    return lat, chk


def golden_path(golden_dir, model_snap, target):
    key = hashlib.sha1(f"{model_snap}|{target}".encode()).hexdigest()[:12]
    safe = target.replace(".", "_")
    return pathlib.Path(golden_dir) / f"golden_{safe}_{key}.pt"


def ensure_golden(cfg, target, golden_dir):
    """Capture a golden for `target` on a fresh pristine before container (cached)."""
    gp = golden_path(golden_dir, cfg["model_snap"], target)
    if gp.exists():
        return gp
    gp.parent.mkdir(parents=True, exist_ok=True)
    cname = f"judge_golden_{int(time.time())}"
    # Honor a slurm allocation's cgroup-scoped GPU (exported by slurm_gpu.sh) so the
    # golden-capture container binds the slurm-assigned device; else fall back to config.
    gpu = os.environ.get("DOCKER_GPU_ARG") or f'"device={cfg.get("gpu","3")}"'
    sh(["docker", "rm", "-f", cname])
    sh(["docker", "run", "-d", "--name", cname, "--gpus", gpu,
        "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "HF_HUB_OFFLINE=1",
        "-v", "/raid/yilegu/models:/raid/yilegu/models:ro",
        "-v", "/raid/yilegu/flashinfer_cache:/root/.cache/flashinfer",
        "--workdir", "/tmp", cfg["before_image"], "sleep", "infinity"])
    try:
        sh(["docker", "cp", str(EXTRACTOR), f"{cname}:/tmp/extract_and_profile.py"])
        out = sh(["docker", "exec", "--workdir", "/tmp",
                  "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "HF_HUB_OFFLINE=1", cname,
                  "python3", "/tmp/extract_and_profile.py",
                  "--model", cfg["model_snap"], "--framework", "vllm",
                  "--target", target, "--tokens", str(cfg.get("tokens", 8)),
                  "--iters", str(cfg.get("iters", 3000)),
                  "--capture", "/tmp/golden.pt"], timeout=1200)
        if "saved golden" not in out:
            raise RuntimeError(f"golden capture failed for {target}:\n{out[-1500:]}")
        sh(["docker", "cp", f"{cname}:/tmp/golden.pt", str(gp)])
    finally:
        sh(["docker", "rm", "-f", cname])
    return gp


def replay_on_agent(cfg, cont, target, gp):
    sh(["docker", "cp", str(EXTRACTOR), f"{cont}:/tmp/extract_and_profile.py"])
    sh(["docker", "cp", str(gp), f"{cont}:/tmp/golden_judge.pt"])
    out = sh(["docker", "exec", "--workdir", "/tmp",
              "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "HF_HUB_OFFLINE=1", cont,
              "python3", "/tmp/extract_and_profile.py",
              "--model", cfg["model_snap"], "--framework", "vllm",
              "--target", target, "--replay", "/tmp/golden_judge.pt",
              "--iters", str(cfg.get("iters", 3000)),
              "--err-max", str(cfg.get("err_max", 0.05))], timeout=1200)
    lat, chk = parse_extractor(out)
    return lat, chk, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-container", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--golden-dir", default="/raid/yilegu/eval_goldens")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())

    per_target, all_correct = [], True
    for i, target in enumerate(cfg["targets"]):
        gp = ensure_golden(cfg, target, args.golden_dir)
        lat, chk, raw = replay_on_agent(cfg, args.agent_container, target, gp)
        ok = bool(chk and chk.get("pass"))
        all_correct = all_correct and ok
        per_target.append({"target": target, "latency_us": lat,
                           "correctness": chk, "primary": i == 0,
                           "raw_tail": None if lat is not None else raw[-800:]})

    v = {"agent_container": args.agent_container, "targets": per_target}
    primary = per_target[0]
    lat = primary["latency_us"]
    if lat is None:
        v.update(verdict="FAIL", reason="extractor produced no latency on primary target")
        return _emit(v, args.out)

    before, after = cfg["before_us"], cfg["after_us"]
    recovered = (before - lat) / (before - after)
    v.update(before_us=before, after_us=after, latency_us=lat,
             recovered_frac=round(recovered, 3), correctness=primary["correctness"])
    gates = {"correctness": all_correct,
             "latency_recovered": recovered >= cfg.get("recover_pass", 0.70)}
    v["gates"] = gates
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
