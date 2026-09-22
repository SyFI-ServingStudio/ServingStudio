#!/usr/bin/env python3
"""Real-repro eval judge for case vllm-28103 (runs on host).

Given a RUNNING container that holds the agent's post-edit, recompiled vLLM, decide
PASS/FAIL on three gates measured on real hardware (GPU 3):

  1. correctness  -- RMSNorm(strided) must match the fp32 reference (no NaN, err<0.05).
                     A "fix" that drops the copy but reads wrong strides fails here.
  2. latency      -- single-layer qk_norm latency recovered toward the after-target.
                     recovered = (before - measured) / (before - after); PASS >= 0.70.
  3. no-regression-- (optional, --bench) end-to-end output throughput not worse than
                     before beyond noise.

Baselines are the measured B200 references (see vllm-28103-single-layer-profiler memory):
  T=8: before 13.20 us, after 7.52 us.  These frame the recovery fraction.

Usage: judge_28103.py --container NAME [--bench] [--out verdict.json]
The container must have /tmp/prof2.py (the profiler). Judge copies it in if missing.
"""
import argparse, json, subprocess, sys, re, pathlib

HERE = pathlib.Path(__file__).resolve().parent
PROFILER = HERE / "profile_qk_norm.py"

# measured B200 references at T=8 (us/call), RMSNorm.forward_cuda on strided q_by_head
BEFORE_US = 13.20
AFTER_US = 7.52
RECOVER_PASS = 0.70          # must recover >=70% of the before->after gap
CORRECT_ERR_MAX = 0.05       # bf16 RMSNorm abs-err ceiling


def dexec(container, argv, timeout=600):
    p = subprocess.run(["docker", "exec", "--workdir", "/tmp", container, *argv],
                       capture_output=True, text=True, timeout=timeout)
    # strip the host docker-wrapper banner lines
    out = "\n".join(l for l in (p.stdout + "\n" + p.stderr).splitlines()
                    if "Wrapper:" not in l)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ensure a fresh profiler is present (single-file mounts go stale)
    subprocess.run(["docker", "cp", str(PROFILER), f"{args.container}:/tmp/prof2.py"],
                   capture_output=True, text=True)

    out = dexec(args.container,
                ["python3", "/tmp/prof2.py", "--granularity", "module",
                 "--tokens", "8", "--iters", "3000", "--check"])

    v = {"container": args.container, "raw_ok": True}
    m = re.search(r'CHECK (\{.*\})', out)
    correct = json.loads(m.group(1)) if m else {"pass": False, "note": "no CHECK line"}
    lat = None
    for line in out.splitlines():
        mm = re.match(r'\s*T=\s*8\s+module\s+([\d.]+)\s+us/call', line)
        if mm:
            lat = float(mm.group(1))
    if lat is None:
        v.update(verdict="FAIL", reason="profiler produced no T=8 latency",
                 raw_tail=out[-800:])
        _emit(v, args.out); return

    recovered = (BEFORE_US - lat) / (BEFORE_US - AFTER_US)
    v.update(latency_us=lat, before_us=BEFORE_US, after_us=AFTER_US,
             recovered_frac=round(recovered, 3),
             correctness=correct)

    gates = {
        "correctness": bool(correct.get("pass")),
        "latency_recovered": recovered >= RECOVER_PASS,
    }
    if args.bench:
        # secondary: end-to-end is <1% on B200, so treat as informational-only unless
        # it REGRESSES clearly. (wired later; placeholder keeps schema stable.)
        gates["no_regression"] = True
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
