#!/usr/bin/env python3
"""Host-side GPU broker for one Kimi-K3 trial in slurm mode (user 2026-09-26: submit GPU work as
slurm `main` jobs and hold a GPU only while actually profiling).

The agent container runs WITHOUT a GPU; its /tmp/kimi_single_layer_decode.py is k3_gpu_shim.py,
which drops request files into <dir>/req/. This daemon turns each request into ONE slurm job
(`sbatch --wait slurm_gpu.sh k3_slurm_step.sh --docker <docker-run args>`): a fresh
`before_image` container bound to the slurm-allocated GPU, the agent's edited tree mounted at the
edit path, /workspace/opt_run shared, per-trial sglang JIT cache persisted. stdout goes to
<dir>/res/<id>.out (slurm -o; the shim tails it live), the job's exit code to <id>.rc.
Requests are served strictly one at a time (the agent is sequential; two concurrent measurements
would also double-book GPUs).

Usage: k3_gpu_broker.py --dir D --tree HOST_TREE --edit-path /sgl-workspace/sglang/python/sglang
         --driver HOST_DRIVER --opt-run HOST_DIR --jit-cache HOST_DIR --image lmsysorg/sglang:v0.5.20
         [--mount host:cont[:ro] ...] [--shm 32g] [--partition main] [--job-prefix k3m_case]
Stops when <dir>/STOP exists.
"""
import argparse, json, os, pathlib, shlex, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def submit_and_wait(cmd, out, job_f, poll=5):
    """sbatch (no --wait: its job-id line would sit in a pipe buffer), then poll squeue until the
    job leaves the queue. Exit code: the `[step] EXIT_RC=` marker k3_slurm_step.sh appends to the
    output file, else scontrol's ExitCode, else 1."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        out.write_text(f"[broker] sbatch failed rc={p.returncode}: {p.stdout} {p.stderr}\n")
        return 1, "?"
    job = p.stdout.strip().split(";")[0]
    job_f.write_text(job)
    while True:
        q = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
        if q.returncode != 0 or not q.stdout.strip():
            break
        time.sleep(poll)
    time.sleep(2)                      # let slurm flush the output file
    rc = None
    if out.exists():
        tail = out.read_bytes()[-4000:].decode(errors="replace")
        for line in reversed(tail.splitlines()):
            if line.startswith("[step] EXIT_RC="):
                try:
                    rc = int(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
                break
    if rc is None:
        s = subprocess.run(["scontrol", "show", "job", job], capture_output=True, text=True).stdout
        for tok in s.split():
            if tok.startswith("ExitCode="):
                try:
                    rc = int(tok.split("=")[1].split(":")[0])
                except ValueError:
                    pass
        with open(out, "a") as f:
            f.write(f"\n[broker] no EXIT_RC marker (job killed/timed out?); scontrol ExitCode -> rc={rc}\n")
    return (1 if rc is None else rc), job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--tree", required=True)
    ap.add_argument("--edit-path", default="/sgl-workspace/sglang/python/sglang")
    ap.add_argument("--driver", required=True); ap.add_argument("--opt-run", required=True)
    ap.add_argument("--jit-cache", required=True); ap.add_argument("--image", required=True)
    ap.add_argument("--mount", action="append", default=[]); ap.add_argument("--shm", default="32g")
    ap.add_argument("--partition", default="main"); ap.add_argument("--job-prefix", default="k3m")
    ap.add_argument("--flashinfer-cache", default="/raid/yilegu/flashinfer_cache")
    a = ap.parse_args()
    d = pathlib.Path(a.dir); req_d, res_d = d / "req", d / "res"
    req_d.mkdir(parents=True, exist_ok=True); res_d.mkdir(parents=True, exist_ok=True)
    stop = d / "STOP"
    pathlib.Path(a.opt_run).mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.jit_cache).mkdir(parents=True, exist_ok=True)
    log(f"broker up: dir={d} tree={a.tree} image={a.image} partition={a.partition}")
    while not stop.exists():
        reqs = sorted(req_d.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not reqs:
            time.sleep(2); continue
        rp = reqs[0]
        try:
            r = json.loads(rp.read_text())
        except Exception as e:
            log(f"bad request {rp}: {e}"); rp.rename(rp.with_suffix(".bad")); continue
        rid = r["id"]
        out, rc_f, job_f = (res_d / f"{rid}.{s}" for s in ("out", "rc", "job"))
        docker_args = ["-e", "CUDA_VISIBLE_DEVICES=0", "-e", "HF_HUB_OFFLINE=1",
                       "-e", "SGLANG_OPT_FUSED_KDA_VERIFY=0", "-e", "TOKENIZERS_PARALLELISM=false",
                       "--shm-size", a.shm,
                       "-v", f"{a.tree}:{a.edit_path}",
                       "-v", f"{a.driver}:/tmp/kimi_single_layer_decode.py:ro",
                       "-v", f"{a.opt_run}:/workspace/opt_run",
                       "-v", f"{a.jit_cache}:/root/.cache/sglang/jit",
                       "-v", f"{a.flashinfer_cache}:/root/.cache/flashinfer"]
        for m in a.mount:
            docker_args += ["-v", m]
        docker_args += ["--workdir", r.get("cwd") or "/workspace/opt_run", a.image] + list(r["argv"])
        cmd = ["sbatch", "--parsable", f"--partition={a.partition}",
               f"--job-name={a.job_prefix}_{rid[-6:]}", f"--output={out}",
               str(HERE / "slurm_gpu.sh"), str(HERE / "k3_slurm_step.sh"), "--docker"] + docker_args
        log(f"[{rid}] {r['kind']}: {' '.join(shlex.quote(x) for x in r['argv'])[:300]}")
        rp.rename(rp.with_suffix(".running"))
        t0 = time.time()
        try:
            rc, job = submit_and_wait(cmd, out, job_f)
            rc_f.write_text(str(rc))
            log(f"[{rid}] done rc={rc} in {time.time()-t0:.0f}s (job {job})")
            rp.with_suffix(".running").rename(rp.with_suffix(".done"))
        except OSError as e:
            # 2026-09-27: /raid hit 100% for ~15 min and an ENOSPC on the output file killed the
            # broker, stranding the agent's call. Never die on I/O: requeue the request and retry.
            log(f"[{rid}] I/O error ({e}); requeue and retry in 60s")
            try:
                rp.with_suffix(".running").rename(rp)
            except OSError:
                pass
            time.sleep(60)
    log("broker stopped (STOP file)")


if __name__ == "__main__":
    sys.exit(main())
