#!/usr/bin/env python3
"""GPU proxy for the Kimi-K3 agent container (slurm mode).

This container has NO GPU. Installed as /tmp/kimi_single_layer_decode.py it forwards every
invocation of the measurement driver to a host-side broker (k3_gpu_broker.py) that runs the REAL
driver (read-only copy: /tmp/k3_driver_src/kimi_single_layer_decode.py) in a fresh sglang
container on a slurm-allocated B200, with YOUR edited tree mounted at the same path
(/sgl-workspace/sglang/python/sglang) and /workspace/opt_run shared. stdout/stderr of the GPU job
are streamed back here; the exit code is the driver's.

  python3 /tmp/kimi_single_layer_decode.py --point ... <flags>     # driver call (transparent)
  /tmp/gpu_run.sh <any command>                                     # generic GPU command in the
                                                                    # same measurement container
Paths: /workspace/opt_run is the shared directory. Any `/tmp/<x>` output path you pass is
rewritten to /workspace/opt_run/tmp/<x> (and symlinked back to /tmp/<x> in this container), so
`--capture /tmp/golden.pt` keeps working. Environment variables set in this shell do NOT reach the
GPU job (only edits to the source tree carry over, exactly like the judge).
Each call costs slurm queue + container start (~1-3 min): batch your points into one call.
"""
import json, os, pathlib, shutil, sys, time, uuid

BROKER = pathlib.Path(os.environ.get("K3_GPU_BROKER_DIR", "/workspace/.gpu"))
SHARED = "/workspace/opt_run"
DRIVER_SRC = "/tmp/k3_driver_src/kimi_single_layer_decode.py"
TIMEOUT = int(os.environ.get("K3_GPU_TIMEOUT", "5400"))


PATH_FLAGS = {"--capture", "--replay", "--profile-kernels", "--json-out", "--nvtx-align",
              "--flashinfer-autotune-cache", "-o", "--output", "--out"}


def _rewrite(tok, cwd, prev=None):
    """/tmp/<x> -> /workspace/opt_run/tmp/<x>; a RELATIVE value of a known output flag when cwd is
    /tmp likewise (numbers such as `--hidden-scale 1.0` are never touched)."""
    if tok.startswith("/tmp/") and not tok.startswith("/tmp/k3_driver_src"):
        new = f"{SHARED}/tmp/{tok[len('/tmp/'):]}"
    elif cwd == "/tmp" and prev in PATH_FLAGS and not tok.startswith("-") and not tok.startswith("/"):
        new = f"{SHARED}/tmp/{tok}"
    else:
        return tok
    pathlib.Path(new).parent.mkdir(parents=True, exist_ok=True)
    link = pathlib.Path("/tmp") / pathlib.Path(new).relative_to(f"{SHARED}/tmp")
    try:
        if not link.exists() and not link.is_symlink():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(new)
    except OSError:
        pass
    return new


def main():
    argv = sys.argv[1:]
    cwd = os.getcwd()
    if argv[:1] == ["--exec"]:
        kind, cmd = "exec", argv[1:]
        if cmd[:1] == ["--"]:
            cmd = cmd[1:]
        if not (cwd.startswith("/sgl-workspace") or cwd.startswith(SHARED)):
            cwd = SHARED
        cmd = [_rewrite(t, os.getcwd(), cmd[i - 1] if i else None) for i, t in enumerate(cmd)]
    else:
        kind = "driver"
        cmd = ["python3", "/tmp/kimi_single_layer_decode.py"] + \
              [_rewrite(t, cwd, argv[i - 1] if i else None) for i, t in enumerate(argv)]
        cwd = SHARED if cwd == "/tmp" else cwd
    if not cmd:
        print(__doc__); return 2
    (BROKER / "req").mkdir(parents=True, exist_ok=True)
    (BROKER / "res").mkdir(parents=True, exist_ok=True)
    rid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    req = {"id": rid, "kind": kind, "argv": cmd, "cwd": cwd, "created": time.time()}
    tmp = BROKER / "req" / f"{rid}.json.tmp"
    tmp.write_text(json.dumps(req)); tmp.rename(BROKER / "req" / f"{rid}.json")
    out, rc_f, job_f = (BROKER / "res" / f"{rid}.{s}" for s in ("out", "rc", "job"))
    print(f"[gpu-proxy] request {rid}: {' '.join(cmd)}  (cwd {cwd}) -> slurm B200 job; waiting", file=sys.stderr, flush=True)
    t0, pos, announced = time.time(), 0, False
    while True:
        if not announced and job_f.exists():
            print(f"[gpu-proxy] slurm job {job_f.read_text().strip()} submitted; output streams below", file=sys.stderr, flush=True)
            announced = True
        if out.exists():
            with open(out, "rb") as f:
                f.seek(pos); chunk = f.read(); pos += len(chunk)
            if chunk:
                sys.stdout.buffer.write(chunk); sys.stdout.flush()
        if rc_f.exists():
            time.sleep(0.5)                      # let the last flush land
            with open(out, "rb") as f:
                f.seek(pos); chunk = f.read()
            if chunk:
                sys.stdout.buffer.write(chunk); sys.stdout.flush()
            try:
                rc = int(rc_f.read_text().strip() or 1)
            except ValueError:
                rc = 1
            print(f"[gpu-proxy] done rc={rc} in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
            return rc
        if time.time() - t0 > TIMEOUT:
            print(f"[gpu-proxy] TIMEOUT after {TIMEOUT}s waiting for the GPU job (id {rid})", file=sys.stderr, flush=True)
            return 124
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
