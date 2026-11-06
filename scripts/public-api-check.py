"""Check a ServingStudio public API end to end: the catalogs, that every member
they offer is measured, one prediction with its Read more payload, and one
short simulation.

    python3 public-api-check.py BASE_URL [--commit SHA] [--quick]

BASE_URL is the origin in front of /api/public/v1: the public site, or a
container's own http://127.0.0.1:PORT. deploy-public-api.sh pipes this file to
python3 on the API host over ssh, so it uses only the standard library and
runs on the host's Python (3.10 on pakuchi).
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PREFIX = "/api/public/v1"


def call(base, path, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + PREFIX + path,
        data=data,
        headers={"content-type": "application/json"} if data else {},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            payload = answer.read()
            status = answer.status
    except urllib.error.HTTPError as error:
        payload, status = error.read(), error.code
    elapsed = time.monotonic() - start
    if status >= 300:
        raise SystemExit(f"FAIL {path}: HTTP {status} {payload[:300]!r}")
    print(f"ok   {status} {elapsed:6.2f}s {path}")
    return json.loads(payload) if payload else None


def predict(base):
    """Predict on the first single-group iteration preset, then read its
    Read more payload."""
    models = call(base, "/models")
    for checkpoint in models["checkpoints"]:
        for preset in checkpoint["presets"]:
            member = preset["members"][0]
            shape = member.get("predict") or {}
            if member.get("error") or shape.get("selector") != "iter" or shape.get("groups") != 1:
                continue
            body = {
                "preset": preset["id"],
                "params": member["params"],
                "cases": [{"groups": [{"decode_kv_lens": [1024, 2048, 4096]}]}],
                "analyze": True,
            }
            answer = call(base, "/predict", body)
            total = answer["cases"][0]["sections"][0]["nodes"][0]["ms"]
            print(f"     {preset['id']}: {total:.3f} ms")
            call(
                base,
                f"/analyzer/predictions/{answer['prediction_id']}"
                "/cases/0/subjects/optimality-waterfall/payload",
            )
            return
    raise SystemExit("FAIL /predict: no single-group iteration preset to try")


def measured(base):
    """Fail when a member lacks profile.db rows. The service still starts and
    marks such a member unavailable, so this is what stops a deploy whose
    timing change needs rows its DB does not hold. A capture whose requests do
    not fit the member (``misfit``) is a property of the preset, not a gap."""
    gaps = []
    for checkpoint in call(base, "/models")["checkpoints"]:
        for preset in checkpoint["presets"]:
            for member in preset["members"]:
                if member.get("missing"):
                    gaps.append((preset["id"], member["params"], member["missing"]))
    for preset in call(base, "/simulations/presets")["presets"]:
        for member in preset["members"]:
            for capture, why in (member.get("unavailable") or {}).items():
                if why.get("missing"):
                    gaps.append((preset["id"], {**member["params"], "capture": capture},
                                 why["missing"]))
    if gaps:
        for preset, params, missing in gaps[:20]:
            print(f"     missing {preset} {params}: {missing}")
        raise SystemExit(f"FAIL measured: {len(gaps)} members lack profile.db rows")
    print("     every member is measured")


def simulate(base, deadline_s=600):
    """Run the first sim preset's first member on its default capture."""
    presets = call(base, "/simulations/presets")["presets"]
    preset = presets[0]
    params = {axis["name"]: axis["values"][0] for axis in preset.get("axes", [])}
    started = call(
        base,
        "/simulate",
        {"preset": preset["id"], "params": params, "workload": {"source": "capture"}},
    )
    simulation_id = started["simulation_id"]
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        record = call(base, f"/simulations/{simulation_id}")
        if record["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(5)
    else:
        raise SystemExit(f"FAIL /simulate: {simulation_id} not finished in {deadline_s}s")
    if record["status"] != "done":
        raise SystemExit(f"FAIL /simulate: {record['status']}: {record.get('error')}")
    print(f"     {preset['id']} {params}: done in {time.monotonic() - start:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--commit", help="the Sim commit /health must report")
    parser.add_argument("--quick", action="store_true", help="catalogs only")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = call(base, "/health")
    print(f"     sim_commit {health.get('sim_commit')}")
    if args.commit and health.get("sim_commit") != args.commit:
        raise SystemExit(f"FAIL /health: serves {health.get('sim_commit')}, expected {args.commit}")
    for path in ("/kernels", "/workloads", "/analyzer/kernel-kinds"):
        call(base, path)
    measured(base)
    if not args.quick:
        predict(base)
        simulate(base)
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
