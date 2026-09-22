#!/usr/bin/env python3
"""Thin stdlib HTTP wrapper over the vibesim_api shim (5 agent-facing verbs).

Runs the same functions the CLI uses, so the container exposes the API two ways:
  - CLI:  python3 /opt/vibesim/app/vibesim_api.py <verb> ...   (skill docs describe this)
  - HTTP: GET/POST /api/v1/<verb>                              (cross-container calls)

The evaluator-only `compare` verb is deliberately NOT routed -- the after state is
ground truth and must never be reachable from the agent side.

No third-party deps: python3 stdlib only (matches the shim). Bind 0.0.0.0:8799.
"""
from __future__ import annotations
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import vibesim_api as api  # same dir on sys.path

HOST = os.environ.get("VIBESIM_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("VIBESIM_API_PORT", "8799"))
# The fixed before-state workspace + prediction this container serves.
WORKSPACE = os.environ.get("VIBESIM_WORKSPACE", ".")
PREBAKED = os.environ.get("VIBESIM_PREBAKED_RUN", "before")


def _default_prediction() -> str:
    # handle form the shim understands: "<w_id>:<run>" when cwd is a repo checkout
    return f"{WORKSPACE}:{PREBAKED}"


class Handler(BaseHTTPRequestHandler):
    server_version = "vibesim-analysis/1"

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter default logging
        return

    def _dispatch(self, verb: str, q: dict, body: dict):
        pred = (q.get("prediction", [None])[0] or body.get("prediction")
                or _default_prediction())
        if verb == "workspace-info":
            ws = q.get("workspace", [WORKSPACE])[0] or WORKSPACE
            return api.workspace_info(ws)
        if verb == "simulate":
            # fixed oracle: returns the baked before prediction (prebaked mode)
            ws = q.get("workspace", [WORKSPACE])[0] or body.get("workspace", WORKSPACE)
            cfg = q.get("config", [None])[0] or body.get("config", "eval-configs/before.json")
            return api.simulate(ws, cfg, body.get("run_name"))
        if verb == "analyze":
            level = q.get("level", [body.get("level", "operator")])[0]
            return api.analyze(pred, level)
        if verb == "optimality":
            scope = q.get("scope", [body.get("scope", "iter")])[0]
            md = int(q.get("max_depth", [body.get("max_depth", 99)])[0])
            ks = q.get("kernel_set", [body.get("kernel_set")])[0]
            ks = ks.split(",") if ks else None
            return api.get_optimality_analysis(pred, scope, md, ks)
        if verb == "kernels":
            ks = q.get("kernel_set", [body.get("kernel_set")])[0]
            if not ks:
                raise ValueError("kernels requires kernel_set")
            return api.get_kernel_metrics(pred, ks.split(","))
        raise KeyError(f"unknown or non-exposed verb: {verb}")

    def _handle(self, body: dict):
        u = urlparse(self.path)
        parts = [s for s in u.path.split("/") if s]
        if u.path == "/healthz":
            return self._send(200, {"ok": True, "workspace": WORKSPACE,
                                    "prebaked_run": PREBAKED})
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "v1":
            verb = parts[2]
            try:
                return self._send(200, self._dispatch(verb, parse_qs(u.query), body))
            except KeyError as e:
                return self._send(404, {"ok": False, "error": str(e)})
            except SystemExit as e:
                return self._send(400, {"ok": False, "error": str(e)})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e),
                                        "trace": traceback.format_exc().splitlines()[-5:]})
        return self._send(404, {"ok": False, "error": "not found",
                                "routes": ["/healthz", "/api/v1/{workspace-info,"
                                           "simulate,analyze,optimality,kernels}"]})

    def do_GET(self):
        self._handle({})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        self._handle(body)


def main():
    # analyzer resolves gpu/spec.json + model/work relative to CWD (the repo)
    os.chdir(os.environ.get("VIBESIM_REPO", "/opt/vibesim/repo"))
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"vibesim analysis API on http://{HOST}:{PORT}  "
          f"(workspace={WORKSPACE} prebaked={PREBAKED} cwd={os.getcwd()})",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
