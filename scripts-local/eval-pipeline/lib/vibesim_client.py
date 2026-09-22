#!/usr/bin/env python3
"""Thin HTTP client for the VibeSim backend (host 172.17.0.1:8766).

Token: read from env VIBESIM_API_TOKEN if set (never hard-coded here). The dev
backend is currently un-gated, so calls work without it.
"""
from __future__ import annotations
import json, os, time, urllib.request, urllib.error

BASE = os.environ.get("VIBESIM_BACKEND", "http://172.17.0.1:8766")
TOKEN = os.environ.get("VIBESIM_API_TOKEN")
LUNA_MAX = {"model": "gpt-5.6-luna", "effort": "max", "serviceTier": "default"}


def _req(method: str, path: str, body: dict | None = None, timeout: float = 3600.0) -> dict:
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode()[:500]}") from e


def create_workspace(display_name: str) -> str:
    return _req("POST", "/api/agent/workspaces", {"displayName": display_name})["workspace_id"]


def create_conversation(wid: str, runtime: dict = LUNA_MAX, agent_mode: str = "orchestrated",
                        autonomous: bool = True, sandbox: str = "workspace-write") -> str:
    body = {
        "sandbox": sandbox,
        "autonomous": autonomous,
        "agentMode": agent_mode,
        "codex_runtime": {r: dict(runtime) for r in ("orchestrator", "implementer", "assistant")},
        "eager": False,
    }
    r = _req("POST", f"/api/agent/workspaces/{wid}/conversations", body)
    return r["id"]


def send_message(wid: str, cid: str, text: str, timeout: float = 5400.0) -> dict:
    """Synchronous turn; may take many minutes. Returns the full turn result
    (keys include final, ok, tool_calls, implementer_summaries, intermediate_outputs)."""
    return _req("POST", f"/api/agent/workspaces/{wid}/conversations/{cid}/messages",
                {"text": text}, timeout=timeout)


if __name__ == "__main__":
    import sys
    print(json.dumps(_req("GET", "/api/agent/workspaces"), indent=2)[:2000])
