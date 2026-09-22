#!/usr/bin/env python3
"""Patch a copy of the canonical vibesim_api.py for container use.

Adds a prebaked-simulate short circuit: when VIBESIM_PREBAKED_RUN is set, `simulate`
returns the handle of that already-computed run instead of invoking the Rust/PyO3
launcher. This keeps the documented 5-verb flow valid inside a fixed before-state
oracle container that ships no Rust toolchain. Idempotent; single source of truth
stays the canonical shim.
"""
import sys
from pathlib import Path

ANCHOR = '    run_name = run_name or f"api_predict_{time.strftime(\'%Y%m%d_%H%M%S\')}"'

BLOCK = '''    _prebaked = os.environ.get("VIBESIM_PREBAKED_RUN")
    if _prebaked:
        log_dir = repo / "logs" / _prebaked
        if not log_dir.is_dir():
            raise SystemExit(f"prebaked run not found: {log_dir}")
        meta = json.loads((log_dir / "prediction.meta.json").read_text())
        handle = (f"{repo.parents[0].name}:{_prebaked}"
                  if repo.name == "repo" else str(log_dir))
        return {"ok": True, "prediction_id": meta.get("prediction_id"),
                "prediction": handle, "run_name": _prebaked,
                "log_dir": str(log_dir), "meta": meta,
                "note": "container serves a fixed before-state prediction; "
                        "simulate is idempotent and never launches the "
                        "Rust/PyO3 runtime"}
'''

p = Path(sys.argv[1])
src = p.read_text()
if "VIBESIM_PREBAKED_RUN" in src:
    print("already patched:", p)
    sys.exit(0)
if ANCHOR not in src:
    sys.exit(f"anchor not found in {p}; canonical shim changed shape")
src = src.replace(ANCHOR, BLOCK + ANCHOR, 1)
p.write_text(src)
print("patched prebaked simulate into", p)
