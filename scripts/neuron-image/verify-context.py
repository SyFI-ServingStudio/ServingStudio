"""Validate the frozen local build inputs without importing the SDK."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


root = Path(sys.argv[1]).resolve()
recipe = Path(__file__).resolve().parent
# The recorded context is exact: wheel metadata, source revision, transitive
# dependency lock and runtime archives must match the reviewed experiment.
expected = json.loads((recipe / "artifacts.json").read_text())
if json.loads((root / "artifacts.json").read_text()) != expected:
    raise ValueError("Build context differs from the frozen artifact ledger")
for item in expected:
    path = root / item["path"]
    if not path.is_file() or digest(path) != item["sha256"]:
        raise ValueError(f"Frozen build artifact changed: {path}")
print(f"Verified {len(expected)} frozen artifacts; no SDK or model execution.")
