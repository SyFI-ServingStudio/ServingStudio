"""Device-free dependency and public API check for the pinned Neuron image."""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


expected = {
    "torch": "2.9.0+cpu",
    "torch-xla": "2.9.0",
    "torch-neuronx": "2.9.0.2.15.32035+de43f57c",
    "neuronx-cc": "2.27.5334.0+f702b353",
    "neuronx-distributed": "0.19.28492+435aae2b",
    "neuronx-distributed-inference": "0.9.0+4bcdc54b.dev",
    "libneuronxla": "2.2.17544.0+fb9962bf",
    "nki": "0.6.0+31049202112.g85070674",
    "nki-library": "0.0.2",
    "transformers": "4.57.6",
    "numpy": "2.2.6",
    "ml-dtypes": "0.5.4",
    "islpy": "2026.1",
}
output = Path(sys.argv[1])
if output.exists():
    raise FileExistsError(output)
report = {
    "scope": "Dependency/import validation; no compiler or model correctness claim",
    "python": sys.version,
    "libc": platform.libc_ver(),
    "devices": sorted(str(path) for path in Path("/dev").glob("neuron*")),
    "packages": {},
    "imports": {},
    "passed": False,
}
try:
    if os.environ.get("PJRT_DEVICE") != "CPU" or report["devices"]:
        raise RuntimeError("This check requires PJRT_DEVICE=CPU and no Neuron devices")
    if sys.version_info[:2] != (3, 12) or platform.libc_ver() != ("glibc", "2.39"):
        raise RuntimeError("Expected Python3.12/Ubuntu24 glibc2.39")
    for name, version in expected.items():
        actual = metadata.version(name)
        report["packages"][name] = {"expected": version, "actual": actual}
        if actual != version:
            raise RuntimeError(f"Unexpected package version: {name}")
    for name in (
        "torch", "torch_xla", "torch_neuronx", "nki", "neuronx_distributed",
        "neuronx_distributed_inference.models.llama.modeling_llama",
        "nkilib.core.qkv.qkv", "nkilib.core.mlp.mlp", "nrtpy",
    ):
        module = importlib.import_module(name)
        report["imports"][name] = getattr(module, "__file__", None)
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    report["pip_check"] = {
        "returncode": check.returncode, "stdout": check.stdout, "stderr": check.stderr
    }
    if check.returncode:
        raise RuntimeError("Pinned dependency metadata is inconsistent")
    report["passed"] = True
    print("Pinned Neuron image imports and dependencies passed.")
except BaseException as error:
    report["error"] = repr(error)
    raise
finally:
    output.write_text(json.dumps(report, indent=2) + "\n")
