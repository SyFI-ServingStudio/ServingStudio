#!/usr/bin/env bash
# Keep Neuron's Torch/compiler stack separate from CUDA profiling dependencies.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$workspace_root/.env"
checkout=${1:-$workspace_root/ServingStudioSim}
uv_command=${UV_BIN:-uv}
if ! command -v "$uv_command" >/dev/null; then
    uv_command="$workspace_root/tmp/tools/uv"
fi
"$uv_command" venv --allow-existing --python python3.11 "$checkout/.venv-neuron"
"$uv_command" pip install --python "$checkout/.venv-neuron/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://pip.repos.neuron.amazonaws.com \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    'torch==2.9.0+cpu' 'neuronx-cc==2.27.5334.0+f702b353' \
    'nki==0.6.0+31049202112.g85070674' 'nki-library==0.0.2' \
    'numpy==1.26.4' 'ml-dtypes==0.5.4' pytest
"$checkout/.venv-neuron/bin/python" -c \
    'import importlib.metadata as m; print({p: m.version(p) for p in ("torch", "neuronx-cc", "nki", "nki-library")})'

# The native NKI path needs no Torch/XLA. The standalone production RMSNorm
# path uses a separate tracing stack compatible with AL2023's GLIBC 2.34.
# islpy 2026 changes the compiler's is_subset binding and breaks HLO lowering.
"$uv_command" venv --allow-existing --python python3.11 "$checkout/.venv-neuron-trace"
"$uv_command" pip install --python "$checkout/.venv-neuron-trace/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://pip.repos.neuron.amazonaws.com \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    'torch==2.6.0+cpu' 'torch-neuronx==2.6.0.2.10.16998+e9bf8a50' \
    'torch-xla==2.6.1' 'libneuronxla==2.2.17544.0+fb9962bf' \
    'neuronx-cc==2.27.5334.0+f702b353' \
    'nki==0.6.0+31049202112.g85070674' 'nki-library==0.0.2' \
    'numpy==1.26.4' 'ml-dtypes==0.5.4' pytest
# The compiler metadata requests islpy 2026. Apply the independently validated
# HLO compatibility override separately so the resolver can install the SDK.
"$uv_command" pip install --python "$checkout/.venv-neuron-trace/bin/python" \
    --no-deps 'islpy==2024.2'
PATH="$checkout/.venv-neuron-trace/bin:$PATH" \
    "$checkout/.venv-neuron-trace/bin/python" -c \
    'import torch_neuronx; import importlib.metadata as m; print(m.version("torch-neuronx"))'

# Pinned production source for the experimentally validated dense compiler
# boundary. NxDI's package metadata targets Torch 2.9 (GLIBC >=2.35), while
# this AL2023 host uses the separately verified Torch 2.6 tracing stack.
"$uv_command" pip install --python "$checkout/.venv-neuron-trace/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    'transformers==4.57.6' 'torchvision==0.21.0+cpu' \
    sentencepiece pillow blobfile 'hf-xet>=1.1.10,<2' \
    accelerate cloudpickle einops opt-einsum tenacity
"$uv_command" pip install --python "$checkout/.venv-neuron-trace/bin/python" \
    --no-deps --extra-index-url https://pip.repos.neuron.amazonaws.com \
    'neuronx-distributed==0.19.28492+435aae2b' \
    'neuronx-distributed-inference @ git+https://github.com/aws-neuron/neuronx-distributed-inference.git@4bcdc54bf3b6ccdad490e5cd0680a3dc95270b8e'
