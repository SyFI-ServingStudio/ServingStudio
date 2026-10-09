#!/usr/bin/env bash
# Full-model functional proof using the isolated AL2023 Neuron tracing stack.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$workspace_root/.env"
if [[ $# -ne 4 ]]; then
    echo "usage: $0 compile|validate MODEL_DIR COMPILED_DIR REPORT_DIR" >&2
    exit 2
fi
phase=$1
case "$phase" in
    compile|validate) ;;
    *) echo "phase must be compile or validate" >&2; exit 2 ;;
esac
checkout="$workspace_root/ServingStudioSim"
python_command=${SERVINGSTUDIO_NEURON_TRACE_PYTHON:-$checkout/.venv-neuron-trace/bin/python}
export PATH="$(dirname "$python_command"):$PATH"
export PYTHONPATH="$checkout"
export OMP_NUM_THREADS=4
export NXD_CPU_MODE=0
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_COMPILE_CACHE_URL="$TMPDIR/neuron-compile-cache"
compiled_dir=$(realpath -m "$3")
export BASE_COMPILE_WORK_DIR="${compiled_dir}.work"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
if [[ "$phase" == compile ]]; then
    export PJRT_DEVICE=CPU
else
    # Physical allocation and the shared chip lock are owned by the Python
    # helper, which rejects inherited visibility restrictions before loading.
    export PJRT_DEVICE=NEURON
fi
exec "$python_command" "$checkout/tools/trainium2/validate_checkpoint.py" \
    "$phase" --model-dir "$2" --compiled-dir "$3" --report-dir "$4"
