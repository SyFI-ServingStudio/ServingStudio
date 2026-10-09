#!/usr/bin/env bash
# Offline image reuse; model validation remains a separate operation.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$workspace_root/.env"
mode=${1:?Usage: neuron-image.sh build|check INPUT OUTPUT_DIR}
input=${2:?Expected context directory or immutable image ID}
output_dir=$(realpath -m "${3:?Expected a new output directory}")
socket_uri=${SERVINGSTUDIO_NEURON_DOCKER_HOST:?Set SERVINGSTUDIO_NEURON_DOCKER_HOST to the intended Docker Unix socket}
[[ "$socket_uri" == unix:///* ]] || { echo 'Expected an absolute Unix socket URI.' >&2; exit 2; }
docker_command=(sudo -n docker --host "$socket_uri")
case "$mode" in build|check) ;; *) echo 'Mode must be build or check.' >&2; exit 2 ;; esac
[[ ! -e "$output_dir" ]] || { echo "Preserve existing output: $output_dir" >&2; exit 1; }
"${docker_command[@]}" info --format '{{.DockerRootDir}}'
mkdir -p "$output_dir"
exec > >(tee "$output_dir/job.log") 2>&1
ulimit -c 0
if [[ "$mode" == build ]]; then
    context_dir=$(realpath "$input")
    python3 "$workspace_root/scripts/neuron-image/verify-context.py" "$context_dir"
    base_image=public.ecr.aws/neuron/pytorch-inference-neuronx@sha256:de98dc02a68765df5688d301134debbd677ccdc4fbd6873af34fafafcc3462ec
    "${docker_command[@]}" image inspect "$base_image" >/dev/null
    # This build uses the checked-in recipe and never pulls or publishes.
    sudo -n env DOCKER_BUILDKIT=0 docker --host "$socket_uri" build \
        --network=none --pull=false --iidfile "$output_dir/image-id.txt" \
        --file "$workspace_root/scripts/neuron-image/Dockerfile" "$context_dir"
    image_id=$(cat "$output_dir/image-id.txt")
else
    [[ "$input" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'Check requires an immutable image ID.' >&2; exit 2; }
    image_id=$input
    printf '%s\n' "$image_id" >"$output_dir/image-id.txt"
fi
"${docker_command[@]}" image inspect "$image_id" >"$output_dir/image.json"
mkdir -p "$output_dir/tmp" "$output_dir/home"
"${docker_command[@]}" run --rm --pull=never --network=none --no-healthcheck \
    --ulimit core=0 --security-opt=no-new-privileges --user "$(id -u):$(id -g)" \
    --env PJRT_DEVICE=CPU --env PYTHONDONTWRITEBYTECODE=1 \
    --env HOME=/work/home --env TMPDIR=/work/tmp \
    --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,source=$workspace_root/scripts/neuron-image/check.py,target=/check.py,readonly" \
    --mount "type=bind,source=$output_dir,target=/work" \
    --entrypoint=/opt/trainium-venv/bin/python "$image_id" /check.py /work/result.json
