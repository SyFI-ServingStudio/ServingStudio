#!/usr/bin/env bash
# The API host's half of deploy-public-api.sh, which pipes this file to
# `bash -s` over ssh. Run from the host user's home; directories are relative
# to it.
#
#   stage DIR SHA ORIGIN       clone (shallow) or fetch, then check out SHA
#   datasets DIR REPO@REV...   download each Hugging Face dataset revision the
#                              cache lacks, using the image DIR's lock builds
#   smoke DIR PORT SECONDS     replace the smoke container with one serving DIR
#                              on 127.0.0.1:PORT and wait for /health
#   smoke-stop                 remove the smoke container
#   promote STAGING LIVE BIND PORT SECONDS
#                              check LIVE out at STAGING's commit, copy its
#                              binaries, replace the live container and wait
#   status BIND PORT           containers and /health
set -euo pipefail

live_name=servingstudio-public-api
smoke_name=servingstudio-public-api-smoke
image=servingstudio-public-api
hf_home=${HF_HOME:-$HOME/.cache/huggingface}
binaries=(
  target/release/simulator
  target/release/analyze
  alignment/load_generator/req-frontend/target/release/tracegen
)

wait_health() {
  local url=$1 seconds=$2 waited=0
  until curl -fsS -m 5 -o /dev/null "$url/api/public/v1/health" 2>/dev/null; do
    if ((waited >= seconds)); then
      echo "no /health at $url after ${seconds}s" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "healthy at $url after ~${waited}s"
}

stage() {
  local dir=$1 sha=$2 origin=$3
  if [[ ! -d $dir/.git ]]; then
    git clone -q --depth 1 "$origin" "$dir"
  fi
  if [[ -n $(git -C "$dir" status --porcelain --untracked-files=no) ]]; then
    echo "$dir has local changes; resolve them by hand" >&2
    return 1
  fi
  git -C "$dir" fetch -q --depth 1 origin "$sha"
  git -C "$dir" checkout -q --detach "$sha"
  echo "$dir at $(git -C "$dir" rev-parse --short HEAD)"
}

build_image() {
  local dir=$1 context
  context=$(mktemp -d)
  cp "$dir/public_api/docker/Dockerfile" "$dir/pyproject.toml" "$dir/uv.lock" "$context/"
  docker build -q -t "$image" "$context" >/dev/null
  rm -rf "$context"
}

datasets() {
  local dir=$1 spec repo rev
  shift
  build_image "$dir"
  mkdir -p "$hf_home"
  for spec in "$@"; do
    repo=${spec%@*}
    rev=${spec#*@}
    if [[ -d $hf_home/hub/datasets--${repo//\//--}/snapshots/$rev ]]; then
      echo "dataset $repo@${rev:0:8} cached"
      continue
    fi
    echo "dataset $repo@${rev:0:8} downloading"
    # The container reads the hub offline, so it must be here before it starts.
    docker run --rm --user "$(id -u):$(id -g)" -v "$hf_home:/hf" -e HF_HOME=/hf "$image" \
      python -c "import sys; from huggingface_hub import snapshot_download as d; \
d(repo_id=sys.argv[1], repo_type='dataset', revision=sys.argv[2])" "$repo" "$rev" >/dev/null
  done
}

smoke() {
  local dir=$1 port=$2 seconds=$3
  docker rm -f "$smoke_name" >/dev/null 2>&1 || true
  if ss -ltn | grep -q "[:.]$port "; then
    echo "port $port is in use on $(hostname); set PUBLIC_API_SMOKE_PORT" >&2
    return 1
  fi
  PUBLIC_API_CONTAINER=$smoke_name bash "$dir/public_api/docker/run.sh" 127.0.0.1 "$port"
  if ! wait_health "http://127.0.0.1:$port" "$seconds"; then
    docker logs --tail 30 "$smoke_name" >&2
    return 1
  fi
}

promote() {
  local staging=$1 live=$2 bind=$3 port=$4 seconds=$5 sha origin path
  sha=$(git -C "$staging" rev-parse HEAD)
  origin=$(git -C "$staging" remote get-url origin)
  stage "$live" "$sha" "$origin"
  for path in "${binaries[@]}"; do
    mkdir -p "$(dirname "$live/$path")"
    # The live container may be running this binary (cp fails with "Text
    # file busy"); a rename leaves it on the old inode until it is replaced.
    cp -p "$staging/$path" "$live/$path.new"
    mv -f "$live/$path.new" "$live/$path"
  done
  bash "$live/public_api/docker/run.sh" "$bind" "$port"
  if ! wait_health "http://$bind:$port" "$seconds"; then
    docker logs --tail 30 "$live_name" >&2
    return 1
  fi
}

status() {
  local bind=$1 port=$2
  docker ps -a --filter "name=^servingstudio-public-api" \
    --format '{{.Names}}  {{.Status}}  {{.Ports}}'
  curl -fsS -m 5 "http://$bind:$port/api/public/v1/health" && echo
}

command=$1
shift
case $command in
  stage | datasets | smoke | promote | status) "$command" "$@" ;;
  smoke-stop) docker rm -f "$smoke_name" >/dev/null 2>&1 || true ;;
  *)
    echo "unknown command: $command" >&2
    exit 2
    ;;
esac
