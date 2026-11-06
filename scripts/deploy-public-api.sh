#!/usr/bin/env bash
# Deploys ServingStudio Sim's public API, the service the CSE site proxies
# /api/public/v1 to, from the root ServingStudioSim checkout's HEAD.
#
#   scripts/deploy-public-api.sh deploy [--no-promote]
#       Build the release binaries here; on the host, stage the commit in a
#       separate clone and smoke-test it in a loopback container; then check
#       the live clone out at it, replace the live container, and check the
#       site. --no-promote stops after the smoke test. A failed smoke test
#       leaves the live clone and container untouched.
#   scripts/deploy-public-api.sh check [--quick]
#       Check the live service through the public site, from the host.
#   scripts/deploy-public-api.sh status
#       The host's containers and the live /health.
#
# The host is internal and this repository is public, so it is configured by
# environment variables (export them in your shell profile):
#   PUBLIC_API_HOST         ssh host that runs the container (required)
#   PUBLIC_API_BIND         address the container binds: the one the site's
#                           .htaccess proxies to (required)
#   PUBLIC_API_PORT         default 5220
#   PUBLIC_API_SMOKE_PORT   loopback port of the smoke container, default 5228
#   PUBLIC_API_REMOTE_DIR   directory under the host user's home (or an
#                           absolute path) holding the live and staging
#                           clones, default ServingStudio
#   PUBLIC_API_HF_HOME      Hugging Face home on the host for the pinned
#                           datasets, default the host's $HF_HOME or
#                           ~/.cache/huggingface
#   PUBLIC_SITE_URL         default https://servingstudio.cs.washington.edu
#
# The host needs ssh, git, docker and python3; the binaries are built here and
# copied, and run inside the image (Ubuntu 24.04), so the host's own toolchain
# and glibc do not matter.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd -P)
sim=$root/ServingStudioSim
remote_script=$root/scripts/public-api-remote.sh
check_script=$root/scripts/public-api-check.py

host=${PUBLIC_API_HOST:?set PUBLIC_API_HOST to the ssh host that runs the public API}
bind=${PUBLIC_API_BIND:?set PUBLIC_API_BIND to the address the site proxies to}
port=${PUBLIC_API_PORT:-5220}
smoke_port=${PUBLIC_API_SMOKE_PORT:-5228}
remote_dir=${PUBLIC_API_REMOTE_DIR:-ServingStudio}
hf_home=${PUBLIC_API_HF_HOME:-}
site=${PUBLIC_SITE_URL:-https://servingstudio.cs.washington.edu}
live=$remote_dir/ServingStudioSim
staging=$remote_dir/ServingStudioSim-staging
# The service takes about 100 s to answer after its container starts.
start_seconds=300

on_host() {
  ssh -o BatchMode=yes "$host" ${hf_home:+env HF_HOME="$hf_home"} bash -s -- "$@" <"$remote_script"
}
check_on_host() { ssh -o BatchMode=yes "$host" python3 - "$@" <"$check_script"; }
step() { printf '\n== %s\n' "$*"; }

deploy() {
  local promote=1
  [[ ${1:-} == --no-promote ]] && promote=0

  step "check the root ServingStudioSim checkout"
  if [[ -n $(git -C "$sim" status --porcelain --untracked-files=no) ]]; then
    echo "$sim has uncommitted changes; the deploy serves committed files only" >&2
    exit 1
  fi
  if git -C "$sim" submodule status alignment/load_generator/req-frontend | grep -q '^[-+U]'; then
    echo "req-frontend is not at the recorded commit; run git submodule update" >&2
    exit 1
  fi
  local sha origin
  sha=$(git -C "$sim" rev-parse HEAD)
  origin=$(git -C "$sim" remote get-url origin)
  git -C "$sim" fetch -q origin
  if [[ -z $(git -C "$sim" branch -r --contains "$sha") ]]; then
    echo "$sha is not on origin; push it first (the host fetches it from there)" >&2
    exit 1
  fi
  echo "deploying $(git -C "$sim" log -1 --format='%h %s' "$sha")"

  step "build the release binaries"
  (
    cd "$sim"
    uv run --frozen cargo build --locked --release -p simulator -p analyzer
    cargo build --locked --release \
      --manifest-path alignment/load_generator/req-frontend/Cargo.toml --bin tracegen
  )

  step "stage $sha on $host:$staging"
  on_host stage "$staging" "$sha" "$origin"
  local path
  for path in target/release/simulator target/release/analyze \
    alignment/load_generator/req-frontend/target/release/tracegen; do
    rsync -t --mkpath "$sim/$path" "$host:$staging/$path"
  done
  # Every Hugging Face dataset revision the presets pin (their captures);
  # tests and docs name made-up ones, so only presets/ is read.
  mapfile -t specs < <(git -C "$sim" grep -ohE 'hf://datasets/[^@" ]+@[0-9a-f]{40}' "$sha" -- presets \
    | sed -E 's|^.*hf://datasets/||' | sort -u)
  on_host datasets "$staging" "${specs[@]}"

  step "smoke-test on $host 127.0.0.1:$smoke_port"
  on_host smoke "$staging" "$smoke_port" "$start_seconds" || { on_host smoke-stop; exit 1; }
  if ! check_on_host "http://127.0.0.1:$smoke_port" --commit "$sha"; then
    on_host smoke-stop
    echo "smoke test failed; the live service is untouched" >&2
    exit 1
  fi
  on_host smoke-stop

  if ((!promote)); then
    echo "smoke test passed; --no-promote leaves the live service as it was"
    return
  fi

  step "replace the live container on $host ($bind:$port)"
  on_host promote "$staging" "$live" "$bind" "$port" "$start_seconds"

  step "check $site"
  check_on_host "$site" --commit "$sha"
}

case ${1:-} in
  deploy) shift; deploy "$@" ;;
  check) shift; check_on_host "$site" "$@" ;;
  status) on_host status "$bind" "$port" ;;
  *)
    echo "usage: $0 deploy [--no-promote] | check [--quick] | status" >&2
    exit 2
    ;;
esac
