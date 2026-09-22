# VibeSim analysis container -- fixed before-state diagnosis oracle for case vllm-28103.
#
# Serves the 5 agent-facing vibesim_api verbs (workspace-info, simulate, analyze,
# optimality, kernels) over HTTP + CLI, against a baked before-state prediction.
# No GPU, no Rust toolchain, no PyO3: only the prebuilt `analyze` binary + profile.db.
#
# Base MUST be ubuntu:24.04 -- the analyze binary requires GLIBC_2.39 (glibc on
# debian bookworm / python:slim is 2.36 and will fail to load it).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    VIBESIM_WS_ROOT=/opt/vibesim \
    VIBESIM_REPO=/opt/vibesim/repo \
    VIBESIM_WORKSPACE=. \
    VIBESIM_PREBAKED_RUN=before \
    VIBESIM_API_PORT=8799 \
    TMPDIR=/tmp \
    PYTHONUNBUFFERED=1

# python3 (3.12 on 24.04); shim + server are stdlib-only. python3-numpy goes to
# system dist-packages so the R6/R7 labeler (`model.work.floors`) finds it regardless
# of HOME/user-site. ca-certificates for downstream https; libgomp1 for the analyze
# binary's OpenMP.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-numpy ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vibesim

# The before-only workspace (source@defcc63, purged of after/answer keys) + the
# prebuilt analyze binary + profile.db + baked before prediction under logs/before.
COPY context/repo/        /opt/vibesim/repo/
# API surface: container shim (prebaked simulate) + HTTP server + entrypoint.
COPY context/app/         /opt/vibesim/app/
# Docs, worked example, real outputs, and the Codex skill -- for teaching the agent.
COPY context/vibesim-api/ /opt/vibesim/vibesim-api/
COPY context/skill/       /opt/vibesim/skill/

# The analyze binary spawns `uv run python -m model.work.floors` for the R6/R7 floor.
# Put the offline shim on PATH as `uv` so that resolves to `python3 -m ...` (numpy from
# system dist-packages), instead of real uv trying to sync a 9GB torch venv over network.
COPY context/app/uv /usr/local/bin/uv

RUN chmod +x /opt/vibesim/app/entrypoint.sh \
             /opt/vibesim/repo/target/release/analyze \
             /usr/local/bin/uv

EXPOSE 8799
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if b'\"ok\": true' in urllib.request.urlopen('http://127.0.0.1:8799/healthz').read().lower() else 1)" || exit 1

ENTRYPOINT ["/opt/vibesim/app/entrypoint.sh"]
CMD ["serve"]
