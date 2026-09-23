#!/usr/bin/env bash
# JIT-fill the Kimi-K3 kernel rows a timing-predict preset needs, on the slurm-granted B200,
# into a BRANCH copy of profile.db (never the shared profiling/profile.db — merging that needs
# the user's explicit OK, per skills/impl-register-kernel). Runs in the clean worktree
# (main-k3-rust) because the user's main/ checkout cannot build Rust (rewound submodule).
#   sbatch slurm_gpu.sh ./run_k3_jit_fill.sh [preset ...]
# Default presets: the single-rank KDA and MLA layer presets (alignment) then the full TP8/EP8/PP2.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${VIBESIM_REPO:-/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust}"
DB="${VIBESIM_PROFILE_DB:-$HERE/kimi_single_layer/k3_branch_profile.db}"
[ -f "$DB" ] || cp "$REPO/profiling/profile.db" "$DB"
# Under slurm the cgroup shows only the granted GPU; when run directly on the host, pass
# K3_GPU_INDEX=3 (or 2) so the fill never lands on the first visible device (GPU 0).
if [ -n "${K3_GPU_INDEX:-}" ]; then
  case "$K3_GPU_INDEX" in 2|3) ;; *) echo "!! GPU $K3_GPU_INDEX not authorized (only 2,3)"; exit 1;; esac
  UUID="$(nvidia-smi -i "$K3_GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
else
  UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')"
fi
[ -n "$UUID" ] || { echo "!! no GPU visible"; exit 1; }
export VIBESIM_PROFILE_GPUS="$UUID" VIBESIM_PROFILE_DB="$DB"
export PATH="$HOME/.cargo/bin:$PATH"
export TMPDIR="/raid/tmp/yilegu_k3_tmp"; mkdir -p "$TMPDIR"
PRESETS=("$@")
[ ${#PRESETS[@]} -gt 0 ] || PRESETS=(presets/predict_kimi_k3_b200_rank1_layer_kda.json
                                     presets/predict_kimi_k3_b200_rank1_layer_mla.json
                                     presets/predict_kimi_k3_b200_sglang_tp8ep8pp2.json)
echo "==== K3 JIT fill: repo=$REPO db=$DB gpu=$UUID ===="
cd "$REPO"
for p in "${PRESETS[@]}"; do
  echo; echo "---- $p ----"
  timeout 5400 uv run --no-sync python -m launcher timing-predict "$p" 2>&1 | tail -40
  echo "RESULT $p exit=${PIPESTATUS[0]}"
done
echo; echo "==== rows per K3 table in $DB ===="
python3 - "$DB" <<'PY'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
for t in ("kda_recurrent_decode","kda_fused_decode","mla_decode_attention","mxfp4_fused_moe",
          "gdn_causal_conv_decode","gdn_gated_rms_norm","batched_gemm","single_gemm","rms_norm",
          "residual_rms_norm","mla_cache_append","elementwise","gemm_fp32_output"):
    try:
        n = con.execute(f'select count(*) from "{t}" where gpu_name like "%B200%"').fetchone()[0]
        print(f"{t}: {n} B200 rows")
    except Exception as e:
        print(f"{t}: {e}")
PY
