#!/usr/bin/env bash
# GPU smoke of the Kimi-K3 profiling kinds/backends added on VibeSim branch kimi-k3 (B2):
# one representative K3 shape per kind, measured into a PRIVATE sqlite (never the shared
# profiling/profile.db). Run under slurm so the launcher's docker workers bind the granted
# GPU by UUID (VIBESIM_PROFILE_GPUS accepts UUIDs on this branch):
#   sbatch slurm_gpu.sh ./run_k3_kernel_smoke.sh [DB=kimi_single_layer/k3_smoke.db]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${VIBESIM_REPO:-/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main}"
DB="${1:-$HERE/kimi_single_layer/k3_smoke.db}"
UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$UUID" ] || { echo "!! no GPU visible"; exit 1; }
export VIBESIM_PROFILE_GPUS="$UUID"
export PATH="$HOME/.cargo/bin:$PATH"
# the host (torch) backend builds its CUPTI extension under $TMPDIR/moesim_cupti_ext;
# the shared /raid/tmp one belongs to another user -> use our own.
export TMPDIR="/raid/tmp/yilegu_k3_tmp"; mkdir -p "$TMPDIR"
echo "==== K3 kernel smoke: repo=$REPO db=$DB gpu=$UUID ===="
cd "$REPO"
run () {  # run <kind> <backend> <spec-json>
  local kind="$1" be="$2" spec="$3"
  echo; echo "---- $kind / $be ----"; echo "spec: $spec"
  if timeout 1500 uv run --no-sync python -m launcher kernel-profile run "$kind" --backend "$be" \
       --spec "$spec" --db "$DB" --gpu-name "NVIDIA B200" 2>&1 | tail -25; then
    echo "RESULT $kind/$be OK"
  else
    echo "RESULT $kind/$be FAIL"
  fi
}
KDA='{"batch_size":32,"num_heads":12,"head_k_dim":128,"head_v_dim":128,"dtype":"bf16","state_dtype":"fp32","lower_bound":-5.0}'
# fused KDA decode is only covered for an fp32 recurrent state (kda_fused_decode.covered)
KDA_BF16='{"batch_size":128,"num_heads":12,"head_k_dim":128,"head_v_dim":128,"dtype":"bf16","state_dtype":"fp32","lower_bound":-5.0}'
KDA_STATE_BF16='{"batch_size":128,"num_heads":12,"head_k_dim":128,"head_v_dim":128,"dtype":"bf16","state_dtype":"bf16","lower_bound":-5.0}'
MLA='{"num_heads":12,"kv_lora_rank":512,"rope_dim":64,"q_dtype":"bf16","kv_dtype":"bf16","page_size":64,"batch_size":128,"kv_len":8192}'
MLA_FP8='{"num_heads":12,"kv_lora_rank":512,"rope_dim":64,"q_dtype":"bf16","kv_dtype":"fp8_e4m3","page_size":64,"batch_size":128,"kv_len":8192}'
MOE="$(python3 -c 'import json; print(json.dumps({"num_tokens":128,"hidden_size":3584,"intermediate_size":3072,"num_experts":896,"num_local_experts":112,"top_k":16,"input_dtype":"bf16","weight_format":"mxfp4_e2m1_ue8m0","group_size":32,"routing_method":"deepseek_v3_sigmoid","activation":"situ","n_group":1,"topk_group":1,"routed_scaling_factor":1.0,"gemm1_alpha":4.0,"gemm1_clamp_limit":25.0,"per_expert_batches":[128]*16+[0]*880},separators=(",",":")))')"
run kda_recurrent_decode torch          "$KDA"
run kda_recurrent_decode sglang_triton  "$KDA"
run kda_fused_decode     sglang_fused   "$KDA_BF16"
run kda_recurrent_decode sglang_triton  "$KDA_STATE_BF16"
run mla_decode_attention sglang_cutedsl_mla "$MLA_FP8"
run mla_decode_attention sglang_trtllm_mla  "$MLA"
run mla_decode_attention sglang_triton      "$MLA"
run mxfp4_fused_moe      sglang_trtllm_mxfp4 "$MOE"
run gdn_causal_conv_decode sglang_triton '{"batch_size":32,"channels":4608,"kernel_size":4,"dtype":"bf16","state_dtype":"bf16"}'
run gdn_gated_rms_norm     sglang_triton '{"m":32,"hidden":128,"dtype":"bf16"}'
run batched_gemm           sglang_k3_absorb '{"num_batches":12,"m":1,"n":512,"k":128,"dtype":"bf16"}'
echo; echo "==== summary ===="; grep -h "^RESULT" /dev/null "$0" >/dev/null 2>&1; true
python3 - "$DB" <<'PY'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
for (t,) in con.execute("select name from sqlite_master where type='table' and name not like '\\_%' escape '\\'"):
    n = con.execute(f'select count(*) from "{t}"').fetchone()[0]
    if n: print(f"{t}: {n} rows")
PY
