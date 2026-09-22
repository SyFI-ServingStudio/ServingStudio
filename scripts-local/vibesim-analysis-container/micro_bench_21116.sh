#!/usr/bin/env bash
# Characterize the FUSION half of vllm-pr-21116 (fused MLA QKV a_proj) WITHOUT the 671B DeepSeek-V3
# weights and WITHOUT a from-source vLLM build. The PR's deepseek_v2.py change replaces two
# ReplicatedLinears -- q_a_proj (hidden->q_lora_rank) and kv_a_proj_with_mqa
# (hidden->kv_lora_rank+qk_rope_head_dim) -- with ONE MergedReplicatedLinear
# (hidden->q_lora_rank+kv_lora_rank+qk_rope_head_dim). Both are RANK-LOCAL replicated linears, so
# the isolated op is identical on 1 GPU as on the PR's TP8. We time BEFORE (2 GEMMs) vs AFTER
# (1 fused GEMM) at DeepSeek-V3's real dims across representative token counts, and report the
# gap + a 3-sigma floor -- the same resolvability question the per-case floor answers, applied to
# a faithful synthetic of the fusion (real widths, random weights; weights don't affect latency).
# This does NOT include the PR's separate strided-layernorm CUDA-kernel half.
#
# Usage (under slurm):  sbatch slurm_gpu.sh ./micro_bench_21116.sh
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU="${DOCKER_GPU_ARG:-\"device=3\"}"
IMG="${IMG:-rga-local/sglang-pr-20469:b200-after}"   # any image with torch+cuda; reuse a local one
OUT="$HERE/micro_21116"; mkdir -p "$OUT"
CNAME="micro21116_$(date +%s%N)"

cat > "$OUT/bench.py" <<'PY'
import torch, statistics as st, json
torch.manual_seed(0)
dev="cuda"; dt=torch.bfloat16
H=7168; QL=1536; KV=512; ROPE=64          # DeepSeek-V3 real dims
KVR=KV+ROPE                                # kv_a_proj_with_mqa out = 576
FUS=QL+KVR                                 # fused out = 2112
# rank-local ReplicatedLinear == plain nn.Linear (bias=False, like vLLM ReplicatedLinear default)
q_a  = torch.nn.Linear(H, QL,  bias=False).to(dev,dt)
kv_a = torch.nn.Linear(H, KVR, bias=False).to(dev,dt)
fused= torch.nn.Linear(H, FUS, bias=False).to(dev,dt)
# make fused == concat so outputs match (correctness sanity)
with torch.no_grad():
    fused.weight[:QL].copy_(q_a.weight); fused.weight[QL:].copy_(kv_a.weight)

def timed(fn, iters, reps):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    out=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e)/iters*1000)   # us/call
    return out

def before(x): return torch.cat([q_a(x), kv_a(x)], dim=-1)
def after(x):  return fused(x)

report={"dims":{"H":H,"q_lora_rank":QL,"kv_a":KVR,"fused":FUS},"rows":[]}
for T in [219, 512, 1024, 4096, 16384]:
    x=torch.randn(T,H,device=dev,dtype=dt)
    # correctness
    err=(before(x)-after(x)).abs().max().item()
    b=timed(lambda:before(x), 3000, 5)
    a=timed(lambda:after(x),  3000, 5)
    bm,bs=st.mean(b),st.pstdev(b); am,ap=st.mean(a),st.pstdev(a)
    gap=bm-am; noise=(bs**2+ap**2)**0.5 or 1e-9; floor=3*noise
    row={"T":T,"before_us":round(bm,3),"before_std":round(bs,3),
         "after_us":round(am,3),"after_std":round(ap,3),"gap_us":round(gap,3),
         "noise_us":round(noise,4),"floor3s_us":round(floor,3),
         "resolvable":bool(gap>floor),"speedup":round(bm/am,3),"max_abs_err":round(err,5)}
    report["rows"].append(row)
    print(f"T={T:6d}  before {bm:7.3f}+-{bs:.3f}  after {am:7.3f}+-{ap:.3f}  "
          f"gap {gap:7.3f}  floor3s {floor:6.3f}  resolvable={gap>floor}  x{bm/am:.2f}  err={err:.4g}")
json.dump(report, open("/out/micro_21116_report.json","w"), indent=2)
print("JSON_DONE")
PY

echo "==== 21116 fusion micro-bench (DeepSeek-V3 dims, img=$IMG gpu=$GPU) ===="
docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run --rm --name "$CNAME" --gpus "$GPU" -e CUDA_VISIBLE_DEVICES=0 \
  -v "$OUT":/out --workdir /out --entrypoint python3 "$IMG" /out/bench.py 2>&1 \
  | grep -vE "Warning|warn|deprecated|Wrapper" | tee "$OUT/bench.log"
echo "==== done -> $OUT/micro_21116_report.json ===="
