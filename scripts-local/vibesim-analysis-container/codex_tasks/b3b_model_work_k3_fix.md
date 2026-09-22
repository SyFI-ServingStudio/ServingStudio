# Task: two corrections to the Kimi-K3 `model.work` label (VibeSim repo, branch `kimi-k3`)

You are in the VibeSim repository on git branch `kimi-k3` (do not switch). CPU only
(`CUDA_VISIBLE_DEVICES` empty), `uv run ...`. Commit `9ab54685` added `model/work/models/kimi_k3.py`,
`model/work/attention/mla.py`, the KDA spec in `model/work/attention/linear.py`, MoE/quantization
changes and goldens in `tests/test_model_work.py`. Another agent is concurrently editing
`profiling/` and `simulator/` on this branch: touch ONLY `model/work/**` and `tests/test_model_work.py`,
and `git add` only those paths (never `git add -A`). Follow `skills/impl-add-model-work-label/SKILL.md`
conventions; comments in English; commit message ends with
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Correction 1 — KDA gate ranks (verified in the sglang v0.5.20 source, do not re-derive from memory)
`KimiK3DeltaAttention.__init__` with `use_full_rank_gate=True`
(`/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/k3_src_v0520/models_kimi_k3.py`
lines 1447-1500): `fused_qkvg_proj = MergedColumnParallelLinear(hidden, [P, P, P, P])` with
P = num_heads*head_dim = 12288 — i.e. q, k, v AND the output gate **g** are full-rank 7168 -> 12288
each; `b_proj` 7168 -> num_heads (96); the forget gate **f** stays LOW-RANK: `f_a_proj` 7168 -> 128
(head_dim) then `f_b_proj` 128 -> 12288. The previous task's prompt wrongly said both f and g are
full-rank. Fix the KDA spec accordingly (per-layer KDA projection params = 4*7168*12288 +
7168*96 + 7168*128 + 128*12288 + o_proj 12288*7168), keep semantic names stable where possible
(a new `kda.f_a`/`kda.f_b` split is fine — document it), and update every golden that changes
(show the arithmetic in comments). Re-check the total parameter count against the official ~2.8T.

## Correction 2 — attention-residual stream (production path, currently missing)
`text_config.attn_res_block_size = 12` (real HF config) turns on K3's attention-residual stream
(`KimiK3DecoderLayer.__init__` lines 2429-2445 and `_forward_attn_residual` 2640-2760;
`sglang/srt/layers/attn_residual.py` — readable via
`docker run --rm --entrypoint bash lmsysorg/sglang:v0.5.20 -c 'cat /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py'`).
Per decoder layer this adds: two score projections `self_attention_res_proj` and `mlp_res_proj`
(each hidden 7168 -> 1, bf16), two extra RMSNorms (`self_attention_res_norm`, `mlp_res_norm`,
7168), and two aggregations per layer over the snapshot bank: a layer at index i (0-based) reads
`nvb = ceil(i/12)` banked rows of [T, 7168] bf16 (bank of `ceil(93/12) = 8` rows per token), scores
them (dot with the [7168] score vector after the score norm), softmax-weights and sums them with
the current prefix; write-layers (i % 12 == 0) additionally snapshot one [T, 7168] row into the
bank. Model this as necessary work (bytes = nvb*T*7168*2 read per aggregation, plus the row write
on write layers; flops = 2*nvb*T*7168 per aggregation for the score dot + the weighted sum) with
semantic names under the layer tags (e.g. `kda.attn_res.agg1`, `kda.attn_res.agg2`,
`mla.attn_res.*`, `dense.attn_res.*`) so a location map can bind them, plus the per-layer
`attn_res` weight bytes. Also the model-level final aggregation (`output_attn_res_proj`/`_norm`).
Add goldens for one KDA layer at B=128, L=8192 with nvb = 1 (layer 5) and the model total.

## Deliverable
Commit on `kimi-k3`; `uv run pytest tests/test_model_work.py -q` green; re-run
`uv run python -m model.work model/config/kimi_k3.json --decode 128x8192 --gpu B200 --json` and
report the new total params and decode ms next to the old (2,785,387,508,480 / 195.814 ms).
