# Real-Repro Iterative-Optimizer Eval Harness

Last updated: 2026-09-21. Location: `scripts-local/vibesim-analysis-container/` (untracked, not in git).

## What it tests

Can an optimizer agent, given only a model + workload + a metric to improve — **no issue reference,
no operator name, no target number** — derive the real bottleneck from VibeSim's roofline analysis,
map it to actual source, edit + recompile a real kernel, and make the layer faster *without changing
its numerical output*?

This is the "real-repro" harness: it runs **actual vLLM/sglang** built at the PR's before/after
commits, not a cost-model simulation. It supersedes two earlier designs:
- `scripts-local/eval-pipeline/` — the older k-trial loop against VibeSim's **simulator** (no real
  GPU kernels). Retired because the prompt named the operator (answer leak).
- `scripts-local/vibesim-api/vibesim_api.py` — the 6-verb optimizer-agent API layer.

## The loop (per trial)

Driver: **`run_iter_opt_eval.sh [CONFIG=issue_28103.json] [N_TRIALS=3] [EFFORT=max]`**
(env: `START=<k>` resume, `AGENT_TIMEOUT=<s>` per-turn budget, default 3600).

1. **Seed** — fresh codex container from the case's `codex_image` (before-state vLLM + Codex CLI),
   workdir = the source checkout (`/workspace/vllm`). Extractor copied to `/tmp`.
2. **Agent** — `codex exec -m gpt-5.6-luna -c model_reasoning_effort=max
   --dangerously-bypass-approvals-and-sandbox`, fed the **de-leaked** rendered prompt
   (`agent_task_iter_opt.md` → `agent_task_rendered.md`). The agent profiles the current code,
   builds a VibeSim timing-prediction from that profile, runs `optimality`/`analyze` to find the
   bottleneck, edits the kernel, and organizes work under `/workspace/opt_run/iter_NN/`.
3. **Deterministic rebuild** — the *driver* (not the agent) runs the case `REBUILD_CMD`. Build
   failure ⇒ FAIL ("agent left source non-compiling"). The command ends with a **full-engine
   smoke test** (`LLM(...).generate` from `/tmp`, prints `ENGINE_SMOKE_OK`) so the agent fails on
   the exact engine-init crash the judge would otherwise catch later.
4. **Install** — `cp -a <checkout>/<framework>/. /opt/venv/lib/python3.12/site-packages/<framework>/`
   so the built `_C.so` + `.py` land in the wheel Python actually imports.
5. **Judge** — `judge_general.py` (below). Verdict JSON at `iter_opt_eval_<case>/trial_<k>_verdict.json`.

Artifacts saved per trial: `trial_<k>_agent.log` (transcript), `trial_<k>_opt_run/` (iteration
workspace), `trial_<k>_build.log`, `trial_<k>_verdict.json`.

## The judge — `judge_general.py`

Model-agnostic. Holds the **private answer key** (`targets` in the case JSON); the agent never sees
it. Flow:
1. **Captures its own golden** on a fresh *pristine before* container (cached under `--golden-dir`,
   keyed by model+target) — the agent cannot tamper with the reference.
2. **Replays** that golden's captured input on the agent's rebuilt module via
   `extract_and_profile.py --replay`.
3. **Two gates** (both must pass):
   - `correctness` — replayed output still matches the before-golden (`max_abs_err < err_max`,
     no NaN). Stops an agent from deleting the copy and returning garbage.
   - `latency_recovered` — `recovered_frac = (before_us − measured)/(before_us − after_us)`, on the
     **primary** target (`targets[0]`). PASS iff `recovered_frac ≥ recover_pass` (0.70).
   - Missing latency on the primary target ⇒ FAIL ("extractor produced no latency on primary target").

## The extractor — `extract_and_profile.py`

Model-agnostic capture-replay, three target kinds (`detect_kind`: `::`/`torch.ops` → operator;
`:` → function/micro-op; else module):
- **capture** — hooks the target during a real forward, snapshots its actual inputs (via `_pack`:
  full underlying storage flat + shape/stride/offset, so **strided/non-contiguous** views survive
  save/load — the naive `.cpu()` contiguizes and erases the copy-trigger).
- **replay** — rebuilds inputs with `torch.as_strided`, times the current impl under CUDA events,
  diffs output vs the golden.
- vLLM V1 runs the model in a subprocess, so capture/time/replay execute **inside the worker** via
  `llm.collective_rpc(fn)`. The sglang equivalent is `sglang_profile_patch.py` (Scheduler-attached
  `ep_install`/`ep_time`/`ep_replay`, results-via-file since RPC only returns `(ok, str)`).

## Anti-cheat / integrity (hard-won)

1. **Private answer key** — target module lives only in the case JSON, never in the prompt.
2. **Driver owns build+install+judge** — agent self-measurement is unreliable (`_C.so`
   rebuild/import staleness once made a real 6.99µs look like ~4%).
3. **Config-swap defeated** — the deterministic rebuild recompiles from source; you can't win by
   swapping a config to the after-arch.
4. **Engine smoke in REBUILD_CMD** — agent and judge run the identical command, so a kernel that
   passes isolated timing but breaks CUDA-graph capture fails in the agent's own loop.
5. **De-leaked prompt** — "optimize metric X" only; no necessary/unnecessary/removable-work
   language. Leak scan = 0 answer terms. Trajectories audited for web-search leaks.

## Per-case config (example: `issue_28103.json`)

```
targets       : ["model.layers.0.self_attn.q_norm", "model.layers.0.self_attn.k_norm"]  (PRIVATE)
before/after/codex_image : the three built images
before_us / after_us     : 14.4 / 8.1  (extractor replay-mode baselines, B200)
recover_pass  : 0.70     err_max : 0.05     tokens : 8     iters : 3000
render        : METRIC, FRAMEWORK, CHECKOUT, MODEL_NAME, WORKLOAD, TOKENS, REBUILD_CMD
```

The `render.*` fields fill the prompt template; `REBUILD_CMD` carries the build+install+smoke chain.

## Per-case build + viability

- **Images**: `build_repro_image.sh <case> <variant> <commit> <tag>` (B200 SM10.0, ~15 min) and
  the per-case `build_case_*.sh` / `Dockerfile.*` derive before/after by swapping the changed source
  and (for CUDA) recompiling.
- **Viability gate**: `measure_floor.sh <config>` captures a golden on before, replays REPS× on
  before and after, reports `gap` and `floor = 3σ pooled noise`. A case is a usable eval iff
  `gap > floor`.

## Case status (2026-09-21)

| Case | Model | Change type | Rebuild? | Verdict |
|---|---|---|---|---|
| **28103** | vllm Qwen3-0.6B | `.cu` layernorm + `.py` | yes (build_ext) | **RESOLVABLE**, PASS 2/3 |
| **20469** | sglang Qwen3.5-0.8B | pure `.py` causal_conv1d | no (file swap) | **RESOLVABLE** (gap 234µs, 4.09×) |
| 27931 | vllm Qwen3-0.6B | `.cu` rms_norm vec8 | yes | NULL (gap < floor) |
| 32296 | sglang Qwen3-4B-FP8 | `.cuh` JIT fp8 quant | JIT recompile | NULL (gap < floor) |
| 21116 | vllm DeepSeek-V3 | `.cu` + fused proj | yes | MODEL-BLOCKED (needs q_lora≠null model) |
| 29690 / 30947 | Nemotron / GLM TP4/TP8 | — | — | RESOURCE-BLOCKED (>2 GPU) |

**Rebuild rule of thumb:** `.cu`/`.cpp` (AOT) ⇒ must recompile + reinstall the `.so`; `.cuh` JIT ⇒
JIT recompiles on a content-hashed cache miss (clear cache so before ≠ after); pure `.py` ⇒ swap file.

## GPU / infra

- All GPU work goes through **`sbatch slurm_gpu.sh <cmd>`** (partition `main`, 2h limit; binds
  docker to the cgroup-assigned GPU by UUID via `DOCKER_GPU_ARG`). Node: cayenne, 8× B200.
- Codex model pinned to **gpt-5.6-luna @ max**; `CODEX_HOME` mounted from `/raid/yilegu/codex_home_eval`.
- flashinfer cubins pre-warmed at `/raid/yilegu/flashinfer_cache` (offline runs). Goldens cached at
  `/raid/yilegu/eval_goldens`.

## Related: Kimi-K3 single-layer profiler (same dir, separate tool)

`kimi_single_layer_decode.py` + `run_kimi_single_layer.sh` — profiles ONE K3 decoder layer on one
GPU (the full ~1T MoE won't fit). Not part of the eval loop; it's the roofline-target discovery
tool. Builds a `KimiK3DecoderLayer` with random bf16 weights on `lmsysorg/sglang:v0.5.20`, EP=16 rank
(56 routed experts), synthetic decode inputs + KDA recurrent-state. See
`memory/kimi-single-layer-decode-harness.md` for first numbers (KDA layer is context-length-flat;
MoE is the tunable ~50%).
