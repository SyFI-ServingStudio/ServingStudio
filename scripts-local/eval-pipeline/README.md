# VibeSim optimization eval pipeline

Goal: for each reproduced issue (a real framework PR with a known bottleneck +
fix), test whether a VibeSim agent, given only the **before** state, can
rediscover an optimization that fixes the **ground-truth (GT) bottleneck**.
Run k trials, auto-evaluate each, report pass-rate.

Strictly **CPU / cache-only**: no trial may touch a GPU. A warm
`profiling/profile.db` makes timing-predict + analyze GPU-free; a cold cache
would JIT-grab a GPU (forbidden). Trials that would miss the cache must abort,
not profile.

## Layered evaluation (per trial)

1. **Objective sim gate (hard pass/fail, deterministic, GPU-free).**
   Re-run `analyze optimality-scoped` (cache-only) on the agent's produced
   sim output and compare the GT scope against the recorded before/after:
   - *structural*: the GT "redundant" leaves must disappear from the scope's
     `descendant_leaves` (e.g. `*_input_copy` gone).
   - *metric*: the scope rungs (r0_measured / r5_hardware_limit) must fall to
     within tolerance of the recorded GT "after" values, while
     r6_segmented_necessary (necessary work) stays unchanged.
2. **Semantic judge (score, gpt-5.6-luna).** A judge reads the agent's
   diagnosis + proposed change and rules whether the root cause it identified
   matches the real PR's root cause. Independent of the objective gate.

A trial is **CORRECT** iff the objective gate PASSES. The judge score is a
secondary signal (catches "right diagnosis, wrong/incomplete edit" and
"passed the metric by accident / for the wrong reason").

## Layout

```
scripts-local/eval-pipeline/
  README.md            # this file
  issues/
    vllm-28103.yaml     # GT spec: scope selector, before/after leaves+rungs, PR root cause, tolerances
  lib/                 # (built after arch edit-surface is mapped)
    create_workspace.py  # POST http://172.17.0.1:8766/api/agent/workspaces
    run_agent.py         # create conversation (gpt-5.6-luna @ max, orchestrated) + send task; poll result
    objective_gate.py    # cache-only analyze on agent output -> structural+metric PASS/FAIL
    judge.py             # gpt-5.6-luna semantic verdict vs GT root cause
    run_trial.py         # one trial end to end
    run_eval.py          # k trials -> aggregate report
  runs/<issue>/<ts>/    # per-eval outputs (per-trial logs, gate results, judge verdicts, summary.json)
```

## Backend endpoints (host: http://172.17.0.1:8766, currently not token-gated)

- Create workspace: `POST /api/agent/workspaces` `{"displayName": "..."}`
- Create conversation: `POST /api/agent/workspaces/{wid}/conversations`
  with `codex_runtime` pinning every role to `{"model":"gpt-5.6-luna","effort":"max","serviceTier":"default"}`.
- Send task (synchronous, minutes): `POST /api/agent/workspaces/{wid}/conversations/{cid}/messages` `{"text":"..."}`

Analyzer (read-only, cache-only): `main/target/release/analyze optimality-scoped <logdir> --label "<node_label>"`
writes `reports/optimality_scoped_*.json` and never mutates the run's normal reports.
