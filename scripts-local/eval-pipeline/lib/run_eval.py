#!/usr/bin/env python3
"""Run k eval trials sequentially in one shared workspace, then aggregate.

Trials are sequential because they share the workspace repo (git-reset between
each). Each trial is an independent agent conversation.
"""
from __future__ import annotations
import json, subprocess, sys, time, argparse
from pathlib import Path
import yaml

LIB = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wid", required=True, help="shared base workspace id")
    ap.add_argument("--issue", required=True, type=Path)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--start", type=int, default=1, help="first trial index to run (resume)")
    ap.add_argument("--run-ts", default=None, help="reuse an existing run dir <issue>/<ts>")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(args.issue.read_text())
    ts = args.run_ts or time.strftime("%Y%m%d_%H%M%S")
    run_root = LIB.parent / "runs" / spec["issue_id"] / ts
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"=== eval {spec['issue_id']}  trials {args.start}..{args.k}  wid={args.wid}  -> {run_root}")

    for i in range(args.start, args.k + 1):
        outdir = run_root / f"trial_{i}"
        cmd = [sys.executable, str(LIB / "run_trial.py"),
               "--wid", args.wid, "--issue", str(args.issue),
               "--trial", str(i), "--outdir", str(outdir)]
        if args.skip_judge:
            cmd.append("--skip-judge")
        print(f"--- trial {i}/{args.k} ---")
        subprocess.run(cmd, check=False)

    # Aggregate over ALL trial_* dirs present (covers resumed runs).
    trials = []
    for d in sorted(run_root.glob("trial_*")):
        tj = d / "trial.json"
        n = int(d.name.split("_")[1])
        trials.append(json.loads(tj.read_text()) if tj.exists() else {"trial": n, "error": "no trial.json"})
    k_eff = len(trials)

    passed = [t for t in trials if t.get("correct")]
    summary = {
        "issue": spec["issue_id"], "k": k_eff, "wid": args.wid, "ts": ts,
        "gate_pass_rate": f"{len(passed)}/{k_eff}",
        "judge_match_rate": f"{sum(1 for t in trials if (t.get('judge') or {}).get('matches_root_cause'))}/{k_eff}",
        "trials": [{"trial": t.get("trial"), "correct": t.get("correct"),
                    "gate_leaves": (t.get("gate") or {}).get("scoped_leaves"),
                    "judge_match": (t.get("judge") or {}).get("matches_root_cause"),
                    "agent_seconds": t.get("agent_seconds"),
                    "agent_error": t.get("agent_error")} for t in trials],
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
