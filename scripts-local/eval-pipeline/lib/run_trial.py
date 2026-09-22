#!/usr/bin/env python3
"""Run ONE eval trial: reset repo -> agent turn -> objective gate -> judge.

The objective gate re-simulates from OUR pinned before.json against the agent's
edited repo, so a config/arch-selector swap cannot pass (that pinned config still
selects the before arch). Everything is cache-only / GPU-free.
"""
from __future__ import annotations
import json, subprocess, sys, time, glob, os, re, argparse
from pathlib import Path
import yaml
import vibesim_client as vc
import objective_gate
import judge as judge_mod
import trajectory_scan

WS = Path("/raid/yilegu/roofline_guided_agent/VibeSimWorkspace")
LIB = Path(__file__).resolve().parent
HOME = os.path.expanduser("~")
# Host-side gate shells out to `uv`, `cargo`, `node`, `protoc`; mirror the PATH
# the service scripts export so cargo_build finds the toolchain.
TOOL_PATH = ":".join([f"{HOME}/.cargo/bin", f"{HOME}/.local/bin",
                      f"{HOME}/.local/node22/bin", f"{HOME}/.local/protoc/bin",
                      os.environ.get("PATH", "")])


def _gate_env():
    env = dict(os.environ)
    env["PATH"] = TOOL_PATH
    # Hard zero-GPU guarantee for the host-side gate: some repo versions'
    # timing-predict enables JIT profiling on cache miss (would grab a GPU).
    # With no visible device a miss errors instead — cache-only by force.
    env["CUDA_VISIBLE_DEVICES"] = ""
    # /raid/tmp/vibesim-launcher-locks is owned by another account; keep the
    # launcher's lock/stream tmp under our own tree.
    env["TMPDIR"] = "/raid/yilegu/tmp"
    return env


def sh(cmd, cwd, timeout=3600, env=None):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True,
                       text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


def baseline_commit(repo: Path, spec: dict) -> str:
    """The sealed before-state commit. Prefer an explicit hash pinned in the
    spec (harness.baseline_commit) — the 'eval base' message grep picks the
    NEWEST match, which once selected a maintenance commit made on top of an
    agent's fixed tree and silently pre-solved a rerun."""
    pinned = (spec.get("harness") or {}).get("baseline_commit")
    if pinned:
        rc, out, _ = sh(f"git rev-parse --verify {pinned}^{{commit}}", repo)
        if rc != 0:
            raise RuntimeError(f"pinned baseline {pinned} not found in repo")
        return out.strip()
    _, out, _ = sh("git log --all --grep='eval base' --format=%H -n1", repo)
    c = out.strip().splitlines()[0] if out.strip() else ""
    if not c:
        raise RuntimeError("could not resolve baseline 'eval base' commit")
    return c


def git_reset(repo: Path, baseline: str):
    # Restore the pristine before-state: switch to the baseline branch, hard-reset
    # it to the sealed commit, delete every other (agent-created) branch, and wipe
    # ALL untracked files including .gitignore'd ones (-x) so nothing an agent
    # wrote — logs, tmp artifacts, caches — leaks into the next trial. Exempt only
    # the shared caches that are safe or self-healing: profiling/ (warm profile.db,
    # required for cache-only re-sim), target/ (cargo invalidates stale artifacts
    # when sources revert), .venv (toolchain; pip installs are a residual risk we
    # accept and can audit via the trajectory scan).
    sh("git checkout -f main 2>/dev/null || git checkout -f master", repo)
    sh(f"git reset --hard {baseline}", repo)
    sh("git for-each-ref --format='%(refname:short)' refs/heads "
       "| grep -vx main | grep -vx master | xargs -r git branch -D", repo)
    sh("git clean -fdqx -e profiling -e target -e .venv", repo)
    (repo / "logs").mkdir(exist_ok=True)


def run_gate(repo: Path, spec: dict, outdir: Path) -> dict:
    """Cache-only re-sim from pinned before.json + scoped analyze + gate."""
    env = _gate_env()
    log = {}
    # 1. re-predict (GPU-free; a miss ERRORS, never profiles)
    rc, out, err = sh("uv run python -m launcher timing-predict eval-configs/before.json --no-analyze",
                      repo, timeout=2400, env=env)
    (outdir / "gate_predict.log").write_text(out + "\n--- STDERR ---\n" + err)
    log["predict_rc"] = rc
    logdir = repo / "logs" / "eval_before_run"
    if rc != 0:
        return {"pass": False, "error": "timing-predict failed", **log}
    # 2. scoped analyze on the GT scope. Older repos' own analyze predates
    # `optimality-scoped`; specs may pin a newer binary via harness.analyze_bin.
    path = spec["gt_scope"]["path"]
    analyze_bin = (spec.get("harness") or {}).get("analyze_bin", "./target/release/analyze")
    rc, out, err = sh(f'{analyze_bin} optimality-scoped logs/eval_before_run --path "{path}"',
                      repo, timeout=1200, env=env)
    (outdir / "gate_analyze.log").write_text(out + "\n--- STDERR ---\n" + err)
    log["analyze_rc"] = rc
    reports = sorted(glob.glob(str(logdir / "reports" / "optimality_scoped_*.json")),
                     key=os.path.getmtime, reverse=True)
    if not reports:
        return {"pass": False, "error": "no scoped report produced", **log}
    report = json.loads(Path(reports[0]).read_text())
    (outdir / "scoped_report.json").write_text(json.dumps(report, indent=2))
    result = objective_gate.evaluate(spec, report)
    result.update(log)
    result["scoped_leaves"] = report.get("selection", {}).get("descendant_leaves")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wid", required=True)
    ap.add_argument("--issue", required=True, type=Path)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(args.issue.read_text())
    wid = args.wid
    repo = WS / "agent-workspaces" / wid / "repo"
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rec = {"trial": args.trial, "wid": wid, "issue": spec["issue_id"], "ts": time.time()}

    baseline = baseline_commit(repo, spec)
    rec["baseline_commit"] = baseline
    print(f"[trial {args.trial}] reset repo -> baseline {baseline[:10]}")
    git_reset(repo, baseline)

    print(f"[trial {args.trial}] create agent conversation (gpt-5.6-luna@max)")
    cid = vc.create_conversation(wid, runtime=vc.LUNA_MAX, agent_mode="orchestrated", autonomous=True)
    rec["conversation_id"] = cid
    task_file = (spec.get("harness") or {}).get("task_file", "agent_task.md")
    task = (LIB / task_file).read_text()
    print(f"[trial {args.trial}] send task (synchronous; minutes)")
    t0 = time.time()
    try:
        res = vc.send_message(wid, cid, task, timeout=5400)
    except Exception as e:
        rec["agent_error"] = str(e)
        (outdir / "trial.json").write_text(json.dumps(rec, indent=2))
        print(f"[trial {args.trial}] AGENT ERROR: {e}")
        return
    rec["agent_seconds"] = round(time.time() - t0, 1)
    rec["agent_final"] = res.get("final")
    (outdir / "agent_result.json").write_text(json.dumps(res, indent=2)[:2_000_000])

    # capture the agent's full change vs the sealed baseline (agents COMMIT on a
    # branch, so `git diff HEAD` would be empty — diff against the baseline commit).
    _, diff, _ = sh(f"git diff {baseline}", repo, timeout=120)
    (outdir / "agent.diff").write_text(diff)

    print(f"[trial {args.trial}] objective gate (cache-only re-sim)")
    rec["gate"] = run_gate(repo, spec, outdir)

    if not args.skip_judge:
        print(f"[trial {args.trial}] semantic judge")
        try:
            rec["judge"] = judge_mod.judge(wid, spec["gt_root_cause"], rec.get("agent_final") or "")
        except Exception as e:
            rec["judge"] = {"error": str(e)}

    # anti-reward-hacking: scan the agent's tool-call trajectory
    issue_num = re.findall(r"\d{4,6}", spec["issue_id"])
    rec["trajectory"] = trajectory_scan.scan(cid, issue_numbers=issue_num)

    rec["correct"] = bool(rec["gate"].get("pass")) and not rec["trajectory"]["hard_fail"]
    (outdir / "trial.json").write_text(json.dumps(rec, indent=2))

    # title the conversations so the viz-ui list is legible (no rename API;
    # write the store's title column directly — same field the generator sets)
    try:
        import sqlite3
        db = WS / "agent-workspaces" / wid / "workspace.sqlite"
        con = sqlite3.connect(db, timeout=10)
        con.execute("UPDATE conversations SET title=? WHERE id=?",
                    (f"eval {spec['issue_id']} · trial {args.trial} agent", cid))
        jcid = (rec.get("judge") or {}).get("_conversation_id")
        if jcid:
            con.execute("UPDATE conversations SET title=? WHERE id=?",
                        (f"eval {spec['issue_id']} · trial {args.trial} judge", jcid))
        con.commit(); con.close()
    except Exception as e:
        print(f"[trial {args.trial}] (title update skipped: {e})")
    print(f"[trial {args.trial}] DONE  correct={rec['correct']}  "
          f"gate={rec['gate'].get('pass')}  hack_flags={rec['trajectory']['n_hard']}h/"
          f"{rec['trajectory']['n_soft']}s  "
          f"judge={rec.get('judge',{}).get('matches_root_cause')}")


if __name__ == "__main__":
    main()
