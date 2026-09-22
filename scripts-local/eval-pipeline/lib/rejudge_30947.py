#!/usr/bin/env python3
"""One-off: re-judge sglang-30947 trials 1+2 after the judge-prompt fix
(the old prompt hardcoded 28103's root cause). Preserves the old verdict
as judge_prepatch. Detached run; progress to stdout."""
import json, glob, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yaml
import judge as judge_mod

BASE = pathlib.Path(__file__).resolve().parent.parent
spec = yaml.safe_load((BASE / "issues/sglang-30947.yaml").read_text())
gt = spec["gt_root_cause"]

for n in (1, 2):
    t = sorted(glob.glob(str(BASE / f"runs/sglang-30947/*/trial_{n}/trial.json")))[-1]
    d = json.loads(open(t).read())
    if "judge_prepatch" in d:
        print(f"trial {n}: already re-judged, skip", flush=True)
        continue
    print(f"trial {n}: judging...", flush=True)
    v = judge_mod.judge(d["wid"], gt, d.get("agent_final") or "")
    d["judge_prepatch"] = d.get("judge")
    d["judge"] = v
    open(t, "w").write(json.dumps(d, indent=2))
    print(f"trial {n}: old={d['judge_prepatch'].get('matches_root_cause')} -> "
          f"new={v.get('matches_root_cause')} score={v.get('score')} | "
          f"{(v.get('reasoning') or '')[:140]}", flush=True)
print("DONE", flush=True)
