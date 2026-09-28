#!/usr/bin/env bash
# 2026-09-28: Bedrock (Opus 5.5) flaps between OK and 503; agents die mid-round after 10 retries. A round is VOID
# (not a result) when its agent log shows >= 10 api_retry lines AND either fewer than 15 assistant turns or an agent
# tree byte-identical to the tree it started from (best_tree). After the resume loop finishes, void rounds in the
# mixed/verify cases are archived as trial_<k>_api503<pass>_*, recorded in rounds.jsonl and re-run behind the Bedrock
# gate (bedrock_gate_and_resume.sh -> resume_mixed_verify.sh re-enters every round without a verdict). Up to MAX_PASSES.
# v2 (12:05): tree-identity criterion (r2 of MLA mixed ran 67 turns and edited nothing), NEED_OK=15 (the 5-probe gate
# resumed twice into windows that lasted < 15 min).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MAX_PASSES="${MAX_PASSES:-4}"
export NEED_OK="${NEED_OK:-15}"
void_rounds() { # prints "case k retries turns" lines
  python3 - "$HERE" <<'PY'
import re, sys, pathlib, subprocess
here = pathlib.Path(sys.argv[1])
for case in ("kda_mixed_claude", "mla_mixed_claude", "kda_verify_claude", "mla_verify_claude"):
    d = here / f"iter_opt_eval_k3_{case}"
    for log in sorted(d.glob("trial_[0-9]*_agent.log")):
        m = re.match(r"trial_(\d+)_agent\.log$", log.name)
        if not m: continue
        k = int(m.group(1)); txt = log.read_text(errors="replace")
        retries = txt.count('"subtype":"api_retry"'); turns = txt.count('"type":"assistant"')
        v = d / f"trial_{k}_verdict.json"
        if v.exists() and '"verdict": "PASS"' in v.read_text(): continue
        same = False
        t = d / f"trial_{k}_tree"
        if t.is_dir() and (d / "best_tree").is_dir():
            same = subprocess.run(["diff", "-rq", "-x", "__pycache__", str(d / "best_tree"), str(t)], capture_output=True, text=True).stdout.strip() == ""
        # v3 (12:35): a non-PASS round whose tree is byte-identical to its start tree carries no information, whatever
        # killed it (Bedrock 503 storm, or every GPU request queued behind placeholders as in MLA mixed r1) -> re-run.
        if same or (retries >= 10 and turns < 15):
            print(case, k, retries, turns)
PY
}
for pass_ in $(seq 1 "$MAX_PASSES"); do
  while pgrep -f "[r]esume_mixed_verify\.sh|[b]edrock_gate_and_resume\.sh|[r]un_iter_opt_eval\.sh" >/dev/null; do sleep 60; done
  mapfile -t voids < <(void_rounds)
  [ "${#voids[@]}" -gt 0 ] || { echo "$(date +%F_%T) no void rounds -> done"; exit 0; }
  echo "$(date +%F_%T) pass $pass_: void rounds: ${voids[*]}"
  for line in "${voids[@]}"; do
    set -- $line; case="$1"; k="$2"; O="$HERE/iter_opt_eval_k3_$case"; tag="api503_p${pass_}_$(date +%H%M)"
    for f in "$O"/trial_${k}_*; do b="$(basename "$f")"; case "$b" in *api503*) continue;; esac; mv "$f" "$O/trial_${k}_${tag}_${b#trial_${k}_}"; done
    python3 - "$O/rounds.jsonl" "$case" "$k" "$3" "$4" "$tag" <<'PY'
import json, sys, time
p, case, k, retries, turns, tag = sys.argv[1:]
open(p, "a").write(json.dumps({"round": int(k), "case": case, "verdict": "VOID", "reason": f"no-information round: tree unchanged or agent died early ({retries} Bedrock retries, {turns} assistant turns)",
                               "points": {}, "time": time.strftime("%F %T"), "note": f"artifacts kept as trial_{k}_{tag}_*; re-run by redo_void_rounds.sh"}) + "\n")
PY
  done
  "$HERE/bedrock_gate_and_resume.sh"
done
echo "$(date +%F_%T) redo passes exhausted"
