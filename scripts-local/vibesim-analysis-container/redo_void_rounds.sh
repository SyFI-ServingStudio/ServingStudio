#!/usr/bin/env bash
# 2026-09-28: Bedrock (Opus 5.5) flaps between OK and 503 every 10-20 min. Rounds whose agent died in the 503 storm
# (>= 10 api_retry lines and fewer than 15 assistant turns) are VOID, not results. After resume_mixed_verify.sh
# finishes, find such rounds in the mixed/verify cases, move their artifacts to trial_<k>_api503<pass>_*, record a
# VOID line in rounds.jsonl and re-run them behind the Bedrock gate (bedrock_gate_and_resume.sh -> resume_mixed_verify.sh
# re-enters every round without a verdict). Up to MAX_PASSES passes.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MAX_PASSES="${MAX_PASSES:-3}"
void_rounds() { # prints "case k" lines
  python3 - "$HERE" <<'PY'
import re, sys, pathlib, json
here = pathlib.Path(sys.argv[1])
for case in ("kda_mixed_claude", "mla_mixed_claude", "kda_verify_claude", "mla_verify_claude"):
    d = here / f"iter_opt_eval_k3_{case}"
    for log in sorted(d.glob("trial_[0-9]*_agent.log")):
        m = re.match(r"trial_(\d+)_agent\.log$", log.name)
        if not m: continue
        k = int(m.group(1)); txt = log.read_text(errors="replace")
        retries = txt.count('"subtype":"api_retry"'); turns = txt.count('"type":"assistant"')
        if retries >= 10 and turns < 15:
            print(case, k, retries, turns)
PY
}
for pass_ in $(seq 1 "$MAX_PASSES"); do
  while pgrep -f "[r]esume_mixed_verify\.sh|[b]edrock_gate_and_resume\.sh" >/dev/null; do sleep 60; done
  mapfile -t voids < <(void_rounds)
  [ "${#voids[@]}" -gt 0 ] || { echo "$(date +%F_%T) no void rounds -> done"; exit 0; }
  echo "$(date +%F_%T) pass $pass_: void rounds: ${voids[*]}"
  for line in "${voids[@]}"; do
    set -- $line; case="$1"; k="$2"; O="$HERE/iter_opt_eval_k3_$case"; tag="api503$(printf %c $((96 + pass_ + 2)))"
    for f in "$O"/trial_${k}_*; do b="$(basename "$f")"; case "$b" in *api503*) continue;; esac; mv "$f" "$O/trial_${k}_${tag}_${b#trial_${k}_}"; done
    python3 - "$O/rounds.jsonl" "$case" "$k" "$3" "$4" "$tag" <<'PY'
import json, sys, time
p, case, k, retries, turns, tag = sys.argv[1:]
open(p, "a").write(json.dumps({"round": int(k), "case": case, "verdict": "VOID", "reason": f"agent died in a Bedrock 503 storm ({retries} retries, {turns} assistant turns)",
                               "points": {}, "time": time.strftime("%F %T"), "note": f"artifacts kept as trial_{k}_{tag}_*; re-run by redo_void_rounds.sh"}) + "\n")
PY
  done
  "$HERE/bedrock_gate_and_resume.sh"
done
echo "$(date +%F_%T) redo passes exhausted"
