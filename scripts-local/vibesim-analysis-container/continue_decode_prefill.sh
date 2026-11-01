#!/usr/bin/env bash
# 2026-09-29: continue the Claude Opus 5.5 rounds on the decode and prefill cases that had NOT plateaued when
# campaign 6 stopped (plateau = the last two judged rounds are both non-PASS). Round-robin over the active cases,
# one agent at a time; each round is gated on Bedrock (NEED_OK consecutive probes) + a schedulable slurm GPU, and a
# no-information round (tree byte-identical to its start tree, or >= 10 Bedrock retries with < 15 assistant turns)
# is archived as trial_<k>_void<n>_* and re-run (2026-10-04: an unchanged tree counts as VOID only with < 15 turns;
# an agent that worked and reverted everything is a genuine null) (up to MAX_REDO times) instead of counting as a null.
# At the start (history per case, by round): kda_shapes P P P · mla_shapes P P F · kda_b512 P P F · mla_b512 P P P ·
# mla_lcprefill P P F F P P · mla_prefill P P F. Plateaued and not run: kda_lcprefill, kda_prefill.
# Usage: setsid nohup ./continue_decode_prefill.sh > continue_decode_prefill.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CASES="${CASES:-mla_lcprefill_claude kda_shapes_claude mla_b512_claude mla_shapes_claude kda_b512_claude mla_prefill_claude}"
MAX_NEW="${MAX_NEW:-4}" MAX_REDO="${MAX_REDO:-2}"
# IGNORE_PLATEAU=1 (2026-10-04): run every listed case MAX_NEW rounds regardless of plateau, so KDA and MLA get
# the same number of rounds on the same workloads (list them pairwise in CASES).
export NEED_OK="${NEED_OK:-5}" K3_GATE_ONLY=1 AGENT_TIMEOUT="${AGENT_TIMEOUT:-2700}"
declare -A NEW
for c in $CASES; do NEW[$c]=0; done

state() { # prints "<next_k> <trailing_non_pass>" for case $1
  python3 - "$HERE/iter_opt_eval_k3_$1" <<'PY'
import json, re, sys, pathlib
d = pathlib.Path(sys.argv[1]); last = {}
for l in open(d / "rounds.jsonl"):
    if not l.strip(): continue
    x = json.loads(l)
    if x.get("verdict") != "VOID": last[int(x["round"])] = x.get("verdict")
    else: last.pop(int(x["round"]), None)  # a VOID supersedes the round's earlier record; the round is re-run
ks = [int(m.group(1)) for p in d.glob("trial_*_verdict.json") if (m := re.fullmatch(r"trial_(\d+)_verdict\.json", p.name))]
trail = 0
for k in sorted(last, reverse=True):
    if last[k] == "PASS": break
    trail += 1
print(max(ks + list(last) + [0]) + 1, trail)
PY
}
is_void() { # case k -> prints "retries turns" and exits 0 when the round carries no information
  python3 - "$HERE/iter_opt_eval_k3_$1" "$2" "$3" <<'PY'
import sys, pathlib, subprocess
d, k, start = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
v = d / f"trial_{k}_verdict.json"
if v.exists() and '"verdict": "PASS"' in v.read_text(): sys.exit(1)
log = d / f"trial_{k}_agent.log"; txt = log.read_text(errors="replace") if log.exists() else ""
retries = txt.count('"subtype":"api_retry"'); turns = txt.count('"type":"assistant"')
t = d / f"trial_{k}_tree"
same = t.is_dir() and subprocess.run(["diff", "-rq", "-x", "__pycache__", start, str(t)], capture_output=True, text=True).stdout.strip() == ""
print(retries, turns, "same" if same else "")
b = d / f"trial_{k}_build.log"  # rebuild smoke starved of a GPU -> false "non-compiling" FAIL
starved = b.exists() and "TIMEOUT after" in b.read_text(errors="replace") and "waiting for the GPU job" in b.read_text(errors="replace")
sys.exit(0 if ((same and turns < 15) or starved or (retries >= 10 and turns < 15) or not log.exists() or not v.exists()) else 1)
PY
}

while :; do
  ran=0
  for c in $CASES; do
    read -r k trail < <(state "$c")
    if [ "$trail" -ge 2 ] && [ "${IGNORE_PLATEAU:-0}" != 1 ]; then echo "$(date +%F_%T) $c plateaued (last 2 rounds non-PASS)"; continue; fi
    if [ "${NEW[$c]}" -ge "$MAX_NEW" ]; then echo "$(date +%F_%T) $c: $MAX_NEW new rounds done"; continue; fi
    O="$HERE/iter_opt_eval_k3_$c"
    for redo in $(seq 0 "$MAX_REDO"); do
      "$HERE/bedrock_gate_and_resume.sh"
      rm -rf "$O/.start_tree_r$k"; cp -a "$O/best_tree" "$O/.start_tree_r$k"
      echo "==== $(date +%F_%T) $c round $k (attempt $redo)"
      "$HERE/run_k3_continuous.sh" slurm "$c" "$k" 1
      if info="$(is_void "$c" "$k" "$O/.start_tree_r$k")"; then
        tag="void$((redo + 1))_$(date +%H%M)"
        echo "$(date +%F_%T) $c round $k VOID ($info) -> archived as trial_${k}_${tag}_*, re-run"
        for f in "$O"/trial_${k}_*; do b="$(basename "$f")"; case "$b" in *void*|*api503*) continue;; esac; mv "$f" "$O/trial_${k}_${tag}_${b#trial_${k}_}"; done
        python3 - "$O/rounds.jsonl" "$c" "$k" "$info" "$tag" <<'PY'
import json, sys, time
p, case, k, info, tag = sys.argv[1:]
open(p, "a").write(json.dumps({"round": int(k), "case": case, "verdict": "VOID", "reason": f"no-information round (retries turns same: {info})",
                               "points": {}, "time": time.strftime("%F %T"), "note": f"artifacts kept as trial_{k}_{tag}_*; re-run by continue_decode_prefill.sh"}) + "\n")
PY
        rm -rf "$O/.start_tree_r$k"; continue
      fi
      rm -rf "$O/.start_tree_r$k"; break
    done
    NEW[$c]=$((NEW[$c] + 1)); ran=1
    echo "$(date +%F_%T) $c round $k: $(tail -n 1 "$O/rounds.jsonl" | cut -c1-300)"
  done
  [ "$ran" = 1 ] || break
done
echo "==== $(date +%F_%T) continue_decode_prefill done"
