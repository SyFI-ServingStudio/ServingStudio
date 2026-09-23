#!/usr/bin/env bash
# Start MLA trial 7 (correct oracle 8802) once the re-judge + fill chain has finished.
# Lives in a file so its command line does not contain the trial runner's name: the chain
# gates on `pgrep -f "[r]un_k3_trials_direct.sh"`, and an inline waiter carrying that string
# deadlocked both waiters (2026-09-23 21:29).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$HERE/run_k3_trials_direct.sh"
while pgrep -f "[a]fter_trials_rejudge_and_fill.sh" >/dev/null; do sleep 120; done
echo "$(date +%F_%T) rejudge+fill done -> MLA trial 7 (oracle 8802)"
"$RUNNER" 3 mla:7
