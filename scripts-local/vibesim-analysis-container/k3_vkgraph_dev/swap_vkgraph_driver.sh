#!/usr/bin/env bash
# Put the --verify-graph driver live without disturbing the running KDA-verify round 2 (2026-09-28).
#   1. wait for the GPU smoke test (slurm 2162) to finish with TEST_RC=0 (else stop, swap nothing);
#   2. wait until the round-2 judge job (k3j_k3_kda_verify_claude_2) has been RUNNING for >= 30 s: the judge
#      reads the case config and snapshots the driver in its first second, so from then on the swap cannot
#      change what round 2 is judged with;
#   3. cp the dev driver over the live one (byte-identical to what the test ran) and switch the KDA verify case
#      to --verify-graph (the MLA verify case was switched at 02:05). Round 3 and the MLA rounds then run in
#      graph mode; the judge re-keys goldens automatically (driver bytes are part of the case key).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEV="$HERE/k3_vkgraph_dev"
LOG="$DEV/out/test.slurm.out"
while ! grep -q "^TEST_RC=" "$LOG" 2>/dev/null; do sleep 30; done
if ! grep -q "^TEST_RC=0" "$LOG"; then echo "$(date +%F_%T) smoke test FAILED ($(grep '^TEST_RC=' "$LOG")); NOT swapping"; exit 1; fi
echo "$(date +%F_%T) smoke test passed"
O="$HERE/iter_opt_eval_k3_kda_verify_claude"
if [ ! -f "$O/trial_2_verdict.json" ]; then
  echo "$(date +%F_%T) waiting for the round-2 judge to start (k3j_k3_kda_verify_claude_2)"
  while :; do
    st="$(squeue -h -u "$USER" -n k3j_k3_kda_verify_claude_2 -o "%T %M" 2>/dev/null | head -1)"
    case "$st" in
      RUNNING*) secs="${st#RUNNING }"; m="${secs%%:*}"; s="${secs##*:}"; [ "$((10#$m * 60 + 10#$s))" -ge 30 ] && break ;;
    esac
    [ -f "$O/trial_2_verdict.json" ] && break
    sleep 10
  done
fi
echo "$(date +%F_%T) round-2 judge running (or done) -> swapping"
cp "$DEV/kimi_single_layer_decode.py" "$HERE/kimi_single_layer_decode.py"
cp "$HERE/issue_k3_kda_verify_claude.json" "$HERE/issue_k3_kda_verify_claude.json.pre_vkgraph"
cp "$DEV/issue_k3_kda_verify_claude.vkgraph.json" "$HERE/issue_k3_kda_verify_claude.json"
sha1sum "$HERE/kimi_single_layer_decode.py" "$DEV/kimi_single_layer_decode.py"
grep -c "verify-graph" "$HERE/issue_k3_kda_verify_claude.json" "$HERE/issue_k3_mla_verify_claude.json"
echo "$(date +%F_%T) SWAP DONE"
