#!/usr/bin/env bash
# 2026-09-28 07:15: Bedrock (us-east-1, us.anthropic.claude-opus-5-5) returns 503 ServiceUnavailable; the KDA mixed r3
# and MLA mixed r2 agents exited after 10 retries with zero tokens (evidence kept as trial_*_api503_*). Gate the campaign
# on a live probe and resume when the endpoint answers twice in a row; resume_mixed_verify.sh skips every round that
# already has a verdict, so it continues with KDA mixed r3 and MLA mixed r2-r4.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export AWS_PROFILE="${AWS_PROFILE:-default}" AWS_REGION="${AWS_REGION:-us-east-1}"
probe() {
  timeout 90 python3 - <<'PY'
import boto3, botocore, sys
c = boto3.client("bedrock-runtime", region_name="us-east-1", config=botocore.config.Config(retries={"max_attempts": 1}))
try:
    r = c.converse(modelId="us.anthropic.claude-opus-5-5",
                   messages=[{"role": "user", "content": [{"text": "Reply with OK"}]}], inferenceConfig={"maxTokens": 5})
    print("OK", r["output"]["message"]["content"][0]["text"].strip()); sys.exit(0)
except botocore.exceptions.ClientError as e:
    print("ERR", e.response["ResponseMetadata"]["HTTPStatusCode"], e.response["Error"]["Code"]); sys.exit(1)
except Exception as e:
    print("EXC", type(e).__name__, str(e)[:120]); sys.exit(1)
PY
}
ok=0
while :; do
  out="$(probe 2>&1 | tail -n 1)"
  echo "$(date +%F_%T) probe: $out"
  case "$out" in OK*) ok=$((ok + 1)) ;; *) ok=0 ;; esac
  [ "$ok" -ge 2 ] && break
  sleep $([ "$ok" -eq 1 ] && echo 60 || echo 120)
done
echo "$(date +%F_%T) Bedrock healthy (2 consecutive probes) -> resuming"
exec "$HERE/resume_mixed_verify.sh"
