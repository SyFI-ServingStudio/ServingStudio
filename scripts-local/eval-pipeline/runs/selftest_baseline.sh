export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$HOME/.local/node22/bin:$HOME/.local/protoc/bin:$PATH"
set -x
cd /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/agent-workspaces/w_45893bc43b51/repo
uv run python -m launcher timing-predict eval-configs/before.json --no-analyze && \
./target/release/analyze optimality-scoped logs/eval_before_run --path "iter/1/0/1" && \
echo "SELFTEST_PREDICT_ANALYZE_OK"
