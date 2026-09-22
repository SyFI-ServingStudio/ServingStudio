export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$HOME/.local/node22/bin:$HOME/.local/protoc/bin:$PATH"
set -e
R=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/agent-workspaces/w_45893bc43b51/repo
cd $R
# apply the fix: flip COPY_INPUTS true -> false in the before recipe
python3 - <<'PY'
import re,pathlib
f=pathlib.Path("simulator/src/arch/qwen3_dense_vllm_before.rs")
t=f.read_text()
t2=t.replace("const COPY_INPUTS: bool = true;","const COPY_INPUTS: bool = false;")
assert t2!=t, "COPY_INPUTS line not found/changed"
f.write_text(t2)
print("applied COPY_INPUTS=false")
PY
set -x
uv run python -m launcher timing-predict eval-configs/before.json --no-analyze
./target/release/analyze optimality-scoped logs/eval_before_run --path "iter/1/0/1"
echo "SELFTEST_FIX_SIM_OK"
