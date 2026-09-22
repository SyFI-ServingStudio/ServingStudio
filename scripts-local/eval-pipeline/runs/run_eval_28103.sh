export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$HOME/.local/node22/bin:$HOME/.local/protoc/bin:$PATH"
cd /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/eval-pipeline/lib
python3 run_eval.py --wid w_45893bc43b51 \
  --issue ../issues/vllm-28103.yaml -k 3
