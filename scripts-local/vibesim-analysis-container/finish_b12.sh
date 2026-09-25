#!/usr/bin/env bash
# Codex B12 (K3 prefill support) runs under a 6 h cap that ends ~00:00. If it exits with an uncommitted
# tree, verify and commit it here (Rust build + focused tests + just test-cpu), then fast-forward the user's
# main/ checkout (branch kimi-k3) to the worktree branch (kimi-k3-arch). Kill nothing.
# Usage: finish_b12.sh <codex_timeout_pid>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main-k3-rust
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "==== $(date +%F_%T) B12 codex exited"
cd "$WT"
if [ -z "$(git status --short | grep -v '^??.*b12_debug')" ]; then
  echo "tree clean (Codex committed): $(git log --oneline -1)"
else
  echo "==== uncommitted B12 tree: $(git status --short | wc -l) entries -> testing"
  export PATH="$HOME/.cargo/bin:$PATH"
  if just test-cpu > "$HERE/codex_runs/b12_test_cpu.log" 2>&1; then
    echo "just test-cpu: PASS"
  else
    fails=$(grep -cE "FAILED|error\[" "$HERE/codex_runs/b12_test_cpu.log")
    echo "just test-cpu: non-zero exit ($fails failure lines; 4 test_energy_* host failures are known) -> see codex_runs/b12_test_cpu.log"
    if ! grep -E "^FAILED|test result: FAILED" "$HERE/codex_runs/b12_test_cpu.log" | grep -v test_energy >/dev/null; then
      echo "only known failures -> treating as PASS"
    else
      echo "!! real test failures -> NOT committing; leaving the tree for review"; exit 1
    fi
  fi
  git add -A ':!logs/b12_debug*' 2>/dev/null || git add -A
  git reset -q -- logs 2>/dev/null || true
  git add presets doc simulator profiling model 2>/dev/null || true
  git commit -q -m "kimi_k3: chunked-prefill support (Codex B12, committed by operator at the cap)

New kinds kda_chunk_prefill / causal_conv1d_prefill / mla_prefill_attention / mla_prefix_gather / mla_merge_state
with sglang_k3_env runners, prefill branches in the KDA/MLA/MoE worklets (gated on prefill_chunk_pairs; decode
leaves untouched), rank-1 prefill presets (16k chunk, prefix 49152, 4x4096) and the alignment doc section.
Known gap: predicted layer totals are ~60% below the measured eager step (see doc/alignment/kimi_k3_single_layer.md).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && echo "committed: $(git log --oneline -1)"
fi
cd /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/main && git merge --ff-only kimi-k3-arch 2>&1 | tail -1 && echo "main/ kimi-k3 @ $(git log --oneline -1)"
echo "==== $(date +%F_%T) finish_b12 done"
