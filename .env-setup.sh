# Source this before running just/build/service commands in this workspace.
# Created during local setup: Rust, Node 22, protoc, and just are installed
# under $HOME because this host has no passwordless sudo.
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$HOME/.cargo/bin:$HOME/.local/node22/bin:$HOME/.local/protoc/bin:$PATH"
export TMPDIR="${TMPDIR:-/raid/tmp}"
mkdir -p "$TMPDIR"
