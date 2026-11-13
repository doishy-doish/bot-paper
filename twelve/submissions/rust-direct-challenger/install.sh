#!/usr/bin/env bash
# install.sh: runs once as root on a clean Ubuntu 24.04. Installs the pinned Rust toolchain.
set -euo pipefail
apt-get update && apt-get install -y build-essential curl pkg-config libssl-dev
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustup toolchain install 1.89.0 --profile minimal
rustup default 1.89.0
