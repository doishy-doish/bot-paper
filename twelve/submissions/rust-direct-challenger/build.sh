#!/usr/bin/env bash
# build.sh: runs as a normal user. Builds the release binary (pinned deps in Cargo.toml).
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release
