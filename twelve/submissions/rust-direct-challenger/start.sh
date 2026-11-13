#!/usr/bin/env bash
# start.sh: runs the server in the foreground. Configured only by SPEC env vars.
set -euo pipefail
exec "$(dirname "$0")/target/release/twelve-direct"
