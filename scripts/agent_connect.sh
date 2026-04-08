#!/usr/bin/env bash
# Backward-compatible wrapper.
# Prefer scripts/connect_gridnode.sh for direct control and documentation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -gt 0 ]]; then
    exec "$SCRIPT_DIR/connect_gridnode.sh" -- "$*"
else
    exec "$SCRIPT_DIR/connect_gridnode.sh"
fi
