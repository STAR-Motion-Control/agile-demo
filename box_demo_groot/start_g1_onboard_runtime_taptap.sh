#!/bin/bash
# Refactored runtime with the existing adaptive taptap recovery enabled.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -gt 0 && "$1" != --* ]]; then
    controller="$1"
    shift
    exec bash "$SCRIPT_DIR/start_g1_onboard_runtime.sh" \
        "$controller" --taptap --taptap-recovery adaptive "$@"
fi
exec bash "$SCRIPT_DIR/start_g1_onboard_runtime.sh" \
    --taptap --taptap-recovery adaptive "$@"
