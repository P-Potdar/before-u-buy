#!/bin/sh
# Create a source archive without credentials, dependencies, or build output.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
exec python3 "$ROOT/scripts/package_project.py" "${1:-$ROOT/deliverables/BeforeYouBuy-AI.zip}"
