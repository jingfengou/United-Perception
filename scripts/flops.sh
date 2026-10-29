#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
config="${1:?Usage: flops.sh CONFIG [OPTIONS]}"
shift
exec python -m up flops --config "$config" "$@"
