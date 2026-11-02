#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
gpus="${1:?Usage: train.sh GPUS CONFIG [OPTIONS]}"
config="${2:?Usage: train.sh GPUS CONFIG [OPTIONS]}"
shift 2
exec python -m up train --ng "$gpus" --config "$config" "$@"
