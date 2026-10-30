#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
config="${1:?Usage: inference.sh CONFIG IMAGES CHECKPOINT [OPTIONS]}"
images="${2:?Usage: inference.sh CONFIG IMAGES CHECKPOINT [OPTIONS]}"
checkpoint="${3:?Usage: inference.sh CONFIG IMAGES CHECKPOINT [OPTIONS]}"
shift 3
exec python -m up inference --config "$config" --img_path "$images" --ckpt "$checkpoint" "$@"
