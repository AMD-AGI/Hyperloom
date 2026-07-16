#!/usr/bin/env bash
# Vultr preset (MI355X, docker runtime, /mnt/vast shared FS) around submit.sh.
# Usage: ./submit-vultr.sh [submit.sh options] <model_key> [more_keys...]
#   ./submit-vultr.sh deepseek_r1_sglang
#   ./submit-vultr.sh -b claude deepseek_r1_vllm
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/submit.sh" \
  --partition mi355x \
  --gpu-type mi355x \
  --shared-mount /mnt/vast \
  --source-dir /mnt/vast/hyperloom \
  "$@"
