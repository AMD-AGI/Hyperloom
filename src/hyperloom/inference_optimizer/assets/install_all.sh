#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Compatibility wrapper. The bare-metal installer was renamed to
# install_baremetal.sh; keep this entrypoint for existing docs and scripts.

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${_script_dir}/install_baremetal.sh" "$@"
