# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Return success when changed paths require the Forge loop end-to-end check."""

from __future__ import annotations

import fnmatch
import sys
from collections.abc import Iterable


# This smoke exercises ``kernelforge forge-loop``. Keep shared package surfaces
# in scope because the loop imports them, but do not spend a GPU run on changes
# confined to the independent fusion or GEMM-tuning products.
FORGE_E2E_PATHS = (
    "src/kernelforge/**",
    "pyproject.toml",
    "examples/triton-softmax-forge-loop/**",
    ".github/workflows/forge-e2e.yml",
    ".github/scripts/forge-ci-e2e-dispatch.sh",
    ".github/scripts/forge_e2e_gate.py",
    ".github/scripts/forge_e2e_report.py",
)

FORGE_E2E_EXCLUDED_PATHS = (
    "src/kernelforge/fusion/**",
    "src/kernelforge/gemm_tune/**",
    "src/kernelforge/tests/fusion/**",
)


def requires_forge_e2e(paths: Iterable[str]) -> bool:
    """Whether any normalized repository path affects the Forge loop smoke."""

    for path in paths:
        normalized = path.strip().replace("\\", "/")
        if not normalized:
            continue
        if any(fnmatch.fnmatchcase(normalized, pattern) for pattern in FORGE_E2E_EXCLUDED_PATHS):
            continue
        if any(fnmatch.fnmatchcase(normalized, pattern) for pattern in FORGE_E2E_PATHS):
            return True
    return False


def main() -> int:
    # Consume all input before exiting: this runs after jq with pipefail enabled,
    # and an early exit could SIGPIPE jq and turn a positive match into failure.
    return 0 if requires_forge_e2e(sys.stdin.read().splitlines()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
