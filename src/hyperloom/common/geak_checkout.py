# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Location of the GEAK checkout the kernel phase runs."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_e2e_runner() -> str:
    """Resolve run_e2e.py from $GEAK_E2E_RUNNER / $GEAK_ROOT (GEAK@GEAK)."""
    runner = os.environ.get("GEAK_E2E_RUNNER", "").strip()
    if runner and Path(runner).is_file():
        return runner
    roots: list[str] = []
    root = os.environ.get("GEAK_ROOT", "").strip()
    if root:
        roots.append(root)
    cache_dir = os.environ.get("HYPERLOOM_CACHE_DIR", "").strip()
    if cache_dir:
        # install.sh clones GEAK per revision as <cache>/GEAK@<sha>; prefer the newest such checkout, then the bare
        # dir.
        pinned = sorted(
            (p for p in Path(cache_dir).glob("GEAK@*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        roots.extend(str(p) for p in pinned)
        roots.append(str(Path(cache_dir) / "GEAK"))
    for root in dict.fromkeys(roots):
        cand = Path(root) / "interface" / "run_e2e.py"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "e2e runner not found. Set GEAK_E2E_RUNNER to "
        "<GEAK checkout>/interface/run_e2e.py (the installer exports it), "
        "or set GEAK_ROOT/HYPERLOOM_CACHE_DIR."
    )
