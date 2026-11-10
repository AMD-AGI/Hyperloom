# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Running the kernel-tool steps on their own fails instead of reporting an install that did not happen."""

from __future__ import annotations

import subprocess
from pathlib import Path

_KERNEL_TOOLS = Path(__file__).resolve().parents[1] / "assets" / "install_kernel_tools.sh"


def test_direct_run_fails_and_points_at_install_sh(tmp_path: Path) -> None:
    proc = subprocess.run(
        ["bash", str(_KERNEL_TOOLS)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "install.sh" in proc.stderr
    assert "command not found" not in proc.stderr
