# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Setup reports product identity without inferring a board from shared ISA."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

_INSTALLER = Path(__file__).resolve().parents[1] / "assets" / "install_baremetal.sh"


@pytest.mark.parametrize(
    ("smi_label", "torch_label", "arch", "expected"),
    [
        ("AMD Instinct MI300A", "", "gfx942", "MI300A"),
        ("AMD Instinct MI325X", "", "gfx942", "MI325X"),
        ("N/A", "AMD Instinct MI300A", "gfx942", "AMD Instinct MI300A"),
        ("N/A", "AMD Instinct MI325X", "gfx942", "AMD Instinct MI325X"),
        ("", "AMD Instinct MI300A", "gfx942", "AMD Instinct MI300A"),
        ("N/A", "", "gfx942", "gfx942"),
        ("N/A", "", "gfx950", "gfx950"),
        ("N/A", "", "", "unknown"),
    ],
)
def test_setup_gpu_label(tmp_path, smi_label, torch_label, arch, expected):
    function = re.search(r"^detect_gpu_label\(\) \{.*?^\}", _INSTALLER.read_text(), re.M | re.S)
    assert function is not None
    smi = tmp_path / "rocm-smi"
    smi.write_text(f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(smi_label)}\n", newline="\n")
    smi.chmod(0o755)
    py = tmp_path / "python-probe"
    py.write_text(f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(torch_label)}\n", newline="\n")
    py.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    script = (
        f"set -euo pipefail\n{function.group(0)}\n"
        f"resolve_python() {{ printf '%s\\n' {shlex.quote(str(py))}; }}\n"
        f"detect_gpu_label {shlex.quote(arch)}\n"
    )
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == expected
