# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""install.sh's torch/GPU gate: what it concludes when the probe cannot speak for itself."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_INSTALL_SH = Path(__file__).resolve().parents[1] / "assets" / "install.sh"

pytestmark = pytest.mark.skipif(not Path("/usr/bin/python3").exists(), reason="needs a system python3 to stub")


def _gate_source() -> str:
    body = subprocess.run(
        ["sed", "-n", "/^ensure_torch_compatible_with_gpu()/,/^}/p", str(_INSTALL_SH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert body.strip(), "ensure_torch_compatible_with_gpu() not found in install.sh"
    return body


def _run_gate(tmp_path: Path, *, torch_body: str) -> subprocess.CompletedProcess[str]:
    """Run the gate against a stub interpreter whose ``import torch`` does ``torch_body``."""
    stub_pkg = tmp_path / "pkgs"
    stub_pkg.mkdir(exist_ok=True)
    (stub_pkg / "torch.py").write_text(torch_body, encoding="utf-8")

    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nexec /usr/bin/python3 "$@"\n', encoding="utf-8")
    python.chmod(0o755)

    smi = tmp_path / "rocm-smi"
    smi.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    smi.chmod(0o755)

    script = f"""
set -uo pipefail
warn() {{ echo "WARN: $*"; }}
die() {{ echo "DIE: $*"; exit 1; }}
log() {{ echo "LOG: $*"; }}
PYTHON={python}
{_gate_source()}
ensure_torch_compatible_with_gpu
echo "GATE_RETURNED"
"""
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "PYTHONPATH": str(stub_pkg),
        },
    )


_EXITS_ON_IMPORT = (
    "import sys\n"
    "print('/root/libamd_smi.so: cannot open shared object file: No such file or directory')\n"
    "print('Unable to find libamd_smi.so library try installing amd-smi-lib')\n"
    "sys.exit(1)\n"
)

_DIES_WITHOUT_VERDICT = "import os\nprint('Fatal Python error: Segmentation fault', flush=True)\nos._exit(139)\n"

_ROCM_BUILD = "__version__ = '2.11.0+rocm10.0.0'\n\n\nclass version:\n    hip = '7.15.26333'\n    cuda = None\n"


def test_a_torch_that_exits_during_import_is_not_called_a_cuda_build(tmp_path):
    """``import torch`` can pull in a module that calls ``sys.exit`` instead of raising."""
    result = _run_gate(tmp_path, torch_body=_EXITS_ON_IMPORT)

    assert "is CUDA-built on a ROCm pod" not in result.stdout, result.stdout
    assert "NOT importable" in result.stdout, result.stdout
    assert "SystemExit" in result.stdout, result.stdout


def test_a_probe_without_a_verdict_leaves_the_match_undetermined(tmp_path):
    result = _run_gate(tmp_path, torch_body=_DIES_WITHOUT_VERDICT)

    assert "is CUDA-built on a ROCm pod" not in result.stdout, result.stdout
    assert "undetermined" in result.stdout, result.stdout
    assert "Segmentation fault" in result.stdout, result.stdout
    assert "GATE_RETURNED" in result.stdout, result.stdout


def test_a_rocm_torch_passes_the_gate(tmp_path):
    result = _run_gate(tmp_path, torch_body=_ROCM_BUILD)

    assert "GATE_RETURNED" in result.stdout, result.stdout
    assert "DIE:" not in result.stdout, result.stdout
