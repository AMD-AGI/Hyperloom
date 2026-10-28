# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aiter is imported only when its source root is asked for, so a host with no GPU can import the analysis modules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROBE = (
    "import hyperloom.orchestrator.trace_analysis._kernel_classification as k\n"
    "import hyperloom.measure.profile_traces\n"
    "print('root=' + k._aiter_csrc_root())\n"
)


def _stub_aiter(root: Path, core: str) -> Path:
    package = root / "aiter" / "jit"
    package.mkdir(parents=True)
    (root / "aiter" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text(core, encoding="utf-8")
    return root


def _probe(stub: Path) -> subprocess.CompletedProcess[str]:
    path = [str(stub), *sys.path]
    code = f"import sys\nsys.path[:0] = {path[:1]!r}\n" + _PROBE
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)


def test_a_host_whose_aiter_cannot_start_imports_the_analysis_and_has_no_aiter_root(tmp_path: Path) -> None:
    # aiter.jit.core probes the GPU with rocminfo at import, and raises on a host with no GPU (a hosted app container).
    stub = _stub_aiter(tmp_path, "raise RuntimeError('Get GPU arch from rocminfo failed')\n")
    done = _probe(stub)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "root="
    assert "aiter cannot start on this host" in done.stderr


def test_a_host_with_aiter_resolves_its_source_root(tmp_path: Path) -> None:
    stub = _stub_aiter(tmp_path, "AITER_CSRC_DIR = '/opt/aiter_meta/csrc'\n")
    done = _probe(stub)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "root=/opt/aiter_meta/csrc/"
