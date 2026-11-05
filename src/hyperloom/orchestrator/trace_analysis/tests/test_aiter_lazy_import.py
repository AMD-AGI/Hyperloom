# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aiter's source root is found where ``aiter/jit/core.py`` finds it, without importing aiter.

Importing ``aiter.jit.core`` probes the GPU, opens every card the process sees and raises on a host with none, so the
analysis modules never import it: a host with no GPU can use them, and a toolbox analysing a trace holds no card.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Any import of aiter's JIT core fails this probe, so a root it prints was found without one.
_PROBE = (
    "import hyperloom.orchestrator.trace_analysis._kernel_classification as k\n"
    "import hyperloom.orchestrator.trace_analysis.tracelens_analysis\n"
    "print('root=' + k._aiter_csrc_root())\n"
    "import sys\n"
    "assert 'aiter' not in sys.modules, 'aiter was imported'\n"
)


def _aiter(root: Path, *, install_mode: str | None) -> Path:
    """An aiter package below ``root`` whose JIT core raises as it does on a host with no GPU."""
    package = root / "aiter"
    (package / "jit").mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('No gfx arch found in rocminfo output.')\n")
    (package / "jit" / "__init__.py").write_text("")
    (package / "jit" / "core.py").write_text("raise RuntimeError('No gfx arch found in rocminfo output.')\n")
    if install_mode is not None:
        (package / "install_mode").write_text(install_mode)  # aiter/jit/core.py reads aiter/install_mode
    return root


def _probe(site: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    code = f"import sys\nsys.path[:0] = [{str(site)!r}]\n" + _PROBE
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False, env=env
    )


def _root(done: subprocess.CompletedProcess[str]) -> str:
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1].removeprefix("root=")


def test_a_develop_install_keeps_its_sources_beside_the_package(tmp_path: Path) -> None:
    site = _aiter(tmp_path / "site", install_mode="develop")
    (site / "csrc").mkdir()
    assert _root(_probe(site)) == f"{(site / 'csrc').resolve().as_posix()}/"


def test_a_wheel_install_keeps_its_sources_in_aiter_meta(tmp_path: Path) -> None:
    site = _aiter(tmp_path / "site", install_mode="wheel")
    (site / "aiter_meta" / "csrc").mkdir(parents=True)
    assert _root(_probe(site)) == f"{(site / 'aiter_meta' / 'csrc').resolve().as_posix()}/"


def test_the_meta_directory_variable_wins_as_it_does_for_aiter(tmp_path: Path) -> None:
    site = _aiter(tmp_path / "site", install_mode="develop")
    (site / "csrc").mkdir()
    elsewhere = tmp_path / "farm-meta"
    (elsewhere / "csrc").mkdir(parents=True)
    env = {**os.environ, "AITER_META_DIR": str(elsewhere)}
    assert _root(_probe(site, env)) == f"{(elsewhere / 'csrc').resolve().as_posix()}/"


def test_an_aiter_without_sources_has_no_root(tmp_path: Path) -> None:
    site = _aiter(tmp_path / "site", install_mode="develop")
    assert _root(_probe(site)) == ""
