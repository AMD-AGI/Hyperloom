# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural + static guards for ``ensure_magpie()``.

Magpie is installed from a pinned pip package spec rather than a local editable
checkout. The installer should skip pip when ``import Magpie`` already works,
install the configured package spec when it does not, and resolve ``MAGPIE_PATH``
to the installed package root unless the operator supplied an explicit override.
"""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
IO_INSTALL = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"

PIP_MARKER = "pip-install-called"


def _extract_ensure_magpie() -> str:
    text = IO_INSTALL.read_text(encoding="utf-8")
    m = re.search(r"^ensure_magpie\(\) \{.*?^\}", text, re.S | re.M)
    assert m, "could not locate ensure_magpie() in install.sh"
    return m.group(0)


def _fake_python(tmp_path: Path, *, import_ok: bool, installed_root: Path) -> Path:
    """A stub ``$PYTHON``.

    * ``-c 'import Magpie'``: exits according to ``import_ok`` unless pip has
      already been called.
    * ``-m pip install ...``: touches PIP_MARKER and exits 0.
    * ``-`` (stdin script used to resolve the import root): prints
      ``installed_root``.
    """
    marker = tmp_path / PIP_MARKER
    import_check = "exit 0" if import_ok else f'[ -f "{marker}" ] && exit 0 || exit 1'
    body = f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  touch "{marker}"
  exit 0
fi
if [ "$1" = "-" ]; then
  echo "{installed_root}"
  exit 0
fi
if [ "$1" = "-c" ]; then
  {import_check}
fi
exit 0
"""
    py = tmp_path / "fake_python.sh"
    py.write_text(body, encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return py


def _run_ensure_magpie(
    tmp_path: Path,
    *,
    import_ok: bool,
    explicit_magpie_path: bool = False,
) -> tuple[str, bool]:
    """Run the extracted ensure_magpie body; return (stdout, pip_called)."""
    installed_root = tmp_path / "site-packages"
    installed_root.mkdir(parents=True)
    magpie_dir = tmp_path / "operator" / "Magpie"
    fake_py = _fake_python(tmp_path, import_ok=import_ok, installed_root=installed_root)
    magpie_path_line = (
        f'MAGPIE_PATH="{magpie_dir}"\nMAGPIE_PATH_EXPLICIT=1'
        if explicit_magpie_path
        else f'MAGPIE_PATH="{magpie_dir}"\nMAGPIE_PATH_EXPLICIT=0'
    )

    harness = f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
CHECK_ONLY=0
DRY_RUN=0
MAGPIE_REPO="https://example.invalid/Magpie.git"
MAGPIE_REF="deadbeef"
MAGPIE_PACKAGE_SPEC="magpie-eval @ git+https://example.invalid/Magpie.git@deadbeef"
{magpie_path_line}
PYTHON="{fake_py}"
PIP_EXTRA=()

{_extract_ensure_magpie()}

ensure_magpie
"""
    script = tmp_path / "harness.sh"
    script.write_text(harness, encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, f"ensure_magpie harness failed:\n{proc.stdout}"
    pip_called = (tmp_path / PIP_MARKER).exists()
    return proc.stdout, pip_called


def test_skip_reinstall_when_already_installed_from_magpie_dir(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, import_ok=True)
    assert not pip_called, f"reinstall should have been skipped:\n{out}"
    assert "Magpie already importable; skipping pip install" in out
    assert "MAGPIE_PATH resolved from installed package" in out


def test_install_when_import_fails(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, import_ok=False)
    assert pip_called, f"reinstall should have run on import failure:\n{out}"
    assert "Magpie installed OK from magpie-eval @ git+https://example.invalid/Magpie.git@deadbeef" in out


def test_preserves_explicit_magpie_path(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, import_ok=True, explicit_magpie_path=True)
    assert not pip_called
    assert "MAGPIE_PATH override preserved" in out


# Static guard: the idempotent skip must stay wired in.
def test_io_install_magpie_reinstall_is_idempotent_guarded() -> None:
    body = _extract_ensure_magpie()
    assert "Magpie already importable; skipping pip install" in body
    assert "MAGPIE_PACKAGE_SPEC" in body
    assert "pip install" in body, "reinstall path must still exist for the miss case"
