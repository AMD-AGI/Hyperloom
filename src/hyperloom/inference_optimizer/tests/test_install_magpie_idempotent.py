# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural + static guards for ``ensure_magpie()``."""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
IO_INSTALL = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"
PREFLIGHT = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "cli" / "preflight.py"

MAGPIE_AGENTX_COMMIT = "3642ce66ae46ca4dc125340b3d14a3f4640c369b"

PIP_MARKER = "pip-install-called"


def _extract_ensure_magpie() -> str:
    text = IO_INSTALL.read_text(encoding="utf-8")
    m = re.search(r"^ensure_magpie\(\) \{.*?^\}", text, re.S | re.M)
    assert m, "could not locate ensure_magpie() in install.sh"
    return m.group(0)


def _fake_python(tmp_path: Path, *, capability_ok: bool, installed_root: Path, repair_required: bool = False) -> Path:
    """A stub ``$PYTHON``."""
    marker = tmp_path / PIP_MARKER
    repaired = tmp_path / "pip-force-reinstall-called"
    health_marker = repaired if repair_required else marker
    capability_check = "exit 0" if capability_ok else f'[ -f "{health_marker}" ] && exit 0 || exit 1'
    body = f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  printf '%s\n' "$*" >> "{marker}"
  case " $* " in
    *" --force-reinstall "*) touch "{repaired}" ;;
  esac
  exit 0
fi
if [ "$1" = "-" ]; then
  echo "{installed_root}"
  exit 0
fi
if [ "$1" = "-c" ]; then
  {capability_check}
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
    capability_ok: bool,
    explicit_magpie_path: bool = False,
    repair_required: bool = False,
) -> tuple[str, bool]:
    """Run the extracted ensure_magpie body; return (stdout, pip_called)."""
    installed_root = tmp_path / "site-packages"
    installed_root.mkdir(parents=True)
    magpie_dir = tmp_path / "operator" / "Magpie"
    fake_py = _fake_python(
        tmp_path, capability_ok=capability_ok, installed_root=installed_root, repair_required=repair_required
    )
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
PIP_EXTRA=("--disable-pip-version-check")

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


def test_skip_reinstall_when_native_agentx_capability_is_available(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, capability_ok=True)
    assert not pip_called, f"reinstall should have been skipped:\n{out}"
    assert "Magpie native AgentX capability already available; skipping pip install" in out
    assert "MAGPIE_PATH resolved from installed package" in out


def test_install_when_native_agentx_capability_is_missing(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, capability_ok=False)
    assert pip_called, f"reinstall should have run on capability miss:\n{out}"
    assert "Magpie installed OK from magpie-eval @ git+https://example.invalid/Magpie.git@deadbeef" in out


def test_repairs_an_installed_commit_whose_execution_tree_was_patched(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, capability_ok=False, repair_required=True)

    assert pip_called
    calls = (tmp_path / PIP_MARKER).read_text().splitlines()
    assert len(calls) == 2
    assert "--force-reinstall" not in calls[0]
    assert "--force-reinstall --no-deps" in calls[1]
    assert "Magpie installed OK" in out


def test_preserves_explicit_magpie_path(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, capability_ok=True, explicit_magpie_path=True)
    assert not pip_called
    assert "MAGPIE_PATH override preserved" in out


# Static guard: the idempotent skip must stay wired in.
def test_io_install_magpie_reinstall_is_idempotent_guarded() -> None:
    body = _extract_ensure_magpie()
    assert "AgentXConfig" in body
    assert "run-eval" in body
    assert "_MAGPIE_SOURCE_IDENTITY_CODE" in body
    assert 'scope["_validate_magpie_execution_tree"]' in body
    assert 'git","-C",str(root),"rev-parse","HEAD"' not in body
    assert "Magpie native AgentX capability already available; skipping pip install" in body
    assert "MAGPIE_PACKAGE_SPEC" in body
    assert "pip install" in body, "reinstall path must still exist for the miss case"


def test_default_magpie_pin_has_native_agentx_and_is_consistent() -> None:
    install_text = IO_INSTALL.read_text(encoding="utf-8")
    preflight_text = PREFLIGHT.read_text(encoding="utf-8")

    assert f'MAGPIE_REF="${{MAGPIE_REF:-{MAGPIE_AGENTX_COMMIT}}}"' in install_text
    assert f'_MAGPIE_REF_DEFAULT = "{MAGPIE_AGENTX_COMMIT}"' in preflight_text
    assert 'os.environ.get("MAGPIE_REF") or _MAGPIE_REF_DEFAULT' in preflight_text
