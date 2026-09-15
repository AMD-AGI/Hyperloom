# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for install_baremetal.sh's OpenMPI runtime provisioning for vLLM."""

from __future__ import annotations

import subprocess
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_INSTALL_SH = _ASSETS / "install_baremetal.sh"


def _extract_function(name: str) -> str:
    return subprocess.run(
        ["sed", "-n", f"/^{name}()/,/^}}/p", str(_INSTALL_SH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _run_ensure_openmpi_runtime(
    tmp_path: Path, *, ldconfig_has_mpi: bool, uid: int, apt_outcomes: list[bool]
) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_function("ensure_openmpi_runtime")
    assert fn_src.strip(), "ensure_openmpi_runtime() not found in install_baremetal.sh"
    idx_file = tmp_path / "apt_call_idx"
    outcomes = " ".join("0" if ok else "1" for ok in apt_outcomes)
    stub = f"""
log() {{ echo "LOG: $*"; }}
warn() {{ echo "WARN: $*" >&2; }}
ldconfig() {{ [ "$1" = "-p" ] && {"echo 'libmpi.so.40 => /usr/lib/x86_64-linux-gnu/libmpi.so.40'" if ldconfig_has_mpi else "true"}; }}
apt-get() {{
  if [ "$1" = "update" ]; then return 0; fi
  read -r -a _outcomes <<< "{outcomes}"
  _idx=$(cat "{idx_file}" 2>/dev/null || echo 0)
  echo $((_idx + 1)) > "{idx_file}"
  [ "${{_outcomes[$_idx]:-1}}" = "0" ]
}}
id() {{ echo {uid}; }}
"""
    script = f"set -uo pipefail\n{stub}\n{fn_src}\nensure_openmpi_runtime\n"
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True)


def test_ensure_openmpi_runtime_noop_when_already_resolvable(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=True, uid=0, apt_outcomes=[])
    assert result.returncode == 0, result.stderr
    assert "installed" not in result.stdout


def test_ensure_openmpi_runtime_warns_when_not_root(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=False, uid=1000, apt_outcomes=[])
    assert result.returncode == 0, result.stderr
    assert "not running as root" in result.stderr


def test_ensure_openmpi_runtime_tries_t64_name_first(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=False, uid=0, apt_outcomes=[True])
    assert result.returncode == 0, result.stderr
    assert "libopenmpi3t64" in result.stdout


def test_ensure_openmpi_runtime_falls_back_to_older_debian_name(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=False, uid=0, apt_outcomes=[False, True])
    assert result.returncode == 0, result.stderr
    assert "installed libopenmpi3 " in result.stdout


def test_ensure_openmpi_runtime_warns_when_both_names_fail(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=False, uid=0, apt_outcomes=[False, False])
    assert result.returncode == 0, result.stderr
    assert "could not install" in result.stderr
