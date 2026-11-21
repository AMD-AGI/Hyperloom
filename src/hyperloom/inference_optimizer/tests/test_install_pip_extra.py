# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""install.sh gives ``--break-system-packages`` to the ``pip install`` commands of a system interpreter whose
``pip install --help`` lists it, and to nothing else.

``set_pip_extra`` runs as install.sh defines it, extracted from the real file, against a stand-in Python whose pip
answers ``--help`` the way a given pip release does.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4]
INSTALL_SH = SOURCE_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"
END = "REACHED_END"
FLAG = "--break-system-packages"

# FAKE_VENV=1 is a virtual environment. FAKE_PIP: "old" is a pip before 23.0.1, whose `install --help` does not list
# the option and which, as field images report, still exits 0 on `install --break-system-packages --help`; "23.0" and
# "modern" list it. Every pip call is logged.
_PYTHON = """#!/usr/bin/env bash
if [ "$1" = "-" ]; then
  cat >/dev/null
  exit "$FAKE_VENV"
fi
printf '%s\\n' "$*" >>"$PIPLOG"
if [ "$3" = "--version" ]; then
  echo "pip $FAKE_PIP from /usr/lib/python3/dist-packages/pip (python 3.10)"
  exit 0
fi
if [ "$3" = "install" ] && [[ " $* " == *" --help "* ]]; then
  echo "Install Options:"
  case "$FAKE_PIP" in
    23.0) echo "  --break-system-packages     Allow pip to modify an EXTERNALLY-MANAGED Python installation" ;;
    modern) echo "  --break-system-packages   Allow pip to modify an EXTERNALLY-MANAGED Python installation" ;;
  esac
  echo "  --force-reinstall           Reinstall all packages even if they are already up-to-date."
  exit 0
fi
exit 0
"""


def _function(name: str) -> str:
    found = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", INSTALL_SH.read_text(encoding="utf-8"), re.S | re.M)
    assert found, f"could not locate {name}() in install.sh"
    return found.group(0)


def _run(tmp_path: Path, *, fake_pip: str, venv: bool) -> tuple[int, str, list[str]]:
    stub = tmp_path / "python"
    stub.write_text(_PYTHON, encoding="utf-8")
    stub.chmod(0o755)
    piplog = tmp_path / "pip.log"
    piplog.write_text("", encoding="utf-8")
    script = "\n\n".join(
        [
            "set -euo pipefail",
            'log()  { echo "[log] $*"; }',
            'warn() { echo "[warn] $*" >&2; }',
            'die()  { echo "[die] $*" >&2; exit 1; }',
            f'PYTHON="{stub}"\nPIP_EXTRA=()',
            _function("set_pip_extra"),
            f'set_pip_extra\necho "PIP_EXTRA=${{PIP_EXTRA[*]-}}"\necho {END}',
        ]
    )
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PIPLOG": str(piplog),
        "FAKE_PIP": fake_pip,
        "FAKE_VENV": "1" if venv else "0",
    }
    proc = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, env=env, text=True, capture_output=True, check=False, timeout=60
    )
    return proc.returncode, proc.stdout + proc.stderr, piplog.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("fake_pip", ["23.0", "modern"], ids=["pip-23.0.x", "modern-pip"])
def test_a_system_python_whose_pip_install_lists_the_option_gets_it(tmp_path: Path, fake_pip: str) -> None:
    code, out, _ = _run(tmp_path, fake_pip=fake_pip, venv=False)
    assert code == 0, out
    assert f"PIP_EXTRA={FLAG}\n{END}" in out, out


def test_a_system_python_whose_pip_install_does_not_list_the_option_never_gets_it(tmp_path: Path) -> None:
    code, out, calls = _run(tmp_path, fake_pip="old", venv=False)
    assert code == 1, out
    assert END not in out and f"PIP_EXTRA={FLAG}" not in out, out
    assert "[die] refusing to run pip without a working --break-system-packages" in out, out
    assert all(FLAG not in call.split() for call in calls), calls


def test_a_virtual_environment_never_gets_the_option_and_runs_no_pip(tmp_path: Path) -> None:
    code, out, calls = _run(tmp_path, fake_pip="modern", venv=True)
    assert code == 0, out
    assert f"PIP_EXTRA=\n{END}" in out, out
    assert calls == []


def _commands() -> list[str]:
    """install.sh's lines with backslash continuations joined, so one entry is one whole command."""
    return re.sub(r"\\\n", " ", INSTALL_SH.read_text(encoding="utf-8")).splitlines()


def test_pip_extra_goes_to_pip_install_commands_only() -> None:
    using = [line for line in _commands() if '"${PIP_EXTRA[@]}"' in line]
    assert using, "no command in install.sh uses PIP_EXTRA"
    assert [line for line in using if not re.search(r"-m pip install\b", line)] == []
