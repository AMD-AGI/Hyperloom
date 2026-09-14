# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for install.sh's Ray pin resolution on interpreters without a pinned wheel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

KERNEL_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = KERNEL_ROOT / "scripts" / "install.sh"


def _extract_shell_function(name: str) -> str:
    lines = INSTALL_SH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line == f"{name}() {{")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


def _run_resolve_ray_version(
    *, pin_env: str | None, click_env: str | None, suggested: str
) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_shell_function("resolve_ray_version")
    assert fn_src.strip(), "resolve_ray_version() not found in install.sh"
    stub = f"""
warn() {{ echo "WARN: $*" >&2; }}
lowest_ray_version_for_interpreter() {{ printf '%s' '{suggested}'; }}
_RAY_VERSION_WAS_SET="{'x' if pin_env else ''}"
_RAY_CLI_CLICK_MAX_VERSION_WAS_SET="{'x' if click_env else ''}"
RAY_VERSION="{pin_env or '2.44.1'}"
RAY_CLI_CLICK_MAX_VERSION="{click_env or '8.3.0'}"
RAY_INSTALL_SPEC="ray[default]==${{RAY_VERSION}}"
CLICK_INSTALL_SPEC="click<${{RAY_CLI_CLICK_MAX_VERSION}}"
"""
    report = (
        '\necho "RAY_VERSION=${RAY_VERSION}"'
        '\necho "RAY_INSTALL_SPEC=${RAY_INSTALL_SPEC}"'
        '\necho "CLICK_INSTALL_SPEC=${CLICK_INSTALL_SPEC}"\n'
    )
    script = f"set -uo pipefail\n{stub}\n{fn_src}\nresolve_ray_version\n{report}"
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True)


def test_keeps_pin_when_interpreter_has_a_wheel() -> None:
    result = _run_resolve_ray_version(pin_env=None, click_env=None, suggested="")

    assert result.returncode == 0, result.stderr
    assert "RAY_VERSION=2.44.1" in result.stdout
    assert "CLICK_INSTALL_SPEC=click<8.3.0" in result.stdout
    assert "WARN" not in result.stderr


def test_relaxes_pin_when_interpreter_has_no_wheel() -> None:
    result = _run_resolve_ray_version(pin_env=None, click_env=None, suggested="2.55.0")

    assert result.returncode == 0, result.stderr
    assert "RAY_VERSION=2.55.0" in result.stdout
    assert "RAY_INSTALL_SPEC=ray[default]==2.55.0" in result.stdout
    # The click ceiling is specific to 2.44.1's CLI bug; it must not follow the
    # pin forward, or it would hold click below what newer Ray ships against.
    assert "CLICK_INSTALL_SPEC=" in result.stdout
    assert "CLICK_INSTALL_SPEC=click" not in result.stdout
    # Swapping the certified runtime version has to be visible, never silent.
    assert "no wheel for this interpreter" in result.stderr


def test_operator_supplied_pin_is_never_overridden() -> None:
    result = _run_resolve_ray_version(pin_env="2.44.1", click_env=None, suggested="2.55.0")

    assert result.returncode == 0, result.stderr
    assert "RAY_VERSION=2.44.1" in result.stdout
    assert "CLICK_INSTALL_SPEC=click<8.3.0" in result.stdout
    assert "WARN" not in result.stderr


def test_operator_supplied_click_ceiling_survives_pin_relaxation() -> None:
    result = _run_resolve_ray_version(pin_env=None, click_env="8.9.0", suggested="2.55.0")

    assert result.returncode == 0, result.stderr
    assert "RAY_VERSION=2.55.0" in result.stdout
    assert "CLICK_INSTALL_SPEC=click<8.9.0" in result.stdout


def _run_lowest_ray_version(pip_stdout: str, pinned: str = "2.44.1") -> str:
    fn_src = _extract_shell_function("lowest_ray_version_for_interpreter")
    assert fn_src.strip(), "lowest_ray_version_for_interpreter() not found in install.sh"
    fake_pip = f"""
import sys
if "index" in sys.argv:
    sys.stdout.write({pip_stdout!r})
"""
    script = (
        "set -uo pipefail\n"
        f"{fn_src}\n"
        f"lowest_ray_version_for_interpreter '{pinned}'\n"
    )
    # The helper shells out to `python3 -m pip`; shadow pip with a fake module.
    result = subprocess.run(
        ["bash", "-lc", script],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": _fake_pip_dir(fake_pip)},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


_FAKE_PIP_DIRS: list[str] = []


def _fake_pip_dir(source: str) -> str:
    import tempfile

    d = tempfile.mkdtemp()
    pkg = Path(d) / "pip"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(source)
    _FAKE_PIP_DIRS.append(d)
    return d


def test_lowest_version_picks_the_smallest_available_at_or_above_pin() -> None:
    out = _run_lowest_ray_version(
        "ray (2.58.0)\nAvailable versions: 2.58.0, 2.57.0, 2.55.0, 2.40.0\n"
    )
    assert out == "2.55.0"


def test_lowest_version_prints_nothing_when_pin_is_available() -> None:
    out = _run_lowest_ray_version(
        "ray (2.58.0)\nAvailable versions: 2.58.0, 2.44.1, 2.40.0\n"
    )
    assert out == ""


def test_lowest_version_prints_nothing_when_pip_reports_nothing() -> None:
    assert _run_lowest_ray_version("") == ""


def _run_ensure_ray_install_args(click_spec: str) -> str:
    fn_src = _extract_shell_function("ensure_ray")
    assert fn_src.strip(), "ensure_ray() not found in install.sh"
    stub = f"""
log() {{ :; }}
warn() {{ :; }}
resolve_ray_version() {{ :; }}
run() {{ printf 'ARGS:'; for a in "$@"; do printf '[%s]' "$a"; done; printf '\\n'; }}
CHECK_ONLY=0
DRY_RUN=1
RAY_VERSION="2.55.0"
RAY_CLI_CLICK_MAX_VERSION=""
RAY_INSTALL_SPEC="ray[default]==${{RAY_VERSION}}"
CLICK_INSTALL_SPEC="{click_spec}"
"""
    script = f"set -uo pipefail\n{stub}\n{fn_src}\nensure_ray\n"
    result = subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_pip_gets_no_empty_argument_when_click_ceiling_is_dropped() -> None:
    out = _run_ensure_ray_install_args("")

    assert "ARGS:" in out
    assert "[]" not in out, f"empty argument passed to pip: {out!r}"
    assert "[ray[default]==2.55.0]" in out


def test_pip_still_receives_the_click_spec_when_the_ceiling_applies() -> None:
    out = _run_ensure_ray_install_args("click<8.3.0")

    assert "[click<8.3.0]" in out
    assert "[ray[default]==2.55.0]" in out
