# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for install.sh's Ray pin resolution on interpreters without a pinned wheel."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[1] / "assets" / "install_kernel_tools.sh"


def _extract_shell_function(name: str) -> str:
    lines = INSTALL_SH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line == f"{name}() {{")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


def _run_resolve_ray_version(*, pin_env: str | None, suggested: str) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_shell_function("resolve_ray_version")
    assert fn_src.strip(), "resolve_ray_version() not found in install.sh"
    stub = f"""
warn() {{ echo "WARN: $*" >&2; }}
lowest_ray_version_for_interpreter() {{ printf '%s' '{suggested}'; }}
_RAY_VERSION_WAS_SET="{"x" if pin_env else ""}"
RAY_VERSION="{pin_env or "2.44.1"}"
RAY_INSTALL_SPEC="ray[default]==${{RAY_VERSION}}"
"""
    report = '\necho "RAY_VERSION=${RAY_VERSION}"\necho "RAY_INSTALL_SPEC=${RAY_INSTALL_SPEC}"\n'
    script = f"set -uo pipefail\n{stub}\n{fn_src}\nresolve_ray_version\n{report}"
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True)


def test_keeps_pin_when_interpreter_has_a_wheel() -> None:
    result = _run_resolve_ray_version(pin_env=None, suggested="")

    assert result.returncode == 0, result.stderr
    assert "RAY_VERSION=2.44.1" in result.stdout
    assert "RAY_INSTALL_SPEC=ray[default]==2.44.1" in result.stdout
    assert "WARN" not in result.stderr


def test_relaxes_pin_when_interpreter_has_no_wheel() -> None:
    result = _run_resolve_ray_version(pin_env=None, suggested="2.55.0")

    assert result.returncode == 0, result.stderr
    assert "RAY_VERSION=2.55.0" in result.stdout
    assert "RAY_INSTALL_SPEC=ray[default]==2.55.0" in result.stdout
    # Swapping the certified runtime version has to be visible, never silent.
    assert "no wheel for this interpreter" in result.stderr


def test_operator_supplied_pin_is_never_overridden() -> None:
    result = _run_resolve_ray_version(pin_env="2.44.1", suggested="2.55.0")

    assert result.returncode == 0, result.stderr
    assert "RAY_VERSION=2.44.1" in result.stdout
    assert "WARN" not in result.stderr


def _run_lowest_ray_version(pip_stdout: str, pinned: str = "2.44.1") -> str:
    fn_src = _extract_shell_function("lowest_ray_version_for_interpreter")
    assert fn_src.strip(), "lowest_ray_version_for_interpreter() not found in install.sh"
    fake_pip = f"""
import sys
if "index" in sys.argv:
    sys.stdout.write({pip_stdout!r})
"""
    script = f"set -uo pipefail\n{fn_src}\nlowest_ray_version_for_interpreter '{pinned}'\n"
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
    out = _run_lowest_ray_version("ray (2.58.0)\nAvailable versions: 2.58.0, 2.57.0, 2.55.0, 2.40.0\n")
    assert out == "2.55.0"


def test_lowest_version_prints_nothing_when_pin_is_available() -> None:
    out = _run_lowest_ray_version("ray (2.58.0)\nAvailable versions: 2.58.0, 2.44.1, 2.40.0\n")
    assert out == ""


def test_lowest_version_prints_nothing_when_pip_reports_nothing() -> None:
    assert _run_lowest_ray_version("") == ""


def _run_ensure_ray_install_args() -> str:
    fn_src = _extract_shell_function("ensure_ray")
    assert fn_src.strip(), "ensure_ray() not found in install.sh"
    stub = """
log() { :; }
warn() { :; }
resolve_ray_version() { :; }
run() { printf 'ARGS:'; for a in "$@"; do printf '[%s]' "$a"; done; printf '\\n'; }
CHECK_ONLY=0
DRY_RUN=1
RAY_VERSION="2.55.0"
RAY_INSTALL_SPEC="ray[default]==${RAY_VERSION}"
RAY_START_ARGV="--head"
"""
    script = f"set -uo pipefail\n{stub}\n{fn_src}\nensure_ray\n"
    result = subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_pip_installs_ray_alone() -> None:
    # Nothing but Ray is pinned here: the CLI is certified by what it parses, so
    # the installer never pins or downgrades the click Ray resolves for itself.
    out = _run_ensure_ray_install_args()

    assert "ARGS:" in out
    assert "[]" not in out, f"empty argument passed to pip: {out!r}"
    assert "[ray[default]==2.55.0]" in out
    assert "click" not in out


def _declared_start_argv() -> list[str]:
    """RAY_START_ARGV as the shell expands it, hence as ensure_ray's probe sees it."""
    source = INSTALL_SH.read_text(encoding="utf-8")
    assignment = next((line for line in source.splitlines() if line.startswith("RAY_START_ARGV=")), "")
    assert assignment, "RAY_START_ARGV not found in install.sh"
    return shlex.split(shlex.split(assignment)[0].split("=", 1)[1])


def _head_start_options(body: str) -> set[str]:
    """The options ensure_ray_started hands ray start, the probed ports included."""
    invocation = re.search(r"_ray_start_detached (?:.*\\\n)*.*\n", body)
    assert invocation, "ensure_ray_started no longer starts the head through _ray_start_detached"
    ports = "\n".join(line for line in body.splitlines() if "ray_port_args+=(" in line)
    return set(re.findall(r"--[a-z][a-z-]*", invocation.group(0) + "\n" + ports))


def test_certified_start_argv_carries_the_options_the_head_is_started_with() -> None:
    """The argv ensure_ray parses holds exactly the options it later passes to ray start."""
    declared = _declared_start_argv()
    passed = _head_start_options(_extract_shell_function("ensure_ray_started"))

    assert declared, "RAY_START_ARGV is empty"
    certified = {argument.split("=", 1)[0] for argument in declared}
    assert certified == passed, f"certified {sorted(certified)} but starts the head with {sorted(passed)}"


def test_certified_start_argv_carries_a_value_of_each_option_s_kind() -> None:
    """Names alone never reach Ray's parser: the probe has to hand it real values."""
    declared = dict(
        argument.split("=", 1) if "=" in argument else (argument, "") for argument in _declared_start_argv()
    )

    assert declared["--port"].isdigit()
    assert declared["--ray-client-server-port"].isdigit()
    assert declared["--num-gpus"].isdigit()
    assert declared["--include-dashboard"] in ("true", "false")
    assert json.loads(declared["--resources"]) == {"serving_slot": 1}


def test_ensure_ray_certifies_the_cli_through_ray_s_own_parser() -> None:
    """Resolve the click group Ray publishes and let it parse, rather than read option names."""
    probe = _extract_shell_function("ensure_ray")

    assert "from ray.scripts.scripts import cli" in probe
    assert "make_context(" in probe
    assert "click.UsageError" in probe
    assert "param.opts" not in probe
