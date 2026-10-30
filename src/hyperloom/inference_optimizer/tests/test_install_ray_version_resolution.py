# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for install.sh's Ray pin resolution on interpreters without a pinned wheel."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[1] / "assets" / "install_kernel_tools.sh"
_CLICK_RC = int(re.search(r"^RAY_CLI_PROBE_CLICK_RC=(\d+)$", INSTALL_SH.read_text(encoding="utf-8"), re.M).group(1))


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
    probe = _extract_shell_function("ray_cli_probe")

    assert "from ray.scripts.scripts import cli" in probe
    assert "make_context(" in probe
    assert "click.UsageError" in probe
    assert "param.opts" not in probe


def _fake_ray_site(tmp_path: Path, *, version: str, scripts_source: str) -> Path:
    site = tmp_path / "site"
    (site / "ray" / "scripts").mkdir(parents=True)
    (site / "ray" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (site / "ray" / "scripts" / "__init__.py").write_text("")
    (site / "ray" / "scripts" / "scripts.py").write_text(scripts_source)
    return site


_SENTINEL_CLI = "raise ValueError('<object object at 0x7f> is not a valid Sentinel')\n"
_USAGE_CLI = (
    "import click\n\n@click.group()\ndef cli():\n    pass\n\n"
    "@cli.command()\n@click.option('--other')\ndef start(**kw):\n    pass\n"
)


def _run_real_probe(site: Path, *, ray_version: str) -> subprocess.CompletedProcess[str]:
    constants = "\n".join(
        line
        for line in INSTALL_SH.read_text(encoding="utf-8").splitlines()
        if line.startswith(("RAY_CLI_PROBE_CLICK_RC=", "RAY_CLICK_HOLD_VERSION="))
    )
    script = f"{constants}\n{_extract_shell_function('ray_cli_probe')}\nray_cli_probe\n"
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "PYTHONPATH": str(site)}
    env.update(RAY_VERSION=ray_version, RAY_START_ARGV="--head --port=6379")
    return subprocess.run(["bash", "-c", script], check=False, capture_output=True, text=True, env=env)


def test_the_probe_gives_the_sentinel_import_failure_its_own_exit_code(tmp_path: Path) -> None:
    site = _fake_ray_site(tmp_path, version="2.44.1", scripts_source=_SENTINEL_CLI)

    done = _run_real_probe(site, ray_version="2.44.1")

    assert done.returncode == _CLICK_RC, done.stderr
    assert "is not a valid Sentinel" in done.stderr


def test_the_probe_keeps_exit_1_for_every_other_failure(tmp_path: Path) -> None:
    sentinel_other_ray = _fake_ray_site(tmp_path / "a", version="2.55.0", scripts_source=_SENTINEL_CLI)
    usage = _fake_ray_site(tmp_path / "b", version="2.44.1", scripts_source=_USAGE_CLI)
    plain_import_error = _fake_ray_site(tmp_path / "c", version="2.44.1", scripts_source="raise ImportError('boom')\n")
    mismatch = _fake_ray_site(tmp_path / "d", version="2.44.1", scripts_source=_SENTINEL_CLI)

    assert _run_real_probe(sentinel_other_ray, ray_version="2.55.0").returncode == 1
    assert _run_real_probe(usage, ray_version="2.44.1").returncode == 1
    assert _run_real_probe(plain_import_error, ray_version="2.44.1").returncode == 1
    assert _run_real_probe(mismatch, ray_version="2.45.0").returncode == 1


def _run_ensure_ray_with_probe(
    probe_results: list[int], *, check_only: int = 0, ray_version: str = "2.44.1"
) -> subprocess.CompletedProcess[str]:
    """ensure_ray with pip recorded and the CLI probe answering ``probe_results`` in turn (0 = the CLI starts)."""
    fn_src = _extract_shell_function("ensure_ray")
    constants = "\n".join(
        line
        for line in INSTALL_SH.read_text(encoding="utf-8").splitlines()
        if line.startswith(("RAY_CLI_PROBE_CLICK_RC=", "RAY_CLICK_HOLD_VERSION=", "RAY_CLICK_HOLD_SPEC="))
    )
    results = " ".join(str(r) for r in probe_results)
    stub = f"""
{constants}
set -e
log() {{ :; }}
warn() {{ printf 'WARN:%s\\n' "$*"; }}
resolve_ray_version() {{ :; }}
run() {{ printf 'ARGS:'; for a in "$@"; do printf '[%s]' "$a"; done; printf '\\n'; }}
PROBE_RESULTS=({results})
ray_cli_probe() {{ printf 'PROBE\\n'; local r="${{PROBE_RESULTS[0]}}"; PROBE_RESULTS=("${{PROBE_RESULTS[@]:1}}"); return "$r"; }}
CHECK_ONLY={check_only}
DRY_RUN=0
RAY_VERSION="{ray_version}"
RAY_INSTALL_SPEC="ray[default]==${{RAY_VERSION}}"
"""
    script = f"{fn_src}\n{stub}\nensure_ray\n"
    return subprocess.run(["bash", "-c", script], check=False, capture_output=True, text=True)


def test_a_cli_that_starts_keeps_the_click_ray_resolved() -> None:
    done = _run_ensure_ray_with_probe([0])

    assert done.returncode == 0, done.stderr
    assert done.stdout.count("PROBE") == 1
    assert "click" not in done.stdout


def test_the_sentinel_failure_holds_click_below_8_3_once_and_probes_again() -> None:
    done = _run_ensure_ray_with_probe([_CLICK_RC, 0])

    assert done.returncode == 0, done.stderr
    assert done.stdout.count("[click<8.3]") == 1
    assert done.stdout.count("PROBE") == 2
    assert "WARN:" in done.stdout


def test_a_sentinel_failure_the_hold_does_not_fix_fails_the_install() -> None:
    done = _run_ensure_ray_with_probe([_CLICK_RC, _CLICK_RC])

    assert done.returncode != 0
    assert done.stdout.count("[click<8.3]") == 1


def test_any_other_probe_failure_ends_the_install_without_touching_click() -> None:
    # UsageError / version mismatch / other import error all exit 1 from the probe.
    done = _run_ensure_ray_with_probe([1])

    assert done.returncode != 0
    assert done.stdout.count("PROBE") == 1
    assert "click" not in done.stdout


def test_a_sentinel_failure_under_another_ray_version_leaves_click_alone() -> None:
    done = _run_ensure_ray_with_probe([_CLICK_RC], ray_version="2.55.0")

    assert done.returncode != 0
    assert "click" not in done.stdout


def test_check_only_reports_a_sentinel_failure_without_installing_anything() -> None:
    done = _run_ensure_ray_with_probe([_CLICK_RC], check_only=1)

    assert done.returncode != 0
    assert "ARGS:" not in done.stdout


def test_the_click_hold_has_no_environment_knob() -> None:
    assert "RAY_CLICK_FALLBACK_SPEC" not in INSTALL_SH.read_text(encoding="utf-8")
