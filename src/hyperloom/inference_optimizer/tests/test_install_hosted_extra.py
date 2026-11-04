# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The packaged install of a hosted run installs the ``hosted`` extra, named by the installed wheel's own metadata.

A hosted image installs the wheel with ``--no-deps`` and runs the packaged install.sh, whose packaged branch installs
only ``[llm,forge]``. ``ensure_hosted_extra`` adds the ``hosted`` extra's requirements (signing, the sandbox SDK, the
result archive), read off the installed distribution's Requires-Dist so ``pyproject.toml`` stays their one source.

The functions run as install.sh defines them, extracted from the real file. The interpreter is the real one started
with ``-S``, so the only distributions it sees are the ones a test lays out in the ``pip install --target`` root
(``REPO_ROOT``): the wheel's real metadata and whatever a test says is already installed. ``pip`` is a stub that
records what it was asked for.
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import packaging
import pytest
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10
    from pip._vendor import tomli as tomllib

SOURCE_ROOT = Path(__file__).resolve().parents[4]
INSTALL_SH = SOURCE_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"
DIST = "hyperloom-inference_optimizer"
HOSTED_EXTRA = "hosted"
END = "REACHED_END"

_PYTHON = """#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  printf '%s\\n' "$*" >> "$PIPLOG"
  exit "${PIP_RC:-0}"
fi
exec "$REAL_PYTHON" -S "$@"
"""


def _function(name: str) -> str:
    found = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", INSTALL_SH.read_text(encoding="utf-8"), re.S | re.M)
    assert found, f"could not locate {name}() in install.sh"
    return found.group(0)


def _hosted_requirements() -> dict[str, Requirement]:
    """The ``hosted`` extra as pyproject.toml declares it, by project name."""
    declared = tomllib.loads((SOURCE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {(req := Requirement(spec)).name: req for spec in declared["project"]["optional-dependencies"]["hosted"]}


def _dist_info(root: Path, name: str, version: str, requires: Sequence[str] = ()) -> None:
    info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
        *(f"Requires-Dist: {r}" for r in requires),
    ]
    (info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _packaging_only(tmp_path: Path) -> Path:
    """A directory whose one importable package is ``packaging``, which the requirement parsing needs; it holds no
    distribution's metadata, so it changes nothing about what counts as installed."""
    libs = tmp_path / "libs"
    libs.mkdir()
    (libs / "packaging").symlink_to(Path(packaging.__file__).parent, target_is_directory=True)
    return libs


def _run(
    tmp_path: Path,
    *,
    installed: dict[str, str] | None = None,
    wheel: bool = True,
    packaged: int = 1,
    state_dir: str | None = "/state",
    pip_rc: int = 0,
    python: str | None = None,
    call: str = "ensure_hosted_extra",
) -> tuple[int, str, list[str]]:
    """Run ``call`` (``ensure_hosted_extra``) with install.sh's functions; returns its exit code, output and the pip command lines."""
    root = tmp_path / "target-root"
    root.mkdir(parents=True)
    if wheel:
        _dist_info(root, DIST, "0.0.0", importlib.metadata.metadata(DIST).get_all("Requires-Dist") or [])
    for name, version in (installed or {}).items():
        _dist_info(root, name, version)
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
            "CHECK_ONLY=0\nDRY_RUN=0\nPIP_EXTRA=()",
            f"HYPERLOOM_PACKAGED_INSTALL={packaged}",
            f'REPO_ROOT="{root}"',
            f'PYTHON="{python or stub}"',
            _function("_installed_extra_requirements"),
            _function("ensure_hosted_extra"),
            f"{call}\necho {END}",
        ]
    )
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "REAL_PYTHON": sys.executable,
        "PIPLOG": str(piplog),
        "PIP_RC": str(pip_rc),
        "PYTHONPATH": str(_packaging_only(tmp_path)),
    }
    if state_dir is not None:
        env["HYPERLOOM_STATE_DIR"] = state_dir
    proc = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, env=env, text=True, capture_output=True, check=False, timeout=120
    )
    return proc.returncode, proc.stdout + proc.stderr, piplog.read_text(encoding="utf-8").splitlines()


def _requested(pip_calls: list[str]) -> dict[str, Requirement]:
    (call,) = pip_calls
    words = call.split()
    assert words[:4] == ["-m", "pip", "install", "--quiet"], call
    return {(req := Requirement(word)).name: req for word in words[4:]}


def _same(got: dict[str, Requirement], expected: dict[str, Requirement]) -> None:
    assert {name: str(req.specifier) for name, req in got.items()} == {
        name: str(req.specifier) for name, req in expected.items()
    }


def test_a_hosted_packaged_install_requests_exactly_the_hosted_extra_of_the_wheels_metadata(tmp_path: Path) -> None:
    rc, out, pip_calls = _run(tmp_path)
    assert rc == 0 and END in out, out
    _same(_requested(pip_calls), _hosted_requirements())
    assert set(_hosted_requirements()) == {"cryptography", "e2b", "boto3"}, "the extra this guards"


def test_requirements_already_installed_are_not_requested_and_all_installed_runs_no_pip(tmp_path: Path) -> None:
    rc, out, pip_calls = _run(
        tmp_path / "all", installed={"cryptography": "44.0.1", "e2b": "2.60.0", "boto3": "1.35.0"}
    )
    assert rc == 0 and END in out, out
    assert pip_calls == [], "an image that ships the extra (or an offline one) runs no pip"
    assert "every requirement is already installed" in out

    rc, out, pip_calls = _run(
        tmp_path / "old", installed={"cryptography": "41.0.0", "e2b": "2.60.0", "boto3": "1.35.0"}
    )
    assert rc == 0 and END in out, out
    _same(_requested(pip_calls), {"cryptography": _hosted_requirements()["cryptography"]})


@pytest.mark.parametrize(("packaged", "state_dir"), [(1, None), (1, ""), (0, "/state")])
def test_only_a_hosted_run_of_a_packaged_install_is_touched(
    tmp_path: Path, packaged: int, state_dir: str | None
) -> None:
    """A local packaged install, and a source checkout (whose ``[test]`` has the extra), read nothing and run no pip:
    the interpreter here does not exist, so any call would fail the step."""
    rc, out, pip_calls = _run(tmp_path, packaged=packaged, state_dir=state_dir, python=str(tmp_path / "no-python"))
    assert rc == 0 and END in out, out
    assert pip_calls == []


def test_a_wheel_whose_metadata_cannot_be_read_stops_the_install_by_name(tmp_path: Path) -> None:
    rc, out, pip_calls = _run(tmp_path, wheel=False)
    assert rc != 0 and END not in out
    assert "hosted extra: could not read its requirements" in out
    assert pip_calls == []


def test_a_failed_pip_stops_the_install_by_name(tmp_path: Path) -> None:
    rc, out, pip_calls = _run(tmp_path, pip_rc=3)
    assert rc != 0 and END not in out
    assert "hosted extra: pip install failed" in out and len(pip_calls) == 1


def test_the_step_runs_after_the_packaged_runtime_install() -> None:
    calls = re.findall(r"^(ensure_\w+)$", INSTALL_SH.read_text(encoding="utf-8"), re.M)
    assert calls[calls.index("ensure_inference_optimizer") + 1] == "ensure_hosted_extra", calls


def test_the_forge_profiling_fallback_reads_its_extra_through_the_same_helper(tmp_path: Path) -> None:
    rc, out, _ = _run(tmp_path, call="_installed_extra_requirements forge-profiling")
    assert rc == 0 and END in out, out
    declared = tomllib.loads((SOURCE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = {Requirement(spec).name for spec in declared["project"]["optional-dependencies"]["forge-profiling"]}
    printed = {Requirement(line).name for line in out.splitlines() if line and not line.startswith(END)}
    assert printed == expected
