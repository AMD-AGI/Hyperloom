# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ask the serving interpreter itself about the framework it runs.

A probe runs ``framework_python`` with the launch environment, in the directory the server starts in
(:attr:`~inference_suite.contracts.RunLayout.framework_cwd`), so it resolves the packages, and the patches on the path,
that the server will. The install it reports is what the version check (D6) holds to the recipe, and the
parser check (D7) runs its program through the same probe. A native server is asked instead: its program, found on the
launch environment's ``PATH``, reports its version itself, and its ``--help`` is the flag list D7 holds a launch to.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # nosec B404 - the probe runs the interpreter the config names
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from inference_suite.frameworks import Framework, framework_facts

#: Starting an interpreter on a shared filesystem can take seconds; the install probe imports nothing.
PROBE_TIMEOUT_S = 60.0
_DETAIL_CHARS = 600
#: llama.cpp's ``--version`` line: ``version: 1.0.0-dev (build 12439, commit 8bfd64b5a)``, ``version: 6789 (8bfd64b5a)``
#: before it numbered releases.
_NATIVE_VERSION = re.compile(r"^version: (\S+)", re.MULTILINE)

_INSTALL_PROGRAM = """\
import importlib.metadata, importlib.util, json, os, sys
name = sys.argv[1]
spec = importlib.util.find_spec(name)
try:
    version = importlib.metadata.version(name)
except importlib.metadata.PackageNotFoundError:
    version = None
roots = {}
for package in dict.fromkeys(sys.argv[1:]):
    found = spec if package == name else importlib.util.find_spec(package)
    where = [*(found.submodule_search_locations or ()), found.origin] if found is not None else []
    if where and where[0]:
        roots[package] = os.path.realpath(where[0])
print(json.dumps({
    "executable": os.path.realpath(sys.executable),
    "python": ".".join(str(part) for part in sys.version_info[:3]),
    "origin": os.path.realpath(spec.origin) if spec is not None and spec.origin else None,
    "version": version,
    "import_roots": roots,
}))
"""


class FrameworkProbeError(RuntimeError):
    """The serving interpreter could not answer a probe, or does not have the framework importable."""


@dataclass(frozen=True)
class FrameworkInstall:
    """The framework the serving interpreter imports.

    ``version`` is the installed distribution's version, ``None`` when the package carries no distribution metadata.
    ``import_roots`` maps the framework's package and each requested package the interpreter can import to the
    directory (a module's file) it imports it from.
    """

    executable: str
    python: str
    origin: str
    version: str | None
    import_roots: Mapping[str, str]


def run_probe(
    python: str, program: str, args: Sequence[str], env: Mapping[str, str], *, cwd: Path
) -> Mapping[str, object]:
    """The JSON object ``program`` prints last when ``python`` runs it with ``args`` in ``env``, started in ``cwd``."""
    try:
        completed = subprocess.run(  # nosec B603 - argv[0] is the interpreter the config names
            [python, "-c", program, *args],
            env=dict(env),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise FrameworkProbeError(f"the probe in {python} ran past {PROBE_TIMEOUT_S:g} s") from error
    except OSError as error:
        raise FrameworkProbeError(f"{python} could not run the probe: {error}") from error
    lines = completed.stdout.strip().splitlines()
    try:
        answer = json.loads(lines[-1]) if completed.returncode == 0 and lines else None
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        tail = (completed.stderr or completed.stdout).strip()[-_DETAIL_CHARS:]
        raise FrameworkProbeError(f"the probe in {python} exited {completed.returncode} with no answer: {tail}")
    return answer


def _native_executable(program: str, env: Mapping[str, str]) -> str:
    executable = shutil.which(program, path=env.get("PATH"))
    if executable is None:
        raise FrameworkProbeError(f"{program} is not on the launch PATH")
    return executable


def _run_native(executable: str, option: str, env: Mapping[str, str], *, cwd: Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(  # nosec B603 - argv[0] is the framework's server program
            [executable, option],
            env=dict(env),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise FrameworkProbeError(f"{executable} {option} ran past {PROBE_TIMEOUT_S:g} s") from error
    except OSError as error:
        raise FrameworkProbeError(f"{executable} could not run: {error}") from error


def native_executable(program: str, env: Mapping[str, str]) -> str:
    """The real path of the file that runs as the native server ``program`` where ``env``'s ``PATH`` resolves it.

    Raises:
        FrameworkProbeError: When ``program`` is not on the launch ``PATH``.
    """
    return os.path.realpath(_native_executable(program, env))


#: ``--help`` text by the binary's real path and mtime, so a rebuilt server is read again.
_NATIVE_HELP: dict[tuple[str, int], str] = {}


def native_help(program: str, env: Mapping[str, str], *, cwd: Path) -> str:
    """What the native server ``program``, as ``env``'s ``PATH`` resolves it, prints for ``--help``.

    Raises:
        FrameworkProbeError: When ``program`` is not on the launch ``PATH``, or its ``--help`` fails or prints nothing.
    """
    executable = native_executable(program, env)
    key = (executable, os.stat(executable).st_mtime_ns)
    if key not in _NATIVE_HELP:
        completed = _run_native(executable, "--help", env, cwd=cwd)
        printed = completed.stdout + completed.stderr
        if completed.returncode != 0 or not printed.strip():
            tail = printed.strip()[-_DETAIL_CHARS:]
            raise FrameworkProbeError(f"{executable} --help exited {completed.returncode}: {tail}")
        _NATIVE_HELP[key] = printed
    return _NATIVE_HELP[key]


def _probe_native(program: str, env: Mapping[str, str], *, cwd: Path) -> FrameworkInstall:
    """The native server ``program`` as ``env``'s ``PATH`` resolves it, and the version its ``--version`` prints."""
    executable = _native_executable(program, env)
    completed = _run_native(executable, "--version", env, cwd=cwd)
    printed = completed.stdout + completed.stderr
    found = _NATIVE_VERSION.search(printed)
    if completed.returncode != 0 or found is None:
        tail = printed.strip()[-_DETAIL_CHARS:]
        raise FrameworkProbeError(f"{executable} --version exited {completed.returncode} with no version: {tail}")
    return FrameworkInstall(
        executable=os.path.realpath(executable),
        python="",
        origin=os.path.realpath(executable),
        version=found[1],
        import_roots={},
    )


def probe_install(
    python: str, framework: Framework, env: Mapping[str, str], *, cwd: Path, packages: Sequence[str] = ()
) -> FrameworkInstall:
    """The install of ``framework`` that ``python`` imports in ``env``, started in ``cwd``, and where it imports
    ``framework``'s package and each of ``packages`` from; a native server's program as ``env`` resolves it,
    whatever ``python`` is, with no import roots."""
    facts = framework_facts(framework)
    if facts.server_command is not None and facts.server_command.native:
        return _probe_native(facts.server_command.entrypoint[0], env, cwd=cwd)
    package = facts.python_package
    if package is None:
        raise FrameworkProbeError(f"{framework.value} ships no package to probe")
    answer = run_probe(python, _INSTALL_PROGRAM, (package, *packages), env, cwd=cwd)
    origin = answer.get("origin")
    if not origin:
        raise FrameworkProbeError(f"{python} cannot import {package}")
    version = answer.get("version")
    return FrameworkInstall(
        executable=str(answer["executable"]),
        python=str(answer["python"]),
        origin=str(origin),
        version=None if version is None else str(version),
        import_roots={str(name): str(root) for name, root in dict(answer["import_roots"]).items()},
    )


__all__ = [
    "PROBE_TIMEOUT_S",
    "FrameworkInstall",
    "FrameworkProbeError",
    "native_executable",
    "native_help",
    "probe_install",
    "run_probe",
]
