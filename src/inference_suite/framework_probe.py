# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ask the serving interpreter itself about the framework it runs.

A probe runs ``framework_python`` with the launch environment, in the directory the server starts in
(:attr:`~inference_suite.contracts.RunLayout.framework_cwd`), so it resolves the packages, and the patches on the path,
that the server will. The install it reports is what the version check (D6) holds to the recipe, and the
parser check (D7) runs its program through the same probe.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - the probe runs the interpreter the config names
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from inference_suite.frameworks import Framework, framework_facts

#: Starting an interpreter on a shared filesystem can take seconds; the install probe imports nothing.
PROBE_TIMEOUT_S = 60.0
_DETAIL_CHARS = 600

_INSTALL_PROGRAM = """\
import importlib.metadata, importlib.util, json, os, sys
name = sys.argv[1]
spec = importlib.util.find_spec(name)
try:
    version = importlib.metadata.version(name)
except importlib.metadata.PackageNotFoundError:
    version = None
print(json.dumps({
    "executable": os.path.realpath(sys.executable),
    "python": ".".join(str(part) for part in sys.version_info[:3]),
    "origin": os.path.realpath(spec.origin) if spec is not None and spec.origin else None,
    "version": version,
}))
"""


class FrameworkProbeError(RuntimeError):
    """The serving interpreter could not answer a probe, or does not have the framework importable."""


@dataclass(frozen=True)
class FrameworkInstall:
    """The framework the serving interpreter imports.

    ``version`` is the installed distribution's version, ``None`` when the package carries no distribution metadata.
    """

    executable: str
    python: str
    origin: str
    version: str | None


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


def probe_install(python: str, framework: Framework, env: Mapping[str, str], *, cwd: Path) -> FrameworkInstall:
    """The install of ``framework`` that ``python`` imports in ``env``, started in ``cwd``."""
    package = framework_facts(framework).python_package
    if package is None:
        raise FrameworkProbeError(f"{framework.value} ships no package to probe")
    answer = run_probe(python, _INSTALL_PROGRAM, (package,), env, cwd=cwd)
    origin = answer.get("origin")
    if not origin:
        raise FrameworkProbeError(f"{python} cannot import {package}")
    version = answer.get("version")
    return FrameworkInstall(
        executable=str(answer["executable"]),
        python=str(answer["python"]),
        origin=str(origin),
        version=None if version is None else str(version),
    )


__all__ = ["PROBE_TIMEOUT_S", "FrameworkInstall", "FrameworkProbeError", "probe_install", "run_probe"]
