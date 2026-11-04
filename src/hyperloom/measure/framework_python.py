# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Find a framework package in another Python's environment, for frameworks the image's default Python lacks.

Some framework images carry the serving framework in its own virtual environment, and a test names that interpreter
with ``HYPERLOOM_FRAMEWORK_PYTHON`` in its point; the runner already builds the farm and starts the server with it.
:func:`locate_installed_package` only looks at the process that compiles, so it cannot find the base of a patch for
such a framework. :func:`locator_for_python` asks that interpreter instead, under the same rules: the package must be
an installed one, never a source tree somebody could change under the compile.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hyperloom.measure.compile import ComponentLayout, PackageBase, PackageLocator, UnsupportedComponentError

__all__ = ["locator_for_python"]

_PROBE_TIMEOUT_SEC = 120
_SITE_DIRS = ("site-packages", "dist-packages")
#: Run with ``-I`` from a neutral directory: no ``PYTHONPATH`` and no current directory on the import path.
_PROBE = (
    "import importlib.metadata as md, importlib.util as u, json, sys\n"
    "name = sys.argv[1]\n"
    "try:\n"
    "    spec = u.find_spec(name)\n"
    "except (ImportError, ValueError):\n"
    "    spec = None\n"
    "dists = md.packages_distributions().get(name, [])\n"
    "dist = md.distribution(dists[0]) if dists else None\n"
    "print(json.dumps({\n"
    "    'locations': list(spec.submodule_search_locations or []) if spec else [],\n"
    "    'dist_name': dist.metadata['Name'] if dist else None,\n"
    "    'dist_version': dist.version if dist else None,\n"
    "}))\n"
)


def locator_for_python(python: str) -> PackageLocator:
    """A package locator that looks where ``python`` imports the package from, without importing it here."""

    def locate(layout: ComponentLayout) -> PackageBase:
        return _locate(python, layout)

    return locate


def _locate(python: str, layout: ComponentLayout) -> PackageBase:
    name = layout.import_name
    try:
        done = subprocess.run(
            [python, "-I", "-c", _PROBE, name],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SEC,
            cwd="/",
            check=False,
        )
        found = json.loads(done.stdout) if done.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise UnsupportedComponentError(f"cannot ask {python} where {name} is: {type(exc).__name__}") from exc
    locations = found.get("locations") if isinstance(found, dict) else None
    if not locations:
        raise UnsupportedComponentError(f"package {name} is not installed for {python}")
    path = Path(locations[0]).resolve()
    if not any(part in _SITE_DIRS for part in path.parts):
        raise UnsupportedComponentError(
            f"package {name} at {path} is not installed in site-packages; hosted measurement cannot take it as a base"
        )
    return PackageBase(name, path, found.get("dist_name"), found.get("dist_version"))
