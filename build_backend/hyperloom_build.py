# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PEP 517 backend: ``setuptools.build_meta`` plus the Inspector UI.

Before an sdist or a wheel is built, ``npm ci && npm run build`` in ``frontend/inspector`` writes the UI to
``src/hyperloom/inspector/static/modern/``, where the ``static/modern/**/*`` package-data glob picks it up. The Python
package stays stdlib-only and the installed Inspector needs no Node.

When npm is missing or its build fails, an existing build ships as is; with none, the package ships without the UI
and a warning, and the installed Inspector says how to get it. ``HYPERLOOM_REQUIRE_UI=1`` (release and packaging
CI) fails the build instead. ``HYPERLOOM_SKIP_UI_BUILD=1`` skips the build, and whatever ``static/modern/`` holds
ships as is. A wheel built from an sdist finds no ``frontend/`` (it isn't in the sdist) and uses the UI the sdist
already carries. Editable installs never build the UI: run ``npm run build`` by hand.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 - runs npm from the checkout at build time only
import sys
from pathlib import Path

from setuptools import build_meta as _orig
from setuptools.build_meta import *  # noqa: F403 - re-export the full backend interface

_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND = _ROOT / "frontend" / "inspector"
_OUT = _ROOT / "src" / "hyperloom" / "inspector" / "static" / "modern"


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _run_npm() -> str | None:
    """Build the UI with npm; the reason it could not, or None when it did."""
    npm = shutil.which("npm")
    if npm is None:
        return "npm was not found"
    for args in (["ci", "--no-audit", "--no-fund"], ["run", "build"]):
        step = f"npm {' '.join(args)}"
        print(f"Inspector UI: {step}", file=sys.stderr)
        # stdout would corrupt the PEP 517 hook output some frontends parse; send it to stderr.
        done = subprocess.run([npm, *args], cwd=_FRONTEND, check=False, stdout=sys.stderr)  # nosec B603
        if done.returncode != 0:
            return f"{step} failed with exit code {done.returncode}"
    if not (_OUT / "index.html").is_file():
        # npm reported success: a build that writes no page is a bug in the frontend's build, never a fallback.
        raise RuntimeError(f"Inspector UI: npm run build wrote no {_OUT.relative_to(_ROOT)}/index.html")
    return None


def build_ui() -> None:
    """Build the UI into ``static/modern/``. When it can't, ship the build there is, or warn that there is none."""
    # A failed npm build leaves an existing build that may be stale; a skip or an sdist ships it by design.
    if _flag("HYPERLOOM_SKIP_UI_BUILD"):
        reason, stale = "HYPERLOOM_SKIP_UI_BUILD is set", False
    elif not (_FRONTEND / "package.json").is_file():
        # Building from an sdist: it carries the UI built when the sdist was made.
        reason, stale = "frontend/inspector is not here", False
    else:
        reason, stale = _run_npm(), True
    if reason is None:
        return
    if (_OUT / "index.html").is_file():
        prefix = "WARNING: " if stale else ""
        print(f"{prefix}Inspector UI: {reason}; shipping the existing build as is", file=sys.stderr)
        return
    message = f"Inspector UI: {reason}; the package ships without it (see docs/inspector/README.md)"
    if _flag("HYPERLOOM_REQUIRE_UI"):
        raise RuntimeError(f"{message}, and HYPERLOOM_REQUIRE_UI is set")
    print(f"WARNING: {message}", file=sys.stderr)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    build_ui()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    build_ui()
    return _orig.build_sdist(sdist_directory, config_settings)
