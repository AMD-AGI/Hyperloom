# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real bubblewrap in tests: a host without it (or without user namespaces) skips the test, or fails it where the
isolation tests are required.

CI sets ``HYPERLOOM_REQUIRE_ISOLATION_TESTS=1`` after it installs bubblewrap and lets unprivileged processes make user
namespaces, so there a test that cannot launch a sandbox is a failure naming why, never a silent skip. A probe that
fails for any other reason is a failure everywhere: that is the adapter, not the host.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from hyperloom_common import bwrap
from hyperloom_common.bwrap import BwrapProbe, bwrap_probe

#: Set to ``1`` where the isolation tests must run: an unavailable sandbox then fails the test instead of skipping it.
REQUIRE_ENV = "HYPERLOOM_REQUIRE_ISOLATION_TESTS"

_NO_USER_NAMESPACE = re.compile(r"uid map|new namespace|no permissions to creat", re.IGNORECASE)


def unavailable(reason: str) -> NoReturn:
    """Skip the test for ``reason``, or fail it when :data:`REQUIRE_ENV` is ``1``."""
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{reason} ({REQUIRE_ENV}=1: this test must run here)")
    pytest.skip(reason)


def usable_bwrap() -> BwrapProbe:
    """The probe of the ``bwrap`` on ``PATH``, which passed; else the test skips (no bubblewrap, or no user namespace
    for it) or fails (anything else)."""
    found = bwrap_probe()
    if not found.ok:
        reasons = "; ".join(found.problems)
        if found.path is None or _NO_USER_NAMESPACE.search(reasons):
            unavailable(f"bubblewrap cannot sandbox an agent here: {reasons}")
        pytest.fail(f"bubblewrap is here, but its probe failed: {reasons}")
    return found


def as_owner(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run Python ``code`` with ``args`` as this uid without the capabilities that let root ignore a file's mode (on
    CI, a plain user, already without them): what removing a tree an agent left meets as its owner. Skipped where
    root cannot drop them."""
    drop: list[str] = []
    if os.geteuid() == 0:
        setpriv = shutil.which("setpriv")
        drop = [setpriv or "setpriv", "--bounding-set=-dac_override,-dac_read_search", "--"]
        if setpriv is None or subprocess.run([*drop, "true"], check=False, timeout=60).returncode != 0:
            pytest.skip("root here cannot drop CAP_DAC_OVERRIDE (setpriv), so a file's mode would not hold")
    source = str(Path(bwrap.__file__).parents[1])
    return subprocess.run(
        [*drop, sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {source!r})\n{code}", *args],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", os.defpath)},
        timeout=60,
        check=False,
    )
