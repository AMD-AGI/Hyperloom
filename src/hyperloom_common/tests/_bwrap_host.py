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
from typing import NoReturn

import pytest

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
