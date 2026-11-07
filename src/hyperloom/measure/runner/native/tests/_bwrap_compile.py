# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The compile's bubblewrap tier for real: the account tier taken away, as on a host whose user namespace maps only its
root, and the ``bwrap`` on ``PATH`` probed for real (a host without one skips, or fails where the isolation tests are
required)."""

from __future__ import annotations

import pytest

from hyperloom_common.tests._bwrap_host import usable_bwrap

from hyperloom.measure.runner.native import isolated
from hyperloom.measure.runner.native.isolated import ISOLATION_BWRAP_NETNS, Isolation, local_isolation

#: Why the account tier does not work, as a host that maps only its root says it.
ONE_ID = "uid 65534, gid 65534 are not mapped in this process's user namespace (a stand-in)"


def bwrap_compile_isolation(monkeypatch: pytest.MonkeyPatch) -> Isolation:
    """What :func:`~hyperloom.measure.runner.native.isolated.local_isolation` gives such a host: bubblewrap."""
    usable_bwrap()
    monkeypatch.setattr(isolated, "account_problem", lambda: ONE_ID)
    found = local_isolation(lambda unshare: pytest.fail("a sandbox has a network of its own: nothing to probe"))
    assert found is not None and found.level == ISOLATION_BWRAP_NETNS and found.bwrap is not None, found
    return found
