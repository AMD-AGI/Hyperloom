# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialists the multi-branch tests launch run as on a host without bubblewrap, unless a test picks a tier itself.

The tests that drive ``LocalAgentRunner`` start stand-in CLIs that hold no LLM credential and read and write where a
sandbox would not let them. Where bubblewrap works (CI installs it), the chooser would put each in a sandbox and refuse
a claude CLI that has no login it can read there. The account tier and the refusals are unchanged; the sandboxed launch
itself is tested where it is made (``hyperloom.orchestrator.specialists.tests.test_specialist_sandbox``).
"""

import functools

import pytest

from hyperloom.common.agent_isolation import choose_agent_isolation
from hyperloom.orchestrator.specialists import launch_env
from hyperloom_common.bwrap import BwrapProbe

#: What the bubblewrap probe says on a host without it.
NO_BWRAP = BwrapProbe(None, None, ("bubblewrap: not found on PATH",))


def launch_specialists_without_bwrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Choose a specialist's isolation as on a host whose bubblewrap probe finds none."""
    monkeypatch.setattr(
        launch_env, "choose_agent_isolation", functools.partial(choose_agent_isolation, probe=_no_bwrap)
    )


def _no_bwrap(_path: str | None) -> BwrapProbe:
    return NO_BWRAP
