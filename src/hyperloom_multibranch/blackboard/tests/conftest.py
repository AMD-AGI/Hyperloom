# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import FakeClock
from hyperloom_multibranch.blackboard.tests.sessions import snapshot

__all__ = ["clock", "open_as", "root", "snapshot"]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "blackboard"


@pytest.fixture
def open_as(root: Path, clock: FakeClock) -> Callable[..., LocalBlackboard]:
    def _open(role: Role, branch: str = "", name: str = "") -> LocalBlackboard:
        return LocalBlackboard(root, Writer(role, name or branch or role.value, branch), clock=clock)

    return _open
