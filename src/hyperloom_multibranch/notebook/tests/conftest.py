# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import FakeClock
from hyperloom_multibranch.notebook.tests.sessions import snapshot

__all__ = ["clock", "open_as", "root", "snapshot"]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "notebook"


@pytest.fixture
def open_as(root: Path, clock: FakeClock) -> Callable[..., LocalNotebook]:
    def _open(role: Role, branch: str = "", name: str = "") -> LocalNotebook:
        return LocalNotebook(root, Writer(role, name or branch or role.value, branch), clock=clock)

    return _open
