# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_multibranch.notebook import LocalNotebook, Role, Writer
from hyperloom_multibranch.notebook.tests.factories import FakeClock


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
