# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test here launches its stand-in specialists as on a host without bubblewrap (:mod:`.unconfined_specialists`)."""

import pytest

from hyperloom_multibranch.tests.unconfined_specialists import launch_specialists_without_bwrap


@pytest.fixture(autouse=True)
def specialists_without_bwrap(monkeypatch: pytest.MonkeyPatch) -> None:
    launch_specialists_without_bwrap(monkeypatch)
