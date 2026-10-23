# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rig the start-up script and the cache builder are tested against, built once for the session."""

from __future__ import annotations

import pytest

from hyperloom.measure.runner.tests.bootstrap_rig import Rig, make_rig


@pytest.fixture(scope="session")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Rig:
    """A stand-in for the framework image, a miniature repository and the runner cache built from it."""
    return make_rig(tmp_path_factory.mktemp("rig") / "rig")
