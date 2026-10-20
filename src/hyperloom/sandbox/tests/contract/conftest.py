# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Binds the sandbox contract suite to an implementation; the harnesses are in :mod:`..contract_harness`."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.sandbox.tests.contract_harness import (
    SERVICE_URL_ENV,
    Harness,
    claw_service_harness,
    local_harness,
    standin_harness,
)


@pytest.fixture(params=["local", "e2b"])
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Harness]:
    if request.param == "local":
        build = local_harness()
    elif SERVICE_URL_ENV in os.environ:
        build = claw_service_harness()
    else:
        if request.node.get_closest_marker("claw_policy") is not None:
            pytest.skip(
                f"needs the Claw sandbox service's policy; set {SERVICE_URL_ENV} to run it against a real service"
            )
        build = standin_harness(tmp_path)
    with build as built:
        yield built
