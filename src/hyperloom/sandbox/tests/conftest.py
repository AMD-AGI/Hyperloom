# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for tests that run against the E2B stand-in: skipped, with the reason, where it cannot run."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.sandbox.tests.e2b_standin import ENVD_ENV, StandIn, StandInTemplate, unavailable_reason


@pytest.fixture(scope="session")
def e2b_envd() -> Path:
    """The envd binary the stand-in runs; the test is skipped when this machine cannot run the stand-in."""
    reason = unavailable_reason(os.environ.get(ENVD_ENV))
    if reason is not None:
        pytest.skip(reason)
    return Path(os.environ[ENVD_ENV])


@pytest.fixture(scope="session")
def standin(e2b_envd: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[StandIn]:
    """One stand-in for the whole session, with a template per role and one that starts a command."""
    templates = [
        StandInTemplate("dev", "dev"),
        StandInTemplate("sub", "subrun"),
        StandInTemplate(
            "sealed", "measurement", sealed=True, start=("/bin/sh", "-c", "echo ready > /out/started; sleep 600")
        ),
    ]
    with StandIn(e2b_envd, tmp_path_factory.mktemp("e2b-standin"), templates) as running:
        yield running
