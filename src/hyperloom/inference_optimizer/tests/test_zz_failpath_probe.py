# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TEMPORARY probe: one assertion FAILURE + one fixture-setup ERROR.

Verifies (1) precise nodeid extraction from pytest's -rfE summary (with a
FAILED and an ERROR) and (2) that coverage is still combined+reported when
tests fail but all shards reported. Remove after verification.
"""

import pytest


def test_failpath_probe_forced_failure():
    assert False, "intentional probe FAILED to validate precise-nodeid extraction"


@pytest.fixture
def _probe_broken_fixture():
    raise RuntimeError("intentional probe fixture ERROR to validate ERROR extraction")


def test_failpath_probe_fixture_error(_probe_broken_fixture):
    assert True