# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TEMPORARY probe: forces a shard FAILED + a fixture ERROR.

Validates that the coverage job extracts BOTH a test <failure> (assertion) and
a test <error> (fixture setup) from the JUnit XML. MUST be removed after the
CI fail-visibility verification; do not keep on any long-lived branch.
"""

import pytest


def test_failpath_probe_forced_failure():
    assert False, "intentional probe FAILED to validate CI shard-failure visibility"


@pytest.fixture
def _probe_broken_fixture():
    raise RuntimeError("intentional probe fixture ERROR to validate ERROR extraction")


def test_failpath_probe_fixture_error(_probe_broken_fixture):
    assert True