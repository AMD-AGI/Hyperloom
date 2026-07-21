"""Temporary probe test to verify shard-failure visibility in CI.

This file is intentionally added to trigger a real test failure so the new
CI annotations/job-summary can be observed on a real run. It will be removed
once verified -- see commit history for the removal commit.
"""


def test_zz_probe_intentional_failure():
    assert False, "intentional probe failure to verify CI failure visibility"