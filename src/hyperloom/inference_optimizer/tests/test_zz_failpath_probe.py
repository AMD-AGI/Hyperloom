# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TEMPORARY probe: forces a shard failure to verify CI fail-visibility.

This file is intentionally failing and MUST be removed after verifying that
the failing shard's matrix cell turns red (the 'Fail job if this shard's tests
failed' step) and that the coverage job reports the failure.
"""


def test_failpath_probe_forced_failure():
    assert False, "intentional probe failure to validate CI shard-failure visibility"