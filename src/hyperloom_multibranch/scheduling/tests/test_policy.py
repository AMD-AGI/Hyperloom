# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduling policy version names the policy's values: different values, a different version."""

from __future__ import annotations

from hyperloom_multibranch.scheduling.policy import SCHEDULING_POLICY, policy_version
from hyperloom_multibranch.specialist.prompts import VERSION_HEX


def test_different_policy_values_get_different_versions() -> None:
    changed = {**SCHEDULING_POLICY, "MB_MIN_USEFUL_COLD_STARTS": SCHEDULING_POLICY["MB_MIN_USEFUL_COLD_STARTS"] + 1}

    assert policy_version(changed) != policy_version(SCHEDULING_POLICY)


def test_the_version_does_not_depend_on_the_order_the_values_are_listed_in() -> None:
    reversed_order = dict(reversed(list(SCHEDULING_POLICY.items())))

    assert policy_version(reversed_order) == policy_version(SCHEDULING_POLICY)
    assert len(policy_version(SCHEDULING_POLICY)) == VERSION_HEX
