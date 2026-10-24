# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduling policy a session runs under: the tuned values its decisions rest on, and the version naming them.

Each value stays a constant in the module that uses it; this module only lists which of them are policy, so that two
sessions run under different values tell apart in their manifest fingerprints. Values the run settings or the
manifest already carry, and constants that mirror the main flow's own bounds, are not policy and stay out. A step
that adds or changes a policy constant adds it here in the same change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

from hyperloom_multibranch.notebook import STALL_TESTS
from hyperloom_multibranch.notebook.rsi.frontier import SETTLED_TESTS
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_SHARE, MB_EXPLORE_SHARE, MB_FINAL_SHARE
from hyperloom_multibranch.scheduling.events import MB_BIG_GAIN_BARS, MB_BOTTLENECK_SHIFT_PP
from hyperloom_multibranch.scheduling.final import MB_SWEEP_DEGRADATION_PCT
from hyperloom_multibranch.scheduling.similarity import MB_SIMILARITY_THRESHOLD
from hyperloom_multibranch.scheduling.widening import MB_MIN_USEFUL_COLD_STARTS
from hyperloom_multibranch.specialist.prompts import VERSION_HEX

SCHEDULING_POLICY: Mapping[str, float] = MappingProxyType(
    {
        "MB_BIG_GAIN_BARS": MB_BIG_GAIN_BARS,
        "MB_BOTTLENECK_SHIFT_PP": MB_BOTTLENECK_SHIFT_PP,
        "MB_BRANCH_GRACE_TESTS": SETTLED_TESTS,
        "MB_CLOSE_SHARE": MB_CLOSE_SHARE,
        "MB_EXPLORE_SHARE": MB_EXPLORE_SHARE,
        "MB_FINAL_SHARE": MB_FINAL_SHARE,
        "MB_MIN_USEFUL_COLD_STARTS": MB_MIN_USEFUL_COLD_STARTS,
        "MB_SIMILARITY_THRESHOLD": MB_SIMILARITY_THRESHOLD,
        "MB_SWEEP_DEGRADATION_PCT": MB_SWEEP_DEGRADATION_PCT,
        "STALL_TESTS": STALL_TESTS,
    }
)


def policy_version(policy: Mapping[str, float]) -> str:
    """The first hex digits of the sha256 of ``policy``'s values by name, the same length as the prompt versions."""
    text = json.dumps(dict(policy), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:VERSION_HEX]


__all__ = ["SCHEDULING_POLICY", "policy_version"]
