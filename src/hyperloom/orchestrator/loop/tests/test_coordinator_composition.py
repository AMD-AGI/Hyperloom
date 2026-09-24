# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Coordinator is composed from mixins, and a name two of them define resolves silently by MRO order."""

from __future__ import annotations

from hyperloom.orchestrator.loop.coordinator import Coordinator


def test_every_name_on_the_coordinator_has_exactly_one_definer():
    definer: dict[str, str] = {}
    clashes = []
    for cls in Coordinator.__mro__[:-1]:
        for name in vars(cls):
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in definer:
                clashes.append(f"{name}: {definer[name]} and {cls.__name__}")
            else:
                definer[name] = cls.__name__
    assert clashes == []
