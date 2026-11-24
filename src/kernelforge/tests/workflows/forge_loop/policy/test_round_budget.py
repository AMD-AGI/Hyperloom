# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the remaining campaign budget admits another round."""

from __future__ import annotations

from kernelforge.workflows.forge_loop.policy.round_budget import (
    DISPATCH_FLOOR_SEC,
    DISPATCH_SESSION_SEC,
    MEASUREMENT_SEC,
    admit_dispatch,
)


def test_a_round_is_admitted_when_a_session_and_its_measurement_fit():
    required = max(DISPATCH_FLOOR_SEC, DISPATCH_SESSION_SEC + MEASUREMENT_SEC)

    admitted = admit_dispatch(remaining_sec=required)
    refused = admit_dispatch(remaining_sec=required - 1.0)

    assert (admitted.admitted, admitted.required_sec) == (True, required)
    assert refused.admitted is False
    assert f"{required / 60:.0f} min needed" in refused.summary()
    assert f"{(required - 1.0) / 60:.0f} min remain" in refused.summary()


def test_a_rounds_price_does_not_depend_on_how_many_lanes_it_runs():
    """The lanes of a round run side by side, so a wider round costs no more of the clock."""
    assert admit_dispatch(remaining_sec=10_000.0).required_sec == admit_dispatch(remaining_sec=1.0).required_sec
