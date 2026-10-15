# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A variant that ablates a flag and measures nothing: did the flag come off, or was the idea wrong?"""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors.explore import _ablation_realized

_RADIX = "--disable-radix-cache"


def _observed(flags: str) -> dict[str, str]:
    return {"observed_server_launch_flags": flags}


def test_a_flag_absent_from_the_launch_confirms_the_ablation_took():
    realized = _ablation_realized([_RADIX], _observed("--tp 8 --enable-metrics --context-length 1048576"))

    assert realized == {"requested": [_RADIX], "still_present": [], "took_effect": True}


def test_a_flag_still_on_the_launch_says_the_ablation_was_blocked():
    realized = _ablation_realized([_RADIX], _observed(f"--tp 8 {_RADIX} --enable-metrics"))

    assert realized["still_present"] == [_RADIX]
    assert realized["took_effect"] is False


def test_a_flag_the_launcher_spells_with_a_value_is_matched_on_its_name():
    realized = _ablation_realized(["--mem-fraction-static"], _observed("--tp 8 --mem-fraction-static=0.8"))

    assert realized["took_effect"] is False


def test_an_unobserved_launch_is_unknown_rather_than_confirmed():
    """Returning a took_effect=True here would assert an ablation nobody watched."""
    assert _ablation_realized([_RADIX], {}) is None
    assert _ablation_realized([_RADIX], _observed("")) is None
    assert _ablation_realized([_RADIX], None) is None


def test_a_variant_that_ablates_nothing_reports_nothing():
    assert _ablation_realized([], _observed("--tp 8")) is None
    assert _ablation_realized(None, _observed("--tp 8")) is None
    assert _ablation_realized(["  "], _observed("--tp 8")) is None
