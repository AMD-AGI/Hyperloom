# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the status left on ``geak_pending`` after a declined 2b.

``rebench_unavailable`` and a decline are different facts. The first means the
rebench never got to run, which is a dispatch problem and the candidate should
be retried. The second means the rebench was refused on purpose. When the
refusal was an overlay that cannot install, a retry changes nothing, and a
reader who cannot tell the two apart will retry forever or give up wrongly.
"""

from __future__ import annotations

from hyperloom.orchestrator.phases.kernel import _geak_decline_status


def test_overlay_refusal_gets_its_own_status() -> None:
    assert _geak_decline_status("geak_overlay_unloadable") == "overlay_unloadable"


def test_overlay_refusal_match_is_case_and_space_insensitive() -> None:
    assert _geak_decline_status("  GEAK_Overlay_Unloadable  ") == "overlay_unloadable"


def test_any_other_refusal_is_a_plain_decline() -> None:
    assert _geak_decline_status("no_slot_free") == "rebench_declined"
    assert _geak_decline_status("baseline_missing") == "rebench_declined"


def test_absent_reason_is_a_plain_decline_not_an_overlay_verdict() -> None:
    # A missing reason must never be read as "the overlay was the problem":
    # that would send a retryable candidate to a terminal status.
    assert _geak_decline_status(None) == "rebench_declined"
    assert _geak_decline_status("") == "rebench_declined"
    assert _geak_decline_status("   ") == "rebench_declined"


def test_neither_status_collides_with_the_dispatch_failure_status() -> None:
    # The dispatch-failure path still writes ``rebench_unavailable``; a decline
    # must never overwrite that live diagnostic.
    for reason in ("geak_overlay_unloadable", "anything_else", None):
        assert _geak_decline_status(reason) != "rebench_unavailable"
