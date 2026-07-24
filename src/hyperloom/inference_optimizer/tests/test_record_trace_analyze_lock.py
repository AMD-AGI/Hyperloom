# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior-lock tests for ``SharedState.record_trace_analyze``: 14-key dict,
snapshot_id wrap-on-clear, capped history with anchor, and swallowed history failures."""

from __future__ import annotations

from unittest.mock import patch

from hyperloom.orchestrator.state.shared_state import (
    _ROOFLINE_SNAPSHOTS_CAP,
    SharedState,
)


def _record(state: SharedState, trace: str) -> None:
    state.record_trace_analyze(
        {"trace_input": trace},
        {"hot_kernels": [], "trace_health_warnings": []},
    )


def test_written_dict_has_fourteen_keys() -> None:
    """Docstring says "11-field" but the code actually writes 14 keys."""
    state = SharedState()
    _record(state, "x")
    assert len(state.last_trace_analyze) == 14
    assert set(state.last_trace_analyze) == {
        "trace_input",
        "candidates_path",
        "kernel_roofline_path",
        "hot_kernels_top15",
        "kernel_roofline_top15",
        "skipped_kernels_top",
        "task_groups",
        "reusable_native_kernel_ids",
        "trace_health_warnings",
        "analysis_md_path",
        "analysis_md_text",
        "roofline_snapshot_id",
        "roofline_baseline_gain_at_snapshot",
        "ts",
    }


def test_snapshot_id_wraps_to_one_after_last_trace_analyze_cleared() -> None:
    """After ``last_trace_analyze`` is cleared (as profile promote does), the id
    resets to 1 because it is derived from the previous dict, not a counter."""
    state = SharedState()
    _record(state, "t1")
    assert state.last_trace_analyze["roofline_snapshot_id"] == 1
    _record(state, "t2")
    assert state.last_trace_analyze["roofline_snapshot_id"] == 2

    # Simulate profile promote wiping the canonical dict.
    state.last_trace_analyze = {}
    _record(state, "t3")
    # Actual behavior: id wraps back to 1, not 3.
    assert state.last_trace_analyze["roofline_snapshot_id"] == 1
    assert state.roofline_snapshot_id == 1


def test_roofline_snapshots_history_contains_duplicate_ids_after_clear() -> None:
    """The append-only history keeps the wrapped id, so duplicate ids appear."""
    state = SharedState()
    _record(state, "t1")
    _record(state, "t2")
    state.last_trace_analyze = {}
    _record(state, "t3")

    ids = [snap.get("snapshot_id") for snap in state.roofline_snapshots]
    assert ids == [1, 2, 1]


def test_roofline_snapshots_capped_keeps_baseline_anchor() -> None:
    """Beyond the cap, length is pinned and snapshot #1 stays as index 0."""
    state = SharedState()
    total = _ROOFLINE_SNAPSHOTS_CAP + 10
    for i in range(total):
        _record(state, f"t{i}")

    assert len(state.roofline_snapshots) == _ROOFLINE_SNAPSHOTS_CAP
    # Index 0 is always the first baseline anchor.
    assert state.roofline_snapshots[0]["snapshot_id"] == 1
    # Last is the most recent id.
    assert state.roofline_snapshots[-1]["snapshot_id"] == total
    # Tail is [base, *last (cap-1)]: index 1 is id total-(cap-1)+1.
    assert state.roofline_snapshots[1]["snapshot_id"] == total - (_ROOFLINE_SNAPSHOTS_CAP - 1) + 1


def test_history_block_failure_does_not_drop_canonical_write() -> None:
    """A failure inside the history block is swallowed; ``last_trace_analyze`` is
    still written and no history entry is appended."""
    state = SharedState()
    _record(state, "first")
    assert len(state.roofline_snapshots) == 1

    with patch(
        "hyperloom.orchestrator.kernel.roofline_snapshot.build_roofline_snapshot",
        side_effect=RuntimeError("boom"),
    ):
        _record(state, "second")

    # Canonical write survives the history-block exception.
    assert state.last_trace_analyze["roofline_snapshot_id"] == 2
    assert state.last_trace_analyze["trace_input"] == "second"
    assert state.roofline_snapshot_id == 2
    # History block bailed before appending, so no new entry.
    assert len(state.roofline_snapshots) == 1
