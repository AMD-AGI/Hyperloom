# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for SharedState prompt-rendering helpers (populated branches)."""

from __future__ import annotations

from hyperloom.orchestrator.state.shared_state import SharedState


def test_to_gaps_summary_max_attempts_zero_shows_count_only():
    st = SharedState()
    st.upsert_gap({"canonical_id": "g1", "symptom": "test", "layer": "framework", "severity": "medium"})
    st.append_gap_attempt("g1", {"action": "explore", "outcome": "FAILED", "error_class": "x", "failure_id": "fid1"})
    out = st.to_gaps_summary(max_attempts=0)
    assert "attempts=1" in out
    assert "fid1" not in out


def test_to_gaps_summary_max_attempts_shows_rows():
    st = SharedState()
    st.upsert_gap({"canonical_id": "g1", "symptom": "test", "layer": "framework", "severity": "medium"})
    st.append_gap_attempt("g1", {"action": "explore", "outcome": "FAILED", "error_class": "x", "failure_id": "fid1"})
    out = st.to_gaps_summary(max_attempts=5)
    assert "fid1" in out
    assert "attempt:" in out
