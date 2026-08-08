# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collective-lane wiring: candidate selection, the KERNEL-entry gate, and state.

The lane is Coordinator-driven only (like fusion, unlike gemm_tuning): the gate
is deterministic, so it is deliberately absent from the LLM action surface.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.kernel.request_handlers import (
    KERNEL_REQUEST_HANDLERS,
    _collective_budget,
    select_collective_candidate,
)


def _state_with_remaining(minutes):
    return SimpleNamespace(remaining_minutes=lambda: minutes)


class TestCollectiveBudget:
    """forge-loop falls back to ONE hour without --max-hours, so an unset budget
    silently caps the lane: a 12-hour session once managed a single iteration
    and exited budget_exhausted."""

    def test_budget_follows_the_session_minus_a_reserve(self):
        hours, timeout = _collective_budget(_state_with_remaining(600.0), None, 14400)
        assert hours == 9.25  # 600 - 45 reserve
        assert timeout == 14400

    def test_short_session_also_clamps_the_wrapper_timeout(self):
        # forge-loop rounds --max-hours up to 1.0, so a sub-hour window has to be
        # enforced through the timeout instead.
        hours, timeout = _collective_budget(_state_with_remaining(90.0), None, 14400)
        assert hours == 0.75
        assert timeout == 2700

    def test_explicit_request_wins(self):
        hours, timeout = _collective_budget(_state_with_remaining(600.0), 2.0, 14400)
        assert hours == 2.0
        assert timeout == 14400

    def test_unbounded_session_defers_to_forge(self):
        assert _collective_budget(_state_with_remaining(None), None, 14400) == (None, 14400)

    def test_budget_below_the_reserve_defers_to_forge(self):
        assert _collective_budget(_state_with_remaining(30.0), None, 14400) == (None, 14400)

    def test_state_without_the_hook_is_tolerated(self):
        assert _collective_budget(SimpleNamespace(), None, 14400) == (None, 14400)
from hyperloom.orchestrator.phases.kernel import KernelPhase


def _collective_entry(**extra) -> dict:
    entry = {
        "kernel_id": "k007",
        "name": "hipLaunchKernel->_ZN5aiter18all_reduce... (Synthetic Op)",
        "gpu_pct": 4.5,
        "reusable_native_kernel": True,
        "source_file": "/sgl-workspace/aiter/csrc/include/custom_all_reduce.cuh",
        "kernel_contract": {"kind": "collective", "collective_op": "all_reduce", "world_size": 8},
    }
    entry.update(extra)
    return entry


def _state_with(*entries) -> SimpleNamespace:
    return SimpleNamespace(last_trace_analyze={"hot_kernels_top15": list(entries)})


def _projection(entry: dict) -> dict:
    """The roofline-oriented subset shared state actually keeps.

    ``record_trace_analyze`` drops ``kernel_contract`` (and repo/shapes), so a
    lane reading only this can never recognise a collective.
    """
    keep = ("kernel_id", "name", "gpu_pct", "source_file", "reusable_native_kernel")
    return {k: entry[k] for k in keep if k in entry}


def _state_from_disk(tmp_path, *entries) -> SimpleNamespace:
    """State shaped like a real session: projection in memory, full rows on disk."""
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels": list(entries)}), encoding="utf-8")
    return SimpleNamespace(
        last_trace_analyze={
            "hot_kernels_top15": [_projection(e) for e in entries],
            "candidates_path": str(path),
        }
    )


# --- Candidate selection ------------------------------------------------------


def test_selects_the_hottest_collective():
    hot = _collective_entry(kernel_id="k002", gpu_pct=9.1)
    picked = select_collective_candidate(_state_with(_collective_entry(), hot))
    assert picked["kernel_id"] == "k002"


def test_skips_non_collective_kernels():
    gemm = _collective_entry(kernel_id="k001", gpu_pct=40.0, kernel_contract={"kind": "gemm"})
    picked = select_collective_candidate(_state_with(gemm, _collective_entry()))
    assert picked["kernel_id"] == "k007"


def test_skips_non_reusable_candidates():
    """nccl/rccl reach here already marked non-reusable; they must stay out."""
    vendor = _collective_entry(kernel_id="k003", gpu_pct=30.0, reusable_native_kernel=False)
    assert select_collective_candidate(_state_with(vendor)) is None


def test_skips_candidates_without_source():
    assert select_collective_candidate(_state_with(_collective_entry(source_file=""))) is None


def test_returns_none_without_analysis():
    assert select_collective_candidate(SimpleNamespace(last_trace_analyze=None)) is None
    assert select_collective_candidate(SimpleNamespace()) is None


# --- The enriched rows live on disk, not in shared state ----------------------


def test_reads_the_enriched_rows_from_candidates_path(tmp_path):
    """shared state's projection has no kernel_contract; the file does."""
    state = _state_from_disk(tmp_path, _collective_entry())
    picked = select_collective_candidate(state)
    assert picked is not None
    assert picked["kernel_id"] == "k007"
    # The in-memory projection alone could not have produced this.
    assert all("kernel_contract" not in row for row in state.last_trace_analyze["hot_kernels_top15"])


def test_still_ranks_by_gpu_pct_when_reading_from_disk(tmp_path):
    hot = _collective_entry(kernel_id="k002", gpu_pct=9.1)
    picked = select_collective_candidate(_state_from_disk(tmp_path, _collective_entry(), hot))
    assert picked["kernel_id"] == "k002"


def test_falls_back_to_the_projection_when_the_file_is_missing(tmp_path):
    state = SimpleNamespace(
        last_trace_analyze={
            "hot_kernels_top15": [_collective_entry()],
            "candidates_path": str(tmp_path / "gone.json"),
        }
    )
    assert select_collective_candidate(state) is not None


def test_unreadable_candidates_file_does_not_raise(tmp_path):
    bad = tmp_path / "kernel_candidates.json"
    bad.write_text("{not json", encoding="utf-8")
    state = SimpleNamespace(
        last_trace_analyze={"hot_kernels_top15": [_collective_entry()], "candidates_path": str(bad)}
    )
    assert select_collective_candidate(state) is not None


# --- KERNEL-entry gate --------------------------------------------------------


def _gate(*, tp=8, comm_pct=5.0, last_collective=None, skip_env=None, monkeypatch=None, analysis=True):
    if monkeypatch is not None:
        monkeypatch.setenv("HYPERLOOM_SKIP_COLLECTIVE", skip_env or "")
    state = SimpleNamespace(
        tp=tp,
        last_collective=last_collective or {},
        current_comm_pct=lambda: comm_pct,
        last_trace_analyze={"hot_kernels_top15": []} if analysis else {},
    )
    fake = SimpleNamespace(
        shared_state=state,
        COLLECTIVE_COMM_PCT_FLOOR=KernelPhase.COLLECTIVE_COMM_PCT_FLOOR,
    )
    return KernelPhase._collective_required_before_kernel_opt(fake)


def test_gate_opens_for_multi_gpu_with_exposed_comm():
    assert _gate(tp=8, comm_pct=5.0) is True


@pytest.mark.parametrize("tp", [0, 1])
def test_gate_closed_below_tp2(tp):
    """A single rank issues no collective at all."""
    assert _gate(tp=tp) is False


def test_gate_closed_when_comm_is_overlapped():
    """Communication hidden behind compute is not worth a tuning round."""
    assert _gate(comm_pct=0.2) is False


def test_gate_closed_without_a_roofline_snapshot():
    assert _gate(comm_pct=None) is False


@pytest.mark.parametrize("status", ["ok", "complete", "kept"])
def test_gate_is_idempotent_after_a_completed_run(status):
    assert _gate(last_collective={"status": status}) is False


def test_gate_reopens_after_a_failed_run():
    assert _gate(last_collective={"status": "failed"}) is True


# --- A skip is scoped to its analysis, not to the session --------------------


def _gate_with_analysis(candidates_path: str, last_collective: dict) -> bool:
    state = SimpleNamespace(
        tp=8,
        last_collective=last_collective,
        current_comm_pct=lambda: 5.0,
        last_trace_analyze={"hot_kernels_top15": [], "candidates_path": candidates_path},
    )
    fake = SimpleNamespace(
        shared_state=state,
        COLLECTIVE_COMM_PCT_FLOOR=KernelPhase.COLLECTIVE_COMM_PCT_FLOOR,
    )
    return KernelPhase._collective_required_before_kernel_opt(fake)


def test_skip_is_terminal_for_the_analysis_that_produced_it():
    """Re-deciding the same analysis on every KERNEL re-entry is noise."""
    assert _gate_with_analysis("/run/a/kernel_candidates.json",
                               {"status": "skipped", "analysis_key": "/run/a/kernel_candidates.json"}) is False


def test_skip_does_not_block_a_later_analysis():
    """Nothing clears last_collective, so an unscoped skip would lock the lane
    out for the whole session even after a new trace exposes a collective."""
    assert _gate_with_analysis("/run/b/kernel_candidates.json",
                               {"status": "skipped", "analysis_key": "/run/a/kernel_candidates.json"}) is True


def test_skip_without_an_analysis_key_does_not_block():
    assert _gate_with_analysis("/run/a/kernel_candidates.json", {"status": "skipped"}) is True


def test_gate_closed_before_any_trace_analysis():
    """Candidate selection reads the analysis, so a skip recorded before one
    exists would wrongly become terminal."""
    assert _gate(analysis=False) is False


def test_gate_respects_the_kill_switch(monkeypatch):
    assert _gate(skip_env="1", monkeypatch=monkeypatch) is False


# --- Registration -------------------------------------------------------------


def test_handler_is_registered():
    assert "run_collective" in KERNEL_REQUEST_HANDLERS


def test_lane_is_not_exposed_to_the_llm():
    """Deterministic gate => Coordinator-only, same posture as fusion."""
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        FULL_ENABLED_ACTIONS,
        KERNEL_AGENT_OWNED_ACTIONS,
    )

    assert "collective" not in KERNEL_AGENT_OWNED_ACTIONS
    assert "collective" not in FULL_ENABLED_ACTIONS
