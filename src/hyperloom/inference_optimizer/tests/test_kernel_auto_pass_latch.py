# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A kernel nobody claimed must not keep the KERNEL phase open forever."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.phases import machine_state as ms


class _State:
    """The narrow surface ``kernel_work_pending`` probes."""

    def __init__(self, *, untried: list[str] | None = None, cycle: int = 0) -> None:
        self.macro_cycle = cycle
        self.kernel_auto_pass_cycle: int | None = None
        self.collective_only_mode = False
        self.has_keep_pending_integrate = False
        self.rejected_kernel_ids: list[str] = []
        self.optimization_stack: list[dict[str, Any]] = []
        self.kernel_opt_task_attempts: dict[str, Any] = {}
        self._untried = untried or []

    def untried_hot_reusable_kernels(self) -> list[str]:
        return list(self._untried)


def test_an_untried_kernel_keeps_the_phase_open_before_a_pass_runs() -> None:
    """The existing guard: real outstanding work must not be skipped."""
    assert ms.kernel_work_pending(_State(untried=["k001"])) is True


def test_the_same_kernel_stops_holding_the_phase_once_a_pass_completed() -> None:
    """A nominator that looked and passed leaves no ledger row to clear."""
    state = _State(untried=["k001"])
    ms.mark_kernel_auto_pass_complete(state)
    assert ms.kernel_work_pending(state) is False


def test_the_latch_is_scoped_to_its_macro_cycle() -> None:
    """Entering the next cycle retires it without anyone clearing it."""
    state = _State(untried=["k001"])
    ms.mark_kernel_auto_pass_complete(state)
    assert ms.kernel_auto_pass_complete(state) is True
    state.macro_cycle += 1
    assert ms.kernel_auto_pass_complete(state) is False
    assert ms.kernel_work_pending(state) is True


def test_an_unset_latch_reads_as_incomplete() -> None:
    assert ms.kernel_auto_pass_complete(_State()) is False


@pytest.mark.parametrize("marker", ["x", None, object()])
def test_an_unusable_marker_reads_as_incomplete(marker: Any) -> None:
    """A corrupt marker must not silently let the phase exit early."""
    state = _State(untried=["k001"])
    state.kernel_auto_pass_cycle = marker
    assert ms.kernel_auto_pass_complete(state) is False
    assert ms.kernel_work_pending(state) is True


def test_the_latch_does_not_override_a_keep_awaiting_integrate() -> None:
    """That is genuine outstanding work, not a kernel nobody claimed."""
    state = _State(untried=[])
    state.has_keep_pending_integrate = True
    ms.mark_kernel_auto_pass_complete(state)
    assert ms.kernel_work_pending(state) is True


def test_the_latch_records_the_cycle_it_ran_in() -> None:
    state = _State(cycle=3)
    ms.mark_kernel_auto_pass_complete(state)
    assert state.kernel_auto_pass_cycle == 3


def test_marking_twice_in_a_cycle_is_idempotent() -> None:
    state = _State(cycle=2)
    ms.mark_kernel_auto_pass_complete(state)
    ms.mark_kernel_auto_pass_complete(state)
    assert state.kernel_auto_pass_cycle == 2


def test_a_missing_macro_cycle_defaults_to_zero() -> None:
    class _Bare:
        pass

    bare = _Bare()
    ms.mark_kernel_auto_pass_complete(bare)
    assert bare.kernel_auto_pass_cycle == 0
    assert ms.kernel_auto_pass_complete(bare) is True


def test_the_field_survives_a_shared_state_round_trip() -> None:
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState()
    assert state.kernel_auto_pass_cycle is None
    state.kernel_auto_pass_cycle = 4
    restored = SharedState.from_dict(state.to_dict())
    assert restored.kernel_auto_pass_cycle == 4
