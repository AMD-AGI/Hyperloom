# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a re-nomination may do to a terminal record already in the queue.

The fusion workspace is a constant task_id and patch files are named after the
pattern, so a re-discovered recipe yields a byte-identical
``(source_file, artifact_path)`` -- the key ``enqueue_nominated_patch``
collapses on. The re-enqueue therefore lands on records that already reached a
terminal status, and the two classes of terminal must part ways there:
``dispatch_failed`` is a fault worth retrying, while ``integrated`` and
``rejected`` are settled verdicts that must keep their status so
``evict_terminal`` -- the queue's only deletion point -- can still reap them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.kernel import _kernel_decisions as kd
from hyperloom.orchestrator.kernel import patch_landing as pl


_VERDICTS = ["integrated", "rejected"]


def _patch(kernel: str = "attn", *, micro: float = 1.0) -> SimpleNamespace:
    """A nominated sibling whose artifact path is stable across rounds."""
    return SimpleNamespace(
        kernel_name=f"{kernel}_kernel",
        patch_path=f"/ws/runs/fusion/kernel_entry_fusion/fusion_{kernel}.patch",
        target_file=f"/repo/{kernel}.py",
        env_flag="HL_FUSE_ATTN",
        micro_speedup=micro,
        snapshot_dir="",
        kernel_repo="/repo",
        base_commit="deadbeef",
    )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        pending_kernel_integrations={},
        kernel_opt_task_attempts={},
        kernel_integrate_attempts={},
        optimization_stack=[],
        rejected_kernel_ids=[],
        last_trace_analyze={},
    )


def _settled(status: str, *, kernel: str = "attn", micro: float = 1.0) -> tuple[Any, str]:
    """A queue holding one sibling that already reached ``status``."""
    state = _state()
    record = kd.enqueue_nominated_patch(state, patch=_patch(kernel, micro=micro), lane="fusion")
    assert record is not None
    record["status"] = status
    return state, str(record["integration_id"])


# --- the two classes of terminal -------------------------------------------
def test_the_terminal_statuses_split_into_verdicts_and_one_fault() -> None:
    """One spelling of the partition, so a new terminal status must be classified."""
    assert set(pl.VERDICT_STATUSES) == set(_VERDICTS)
    assert pl.VERDICT_STATUSES | {"dispatch_failed"} == pl.TERMINAL_STATUSES


# --- verdict-class terminals are not re-litigated ----------------------------
@pytest.mark.parametrize("verdict", _VERDICTS)
def test_a_settled_verdict_is_not_revived_to_pending(verdict: str) -> None:
    state, integration_id = _settled(verdict)
    kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion")
    assert state.pending_kernel_integrations[integration_id]["status"] == verdict


@pytest.mark.parametrize("verdict", _VERDICTS)
def test_a_settled_verdict_keeps_the_micro_it_was_judged_on(verdict: str) -> None:
    """evict_terminal ranks post-mortem retention by micro; a settled record's
    number is evidence for its verdict, not a slot for a fresh measurement."""
    state, integration_id = _settled(verdict, micro=1.2)
    kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion")
    assert state.pending_kernel_integrations[integration_id]["micro_speedup"] == 1.2


@pytest.mark.parametrize("verdict", _VERDICTS)
def test_re_offering_a_settled_verdict_queues_nothing(verdict: str) -> None:
    """Both callers count a non-None return as a sibling queued for dispatch."""
    state, _ = _settled(verdict)
    assert kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion") is None


@pytest.mark.parametrize("verdict", _VERDICTS)
def test_re_offering_a_settled_verdict_is_logged(verdict: str, caplog: Any) -> None:
    state, _ = _settled(verdict)
    with caplog.at_level("INFO", logger=kd.log.name):
        kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion")
    assert verdict in caplog.text and "attn_kernel" in caplog.text


@pytest.mark.parametrize("verdict", _VERDICTS)
def test_a_re_offered_verdict_survives_ten_cycles_as_a_verdict(verdict: str, monkeypatch: Any) -> None:
    """A verdict is stable across macro cycles: repeated re-offers of the same
    recipe neither change its status nor add a second record for it."""
    monkeypatch.setenv("HL_KERNEL_PATCH_BUDGET", "1")
    state, integration_id = _settled(verdict, micro=1.0)
    for _ in range(10):
        kd._ensure_kernel_task_state(state)
        kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion")
    assert sorted(state.pending_kernel_integrations) == [integration_id]
    assert state.pending_kernel_integrations[integration_id]["status"] == verdict


@pytest.mark.parametrize("verdict", _VERDICTS)
def test_a_re_offered_verdict_is_still_reaped_once_over_the_cap(verdict: str, monkeypatch: Any) -> None:
    """Keeping the verdict keeps the record eligible for the queue's only
    deletion point, so a re-offered sibling is still reaped like any terminal."""
    monkeypatch.setenv("HL_KERNEL_PATCH_BUDGET", "1")
    state, integration_id = _settled(verdict, micro=1.0)
    for _ in range(10):
        kd._ensure_kernel_task_state(state)
        kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion")
    # budget=1 * TERMINAL_RETENTION_MULTIPLE=2 => two stronger terminals push the
    # re-offered record past the post-mortem cap, and it is the weakest.
    survivors = []
    for filler in ("gemm", "norm"):
        strong = kd.enqueue_nominated_patch(state, patch=_patch(filler, micro=8.0), lane="fusion")
        strong["status"] = "integrated"
        survivors.append(str(strong["integration_id"]))
    kd._ensure_kernel_task_state(state)
    assert sorted(state.pending_kernel_integrations) == sorted(survivors)
    assert integration_id not in survivors


# --- fault-class terminals keep their retry ----------------------------------
def test_a_dispatch_failure_is_revived_for_another_attempt() -> None:
    """The drain crashed, so the patch never got a verdict; retrying is the point."""
    state, integration_id = _settled("dispatch_failed")
    revived = kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion")
    assert revived is not None
    assert state.pending_kernel_integrations[integration_id]["status"] == "pending"
    assert state.pending_kernel_integrations[integration_id]["micro_speedup"] == 9.0


def test_a_revived_dispatch_failure_reaches_the_drain_again() -> None:
    state, integration_id = _settled("dispatch_failed")
    kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion")
    dispatched = [str(row["integration_id"]) for row in kd.pending_kernel_integration_records(state)]
    assert dispatched == [integration_id]


def test_a_settled_verdict_is_held_back_by_the_pending_gate() -> None:
    """Only the status gate is under test here; the permanent
    ``kernel_integrate_attempts`` ledger is what really carries idempotency."""
    state, _ = _settled("rejected")
    kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion")
    assert [str(row["integration_id"]) for row in kd.pending_kernel_integration_records(state)] == []


# --- the live path is untouched ----------------------------------------------
def test_a_pending_record_still_refreshes_its_mutable_fields() -> None:
    state, integration_id = _settled("pending")
    refreshed = kd.enqueue_nominated_patch(state, patch=_patch(micro=9.0), lane="fusion")
    assert refreshed is not None
    assert state.pending_kernel_integrations[integration_id]["micro_speedup"] == 9.0
    assert sorted(state.pending_kernel_integrations) == [integration_id]
