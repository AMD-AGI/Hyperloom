# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior-lock tests for optimize exit-code semantics: multi-node topology gates exit 2, and a session already held exits 3 (SESSION_BUSY_EXIT_CODE)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

import hyperloom.inference_optimizer.cli as ocli
from hyperloom.inference_optimizer.session.lock import SessionLock
from hyperloom.orchestrator.state.shared_state import SharedState


def _record_terminal_writes(monkeypatch) -> list[str]:
    """Capture every terminal artifact the close-out writes, in order."""
    order: list[str] = []
    for name, label in (
        ("write_minimal_final_json", "final_json"),
        ("write_breakdown_json", "breakdown"),
        ("write_minimal_final_report", "final_md"),
        ("package_session_artifacts", "package"),
    ):
        monkeypatch.setattr(
            f"hyperloom.inference_optimizer.breakdown.{name}",
            lambda *_a, _label=label, **_kw: order.append(_label),
        )
    return order


def test_resumable_restart_writes_no_terminal_artifacts(tmp_path: Path, monkeypatch) -> None:
    order = _record_terminal_writes(monkeypatch)

    ocli._write_cli_terminal_artifacts(
        tmp_path,
        SharedState(session_id="s"),
        "supervisor_restart_requested",
    )

    assert order == []


def test_terminal_artifacts_keep_the_existing_write_order(tmp_path: Path, monkeypatch) -> None:
    order = _record_terminal_writes(monkeypatch)

    ocli._write_cli_terminal_artifacts(tmp_path, SharedState(session_id="s"), "signal")

    assert order == ["final_json", "breakdown", "final_md", "package"]


def test_completed_close_still_gets_its_close_out_package(tmp_path: Path, monkeypatch) -> None:
    """The sequencer wrote the reports; the package is the session's, not the sequencer's."""
    order = _record_terminal_writes(monkeypatch)

    state = SharedState(session_id="s", close_sequence_done=True)
    ocli._write_cli_terminal_artifacts(tmp_path, state, "signal")

    assert order == ["final_json", "package"]


def test_multinode_tp_exceeds_total_gpus_exits_2() -> None:
    """Gate 1: TP larger than nodes*gpus_per_node fails fast with exit code 2."""
    # nodes=2, gpus_per_node=1 -> total_gpus=2 < tp=4.
    args = argparse.Namespace(nodes=2, tp=4, ep=1, gpus_per_node=1)
    with pytest.raises(SystemExit) as exc:
        asyncio.run(ocli._run_optimize(args))
    assert exc.value.code == 2


def test_multinode_ep_exceeds_tp_exits_2() -> None:
    """Gate 2: EP greater than TP fails fast with exit code 2."""
    # total_gpus=16 >= tp=2 so gate 1 passes; ep=4 > tp=2 trips gate 2.
    args = argparse.Namespace(nodes=2, tp=2, ep=4, gpus_per_node=8)
    with pytest.raises(SystemExit) as exc:
        asyncio.run(ocli._run_optimize(args))
    assert exc.value.code == 2


def test_session_busy_exits_with_session_busy_code(tmp_path: Path) -> None:
    """A second optimizer on a live-locked session exits SESSION_BUSY_EXIT_CODE (3)."""
    held = SessionLock(tmp_path)
    held.acquire()
    try:
        with pytest.raises(SystemExit) as exc:
            ocli._acquire_session_lock_or_exit(tmp_path)
        assert exc.value.code == ocli.SESSION_BUSY_EXIT_CODE == 3
    finally:
        held.release()
