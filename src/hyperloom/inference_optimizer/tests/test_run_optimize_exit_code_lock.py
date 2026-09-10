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


def _terminal_report_names(session_dir: Path) -> list[str]:
    return sorted(path.name for path in (session_dir / "reports").glob("final.*"))


def test_supervisor_restart_writes_no_cli_terminal_reports(tmp_path: Path) -> None:
    """A resumable watchdog stop must not run either CLI final-report writer."""
    ocli._write_cli_terminal_reports(tmp_path, SharedState(session_id="s"), "supervisor_restart_requested")

    assert _terminal_report_names(tmp_path) == []


def test_terminal_stop_writes_both_cli_terminal_reports(tmp_path: Path) -> None:
    """A terminal stop keeps the crash-safe reports the CLI has always written."""
    ocli._write_cli_terminal_reports(tmp_path, SharedState(session_id="s"), "signal")

    assert _terminal_report_names(tmp_path) == ["final.json", "final.md"]


def test_a_finished_close_sequence_keeps_the_cli_out_of_final_md(tmp_path: Path) -> None:
    """final.md is the sequencer's artifact; the CLI only stands in when the sequencer never wrote it."""
    ocli._write_cli_terminal_reports(tmp_path, SharedState(session_id="s", close_sequence_done=True), "signal")

    assert _terminal_report_names(tmp_path) == ["final.json"]


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
