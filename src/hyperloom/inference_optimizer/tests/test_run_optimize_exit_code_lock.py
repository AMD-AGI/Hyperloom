# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior-lock tests for optimize exit-code semantics: multi-node topology gates exit 2, and a session already held exits 3 (SESSION_BUSY_EXIT_CODE)."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

import hyperloom.inference_optimizer.cli as ocli
from hyperloom.inference_optimizer.session.lock import SessionLock
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan
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


def _backends() -> dict[str, Backend]:
    heartbeat = Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})
    plan = ScriptedPlan(turns=[], default_intent=heartbeat)
    return {name: MockBackend(plan, name=name) for name in ("orchestration", "critic", "robustness")}


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


@pytest.mark.asyncio
async def test_resumable_classification_survives_state_save_failure(tmp_path: Path, monkeypatch) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    SharedState(session_id="s").save(session_dir)
    (session_dir / "manifest.json").write_text(
        '{"schema_version": 4, "session_id": "s"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setattr(
        ocli,
        "clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )
    monkeypatch.setattr(ocli, "_preflight", lambda _args: ("", ""))
    monkeypatch.setattr(ocli, "_resolve_models_for_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ocli, "_build_backends", lambda **_kwargs: _backends())
    monkeypatch.setattr(ocli, "spawn_supervisor", lambda *_args, **_kwargs: None)

    original_run = Coordinator.run

    async def _run_then_fail_resumable_save(self, **kwargs):
        self._resumed_from = {"is_resume": False, "rebuilt": True}
        self._signals = SimpleNamespace(
            received={signal.SIGHUP},
            close=lambda: frozenset({signal.SIGHUP}),
        )
        self._stop.set()
        real_save = self.shared_state.save

        def _fail_save(path) -> None:
            if self.stop_classification == "supervisor_restart_requested":
                raise OSError("disk full")
            real_save(path)

        self.shared_state.save = _fail_save
        kwargs["install_signal_handlers"] = False
        return await original_run(self, **kwargs)

    monkeypatch.setattr(Coordinator, "run", _run_then_fail_resumable_save)
    order = _record_terminal_writes(monkeypatch)
    reasons: list[str | None] = []
    write_terminal = ocli._write_cli_terminal_artifacts

    def _capture_terminal_reason(path, state, stop_reason) -> None:
        reasons.append(stop_reason)
        write_terminal(path, state, stop_reason)

    monkeypatch.setattr(ocli, "_write_cli_terminal_artifacts", _capture_terminal_reason)
    args = ocli._build_parser().parse_args(
        [
            "optimize",
            "--resume-from",
            str(session_dir),
            "--critic-mock",
            "--robustness-mock",
            "--no-kernel",
            "--no-framework-agent",
            "--degraded-kb",
            "--degraded-pr",
            "--research-lane-capacity",
            "0",
            "--max-ticks",
            "1",
        ]
    )

    with pytest.raises(OSError, match="disk full"):
        await ocli._run_optimize(args)

    assert reasons == ["supervisor_restart_requested"]
    assert order == []


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
