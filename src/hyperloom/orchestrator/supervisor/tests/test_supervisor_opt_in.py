# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The supervisor is opt-in: nothing is spawned unless the switch says so."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.supervisor.launcher import (
    SUPERVISOR_ENABLE_ENV,
    spawn_supervisor,
)


@pytest.fixture
def no_popen(monkeypatch):
    """Fail loudly if the launcher tries to start a process."""
    spawned: list[list[str]] = []

    def _popen(argv, **_kwargs):
        spawned.append(list(argv))
        raise AssertionError(f"supervisor should not have been spawned: {argv}")

    monkeypatch.setattr("hyperloom.orchestrator.supervisor.launcher.subprocess.Popen", _popen)
    return spawned


@pytest.mark.parametrize("env", [{}, {SUPERVISOR_ENABLE_ENV: "0"}, {SUPERVISOR_ENABLE_ENV: "false"}])
def test_not_spawned_without_an_opt_in(tmp_path: Path, no_popen, env) -> None:
    assert spawn_supervisor(tmp_path, session_sec=0, env=env) is None
    assert no_popen == []


def test_opt_in_reaches_the_spawn(tmp_path: Path, monkeypatch) -> None:
    argv_seen: list[list[str]] = []

    class _FakeProc:
        pid = 4242

    def _popen(argv, **_kwargs):
        argv_seen.append(list(argv))
        return _FakeProc()

    monkeypatch.setattr("hyperloom.orchestrator.supervisor.launcher.subprocess.Popen", _popen)

    proc = spawn_supervisor(tmp_path, session_sec=0, env={SUPERVISOR_ENABLE_ENV: "1"})

    assert proc is not None
    assert len(argv_seen) == 1
    assert "hyperloom.orchestrator.supervisor" in argv_seen[0]
