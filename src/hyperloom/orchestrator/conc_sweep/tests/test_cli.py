# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``sweep_session`` runs the sweep with its session pinned for the rounds and packages the session after the report."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.orchestrator.actions.executors.grid._grid_inputs import _resolve_session_dir
from hyperloom.orchestrator.conc_sweep import cli
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, "")
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR)
    path = tmp_path / "workspace" / "model" / "20261124T000000Z-abcd1234"
    (path / "reports").mkdir(parents=True)
    SharedState(session_id="abcd1234").save(path)
    return path


def test_a_standalone_sweep_resolves_its_rounds_to_its_own_session(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved: list[Path] = []

    async def run_conc_sweep(path: Path, **_: Any) -> dict[str, Any]:
        resolved.append(_resolve_session_dir())
        return {"status": "succeeded"}

    monkeypatch.setattr(cli, "run_conc_sweep", run_conc_sweep)
    monkeypatch.setattr(cli, "package_session_artifacts", lambda *_, **__: None)

    asyncio.run(cli.sweep_session(session_dir))

    assert [path.resolve() for path in resolved] == [session_dir.resolve()]


def test_a_relative_session_dir_is_pinned_as_an_absolute_one(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run_conc_sweep(path: Path, **_: Any) -> dict[str, Any]:
        return {"status": "succeeded"}

    monkeypatch.setattr(cli, "run_conc_sweep", run_conc_sweep)
    monkeypatch.setattr(cli, "package_session_artifacts", lambda *_, **__: None)
    monkeypatch.chdir(session_dir.parent)

    asyncio.run(cli.sweep_session(Path(session_dir.name)))

    assert os.environ[ENV_CURRENT_SESSION_DIR] == str(session_dir.resolve())


def test_the_session_is_packaged_again_once_the_report_is_written(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, ...]] = []

    async def run_conc_sweep(path: Path, **_: Any) -> dict[str, Any]:
        events.append(("sweep",))
        return {"status": "succeeded"}

    def package(path: Path, *, session_id: str) -> Path:
        events.append(("package", str(path), session_id))
        return path / "pkg.zip"

    monkeypatch.setattr(cli, "run_conc_sweep", run_conc_sweep)
    monkeypatch.setattr(cli, "package_session_artifacts", package)

    asyncio.run(cli.sweep_session(session_dir))

    assert events == [("sweep",), ("package", str(session_dir), "abcd1234")]


def test_the_command_sweeps_the_operators_points_with_no_deadline(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    async def run_conc_sweep(path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setattr(cli, "run_conc_sweep", run_conc_sweep)
    monkeypatch.setattr(cli, "package_session_artifacts", lambda *_, **__: None)

    assert cli.main([str(session_dir), "--conc-sweep-concs", "8,2"]) == 0

    assert calls == [{"concs": (2, 8), "deadline_unix": None}]
