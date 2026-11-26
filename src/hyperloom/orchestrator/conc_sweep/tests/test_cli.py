# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``sweep_session`` runs the sweep with its session pinned for the rounds and packages the session after the report."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.session.manifest import manifest_path
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.orchestrator.actions.executors.grid._grid_inputs import _resolve_session_dir
from hyperloom.orchestrator.conc_sweep import cli
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.tests._fixtures import MI325X_CARD, MI355X_CARD, publish_card
from hyperloom_common import gpu_card


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


def test_the_command_sweeps_the_operators_points(session_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def run_conc_sweep(path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setattr(cli, "run_conc_sweep", run_conc_sweep)
    monkeypatch.setattr(cli, "package_session_artifacts", lambda *_, **__: None)
    _manifest(session_dir, gpu_card=dataclasses.asdict(MI325X_CARD))

    assert cli.main([str(session_dir), "--conc-sweep-concs", "8,2"]) == 0

    assert calls == [{"concs": (2, 8)}]


def _manifest(session_dir: Path, **keys: Any) -> None:
    manifest_path(session_dir).write_text(json.dumps(keys), encoding="utf-8")


def test_a_sweep_run_on_its_own_reads_the_card_its_session_archived(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep is comparable with its session on the same hardware, so the archived card outranks this machine's."""
    publish_card(monkeypatch, MI355X_CARD)
    _manifest(session_dir, gpu_card=dataclasses.asdict(MI325X_CARD))

    cli._publish_session_card(session_dir)

    assert gpu_card.session_card() == MI325X_CARD


def test_a_sweep_of_a_session_that_archived_no_card_reads_this_machines(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish_card(monkeypatch, MI325X_CARD)
    _manifest(session_dir, gpu_card=None)
    monkeypatch.setattr(cli, "read_session_card", lambda nodes: MI355X_CARD)

    cli._publish_session_card(session_dir)

    assert gpu_card.session_card() == MI355X_CARD
