# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The entry point opens the builds of a framework whose server is built from source and hands them to integration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer import native_builds
from hyperloom.inference_optimizer.cli.bootstrap import _install_executors
from hyperloom.plugs import (
    Implementation,
    PlugHost,
    PlugRegistry,
    Socket,
    activated,
    builtin_registry,
    resolve_manifest,
)
from hyperloom.plugs.native_builds import NATIVE_BUILDS_SOCKET


def _registry_of(engine: object) -> PlugRegistry:
    """A registry whose ``native_builds`` socket ``engine`` fills."""
    registry = PlugRegistry()
    registry.define(Socket(NATIVE_BUILDS_SOCKET, object, default="stand-in"))
    registry.register(Implementation(NATIVE_BUILDS_SOCKET, "stand-in", lambda context: engine))
    return registry


class _Sub:
    def __init__(self) -> None:
        self.executor_registry: dict[str, object] = {}

    def register_executor(self, kind: str, fn: object) -> None:
        self.executor_registry[kind] = fn


def _install(framework: str, session_dir: Path) -> SimpleNamespace:
    coordinator = SimpleNamespace(sub=_Sub(), shared_state=SimpleNamespace(framework=framework))
    args = argparse.Namespace(research_lane_capacity=0, compare_against_gpu=None)
    _install_executors(coordinator, args, session_dir=session_dir, knowledge_plane=None)
    return coordinator


def test_a_session_whose_framework_server_cannot_be_built_from_source_stops_before_any_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("FRAMEWORK_REPO_PATH", raising=False)

    with pytest.raises(SystemExit) as stopped:
        _install("llamacpp", tmp_path)

    assert stopped.value.code == 2
    assert "builds its server from its source: pass --framework-path" in capsys.readouterr().err


def test_an_image_served_framework_builds_nothing_and_imports_no_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = "hyperloom.measure.runner.native.session_builds"
    monkeypatch.delitem(sys.modules, engine, raising=False)

    coordinator = _install("sglang", tmp_path)

    assert coordinator.sub.executor_registry["integrate_patch"].native is None
    assert not (tmp_path / "native").exists()
    assert engine not in sys.modules


def test_the_builds_are_opened_by_a_host_of_their_own_that_no_other_framework_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = SimpleNamespace(checkout=tmp_path)
    opened = {(tmp_path, "llamacpp"): builds}
    engine = SimpleNamespace(open_builds=lambda session_dir, framework: opened[session_dir, framework])
    registries: list[PlugRegistry] = []
    monkeypatch.setattr(
        native_builds, "builtin_registry", lambda: registries.append(_registry_of(engine)) or registries[-1]
    )
    session = builtin_registry()
    host = PlugHost(resolve_manifest(session), session, session_dir=None)
    host.start()

    with activated(host):
        assert native_builds.start_candidate_builds(tmp_path, "sglang") is None
        assert registries == []
        assert native_builds.start_candidate_builds(tmp_path, "llamacpp") is builds
    assert not host.has(NATIVE_BUILDS_SOCKET)
