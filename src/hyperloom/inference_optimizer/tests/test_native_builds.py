# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The entry point opens the builds of a framework whose server is built from source and hands them to integration,
and keeps the options that server prints."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer import native_builds
from hyperloom.inference_optimizer.cli import bootstrap
from hyperloom.inference_optimizer.cli.bootstrap import _install_executors
from hyperloom.inference_optimizer.native_builds import OptionCoverage
from hyperloom.orchestrator.state.tests._session_base import freeze_session_base
from hyperloom.plugs import Implementation, PlugHost, PlugRegistry, ResolvedManifest, Socket, activated
from hyperloom.plugs.manifest import SocketSetting
from hyperloom.plugs.native_builds import NATIVE_BUILDS_SOCKET

_HELP = """\
-c,    --ctx-size N                     size of the prompt context (default: 0)
--keep N                                number of tokens to keep from the initial prompt
"""


@contextmanager
def _hosting(engine: object) -> Iterator[None]:
    """A plug host whose ``native_builds`` socket ``engine`` fills, as the process's current one."""
    registry = PlugRegistry()
    registry.define(Socket(NATIVE_BUILDS_SOCKET, object))
    registry.register(Implementation(NATIVE_BUILDS_SOCKET, "stand-in", lambda context: engine))
    manifest = ResolvedManifest("test", {NATIVE_BUILDS_SOCKET: SocketSetting("stand-in", {}, "cli")})
    host = PlugHost(manifest, registry, session_dir=None)
    host.start()
    with activated(host):
        yield


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


def test_an_image_served_framework_builds_nothing_keeps_no_options_and_imports_no_engine_for_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engines = {f"hyperloom.measure.runner.native.{module}" for module in ("session_builds", "session_options")}
    for module in engines:
        monkeypatch.delitem(sys.modules, module, raising=False)

    coordinator = _install("sglang", tmp_path)

    assert coordinator.sub.executor_registry["integrate_patch"].native is None
    assert not (tmp_path / "native").exists() and list(tmp_path.glob("*options*")) == []
    assert native_builds.server_options(tmp_path, "sglang") is None
    assert native_builds.server_options(None, "llamacpp") is None
    assert not engines & set(sys.modules)


def test_the_builds_and_the_options_are_the_ones_the_current_plug_host_gives(tmp_path: Path) -> None:
    builds, kept = SimpleNamespace(checkout=tmp_path), SimpleNamespace(file=tmp_path / "options.json")
    opened = {(tmp_path, "llamacpp"): builds}
    engine = SimpleNamespace(
        open_builds=lambda session_dir, framework: opened[session_dir, framework],
        record_options=lambda session_dir, framework: opened[session_dir, framework] and kept.file,
        options={tmp_path: kept}.get,
    )

    with _hosting(engine):
        assert native_builds.start_candidate_builds(tmp_path, "llamacpp") is builds
        assert native_builds.start_candidate_builds(tmp_path, "sglang") is None
        assert native_builds.record_server_options(tmp_path, "llamacpp") == kept.file
        assert native_builds.record_server_options(tmp_path, "sglang") is None
        assert native_builds.server_options(tmp_path, "llamacpp") is kept


def _built_server(root: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The builds of a session whose stand-in server, first on ``PATH`` as an accepted build is, prints ``_HELP``."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (root / "help.txt").write_text(_HELP, encoding="utf-8")
    program = bin_dir / "llama-server"
    program.write_text(f'#!/bin/sh\n/bin/cat "{root / "help.txt"}"\n', encoding="utf-8")
    program.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    builds = SimpleNamespace(checkout=root)
    monkeypatch.setattr(bootstrap, "start_candidate_builds", lambda session_dir, framework: builds)
    return builds


def test_a_session_that_builds_its_server_keeps_the_options_that_server_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = _built_server(tmp_path, monkeypatch)
    session = tmp_path / "session"
    session.mkdir()
    freeze_session_base(session, ("-c", "4096"))

    coordinator = _install("llamacpp", session)

    assert coordinator.sub.executor_registry["integrate_patch"].native is builds
    kept = native_builds.server_options(session, "llamacpp")
    assert kept is not None and kept.file.parent == session
    assert kept.coverage(()) == OptionCoverage(unset=1, missing=("--keep",), unknown=())
    assert kept.coverage(("--keep", "-c", "--nope")) == OptionCoverage(unset=1, missing=(), unknown=("--nope",))
