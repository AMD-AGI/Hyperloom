# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A started server's checkpoint is read alongside its boot, and the reads end when the boot does, ready or not."""

from __future__ import annotations

import os

import pytest

from inference_suite.runtime import server
from inference_suite.runtime.server import Server
from inference_suite.runtime.watch import Stop
from inference_suite.tests.test_runtime_server import _free_gpus, _fast_polls, _Rig  # noqa: F401 - fixtures

pytestmark = pytest.mark.skipif(not os.path.exists("/proc/net/tcp"), reason="needs Linux procfs")


class _Prefetch:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def stop(self) -> None:
        self.calls.append("stop")


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    seen: list[str] = []

    def start(model_path, environ):
        assert environ is os.environ, "the round's own environment, where the runner turned it on"
        seen.append(f"start {model_path}")
        return _Prefetch(seen)

    monkeypatch.setattr(server, "start_weight_prefetch", start)
    return seen


def test_the_checkpoint_is_read_while_the_server_boots_and_no_longer_once_it_is_ready(tmp_path, calls) -> None:
    rig = _Rig(tmp_path)
    config = rig.config("r1")
    lifecycle = rig.lifecycle(config)

    acquired = lifecycle.acquire()
    assert isinstance(acquired, Server)
    assert calls == [f"start {config.server.model_path}", "stop"], "stopped before the round goes on"
    lifecycle.release()
    assert calls[-1] == "stop" and len(calls) == 2


def test_the_reads_end_when_the_server_dies_before_it_is_ready(tmp_path, calls) -> None:
    rig = _Rig(tmp_path, "exit")
    assert isinstance(rig.lifecycle(rig.config("r1")).acquire(), Stop)
    assert calls[-1] == "stop" and len(calls) == 2


def test_a_prefetch_that_is_off_changes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(server, "start_weight_prefetch", lambda model_path, environ: None)
    rig = _Rig(tmp_path)
    lifecycle = rig.lifecycle(rig.config("r1"))
    assert isinstance(lifecycle.acquire(), Server)
    lifecycle.release()
