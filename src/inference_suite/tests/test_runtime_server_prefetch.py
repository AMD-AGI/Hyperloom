# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A started server's checkpoint is read alongside its boot, the reads end when the boot does, ready or not, and what
they did is left in the round's run_dir."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperloom_common.weight_prefetch import WEIGHT_PREFETCH_ENV
from inference_suite.contracts.layout import WEIGHT_PREFETCH
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

    def write_outcome(self, path: Path) -> None:
        self.calls.append(f"outcome {path.name}")


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    seen: list[str] = []

    def start(model_path, environ):
        assert environ is os.environ, "the round's own environment, where the sandbox's switch is"
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
    assert calls == [f"start {config.server.model_path}", "stop", f"outcome {WEIGHT_PREFETCH}"], "before the round"
    lifecycle.release()
    assert len(calls) == 3


def test_the_reads_end_when_the_server_dies_before_it_is_ready(tmp_path, calls) -> None:
    rig = _Rig(tmp_path, "exit")
    assert isinstance(rig.lifecycle(rig.config("r1")).acquire(), Stop)
    assert calls[1:] == ["stop", f"outcome {WEIGHT_PREFETCH}"]


def test_a_prefetch_that_is_off_changes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(WEIGHT_PREFETCH_ENV, raising=False)
    rig = _Rig(tmp_path)
    lifecycle = rig.lifecycle(rig.config("r1"))
    assert isinstance(lifecycle.acquire(), Server)
    lifecycle.release()
    assert not (tmp_path / "r1" / WEIGHT_PREFETCH).exists(), "off: no prefetch, no record"


def test_a_prefetch_that_is_on_leaves_what_it_did_in_the_run_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(WEIGHT_PREFETCH_ENV, "1")
    rig = _Rig(tmp_path)
    config = rig.config("r1")
    lifecycle = rig.lifecycle(config)
    assert isinstance(lifecycle.acquire(), Server)
    lifecycle.release()

    record = json.loads((tmp_path / "r1" / WEIGHT_PREFETCH).read_text(encoding="utf-8"))
    assert record["model_dir"] == config.server.model_path
    assert record["skipped"] and record["read_bytes"] == 0 and record["unreadable"] == 0
