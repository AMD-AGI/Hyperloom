# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ReplayGpuBackend answers run_grid from an oracle socket; the process backend defaults to real."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors import explore
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.backends import (
    NO_GPU_ERROR,
    RealGpuBackend,
    ReplayGpuBackend,
    bind_backend_coordinator,
    ensure_gpu_backend,
    get_gpu_backend,
    install_gpu_backend,
)


class _Oracle:
    """Answers each request with the next scripted reply and keeps what it was asked."""

    def __init__(self, replies: list[dict]) -> None:
        self.asked: list[dict] = []
        ours, theirs = socket.socketpair()
        self.fd = theirs.detach()
        self._thread = threading.Thread(target=self._serve, args=(ours, replies), daemon=True)
        self._thread.start()

    def _serve(self, sock: socket.socket, replies: list[dict]) -> None:
        reader, writer = sock.makefile("r", encoding="utf-8"), sock.makefile("w", encoding="utf-8")
        for reply in replies:
            line = reader.readline()
            if not line:
                break
            self.asked.append(json.loads(line))
            writer.write(json.dumps(reply) + "\n")
            writer.flush()


def _coordinator(**state):
    defaults = {"baseline_tput": 100.0, "current_best": None, "baseline_accuracy": 0.9}
    return SimpleNamespace(shared_state=SimpleNamespace(**{**defaults, **state}))


def _reply(gain, drop=None):
    return {"outcome": "ESTIMATE", "gain_pct": gain, "accuracy_drop_pts": drop, "match": "dataset", "duration_sec": 0.0}


@pytest.fixture(autouse=True)
def _reset_backend():
    install_gpu_backend(RealGpuBackend())
    yield
    install_gpu_backend(RealGpuBackend())


def test_the_process_backend_is_real_unless_one_is_installed(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_GPU_BACKEND", raising=False)
    assert get_gpu_backend().name == "real"
    replay = install_gpu_backend(ReplayGpuBackend(oracle_fd=None))
    assert ensure_gpu_backend() is replay


def test_explore_benches_and_reads_accuracy_through_the_installed_backend(tmp_path: Path):
    oracle = _Oracle([_reply(10.0, drop=5.0)])
    install_gpu_backend(ReplayGpuBackend(oracle_fd=oracle.fd, time_scale=0.0))
    bind_backend_coordinator(_coordinator())
    grid = [GridVariant(name="v1", extra_server_args="--flag")]
    results = asyncio.run(
        explore.run_grid(base_yaml_path=tmp_path / "c.yaml", base_extra_args="", grid=grid, output_root=tmp_path)
    )
    assert [(r.status, r.output_throughput) for r in results] == [("succeeded", pytest.approx(110.0))]
    assert oracle.asked[0]["extra_server_args"] == "--flag"
    assert explore.parse_eval_results(tmp_path)["accuracy"] == pytest.approx(0.85)


def test_an_answer_is_kept_across_grids_and_its_launch_anchors_the_next(tmp_path: Path):
    oracle = _Oracle([_reply(10.0), _reply(5.0)])
    backend = ReplayGpuBackend(oracle_fd=oracle.fd, time_scale=0.0)
    backend.bind_coordinator(_coordinator())
    v1 = GridVariant(name="v1", extra_server_args="--a")
    first = asyncio.run(backend.run_grid(base_extra_args="", grid=[v1], output_root=tmp_path / "warmup"))
    again = asyncio.run(backend.run_grid(base_extra_args="", grid=[v1], output_root=tmp_path / "decision"))
    on_top = asyncio.run(
        backend.run_grid(
            base_extra_args="--a", grid=[GridVariant(name="v2", extra_server_args="--b")], output_root=tmp_path / "next"
        )
    )
    assert len(oracle.asked) == 2
    assert again[0] is first[0]
    assert on_top[0].output_throughput == pytest.approx(110.0 * 1.05)


def test_accuracy_is_the_baseline_less_the_drop_answered_for_the_slot(tmp_path: Path):
    oracle = _Oracle([_reply(3.0, drop=2.0)])
    backend = ReplayGpuBackend(oracle_fd=oracle.fd, time_scale=0.0)
    backend.bind_coordinator(_coordinator(baseline_accuracy=0.9))
    slot = tmp_path / "slot"
    asyncio.run(
        backend.run_grid(base_extra_args="", grid=[GridVariant(name="v1", extra_server_args="--a")], output_root=slot)
    )
    assert backend.parse_eval_results(slot)["accuracy"] == pytest.approx(0.88)
    assert backend.parse_eval_results(tmp_path / "other")["accuracy"] == pytest.approx(0.9)


def test_every_other_gpu_operation_fails_as_on_a_host_without_a_device():
    backend = ReplayGpuBackend(oracle_fd=None)
    with pytest.raises(RuntimeError, match="ROCm"):
        backend.launch(["vllm", "serve"])
    for op in (backend.apply_patch, backend.kernel_job, backend.profile):
        with pytest.raises(RuntimeError, match=NO_GPU_ERROR):
            asyncio.run(op())
