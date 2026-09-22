# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shipped benchmark harnesses preserve the requested measurement mode."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


_HARNESSES = sorted((Path(__file__).resolve().parents[1] / "data" / "examples").glob("*/graph_harness.py"))
_EXPLICIT_EAGER_HARNESSES = [path for path in _HARNESSES if path.parent.name != "triton2flydsl-mxfp8-grouped-gemm"]


class _Stream:
    def wait_stream(self, other):
        pass


class _Event:
    def __init__(self, *, enable_timing):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 0.25


class _Cuda:
    available = True
    capture_error = None
    replay_error = None
    replay_step = None
    replays = 0
    captures = 0
    Stream = _Stream
    Event = _Event

    def is_available(self):
        return self.available

    def current_stream(self):
        return _Stream()

    def stream(self, stream):
        return nullcontext()

    def synchronize(self):
        pass

    def CUDAGraph(self):
        return SimpleNamespace(replay=self._replay)

    def graph(self, graph):
        self.captures += 1
        if self.capture_error is not None:
            raise self.capture_error
        return nullcontext()

    def _replay(self):
        if self.replay_error is not None:
            raise self.replay_error
        self.replays += 1
        if self.replay_step is not None:
            self.replay_step()


def _load_harness(path, monkeypatch):
    cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    spec = importlib.util.spec_from_file_location("_example_graph_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, cuda


@pytest.fixture(params=_HARNESSES, ids=lambda path: path.parent.name)
def harness(request, monkeypatch):
    return _load_harness(request.param, monkeypatch)


def test_graph_benchmark_returns_replay_timings(harness):
    module, cuda = harness
    output = {"value": 0}

    def step():
        output["value"] = 42

    def dirty():
        output["value"] = 0

    cuda.replay_step = step
    result = module.cuda_graph_bench(step, warmup=1, iters=2, dirty=dirty, verify=lambda: output["value"] == 42)

    assert result["mode"] == "cudagraph"
    assert result["times_ms"] == [0.25, 0.25]
    assert result["median_ms"] == 0.25
    assert cuda.replays >= 2


@pytest.mark.parametrize("stage", ["capture", "replay"])
@pytest.mark.parametrize("error_type", [RuntimeError, PermissionError])
def test_graph_errors_propagate_without_eager_timings(harness, stage, error_type):
    module, cuda = harness
    error = error_type(f"{stage} failed")
    setattr(cuda, f"{stage}_error", error)

    with pytest.raises(error_type, match=f"{stage} failed") as raised:
        module.cuda_graph_bench(lambda: None, warmup=1, iters=2)

    assert raised.value is error


def test_invalid_replay_is_rejected(harness):
    module, _cuda = harness

    with pytest.raises(RuntimeError, match="graph replay did not"):
        module.cuda_graph_bench(lambda: None, warmup=1, iters=2, dirty=lambda: None, verify=lambda: False)


def test_missing_gpu_fails_before_running_the_kernel(harness):
    module, cuda = harness
    cuda.available = False
    calls = []

    with pytest.raises(RuntimeError, match="no GPU available"):
        module.cuda_graph_bench(lambda: calls.append("ran"), warmup=1, iters=2)

    assert calls == []


@pytest.mark.parametrize("path", _EXPLICIT_EAGER_HARNESSES, ids=lambda path: path.parent.name)
def test_explicit_eager_mode_remains_available(path, monkeypatch):
    module, cuda = _load_harness(path, monkeypatch)
    cuda.capture_error = RuntimeError("capture unavailable")
    calls = []

    result = module.cuda_graph_bench(lambda: calls.append("ran"), warmup=1, iters=2, capture=False)

    assert result["mode"] == "eager (capture disabled)"
    assert result["times_ms"] == [0.25, 0.25]
    assert calls == ["ran", "ran", "ran"]
    assert cuda.captures == 0
    assert cuda.replays == 0
