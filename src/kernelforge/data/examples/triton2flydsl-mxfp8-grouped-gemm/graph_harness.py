"""CUDA/HIP graph timing for replay-safe GPU callables."""

from __future__ import annotations

import statistics
from typing import Callable

import torch


class _CaptureInvalid(RuntimeError):
    """Raised when graph replay does not reproduce the expected outputs."""


def _time_eager(step: Callable[[], object], iters: int) -> list[float]:
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _time_graph(
    step: Callable[[], object],
    iters: int,
    dirty: Callable[[], None] | None,
    verify: Callable[[], bool] | None,
) -> list[float]:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    if dirty is not None and verify is not None:
        dirty()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        if not verify():
            raise _CaptureInvalid("graph replay did not reproduce correct outputs")

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def cuda_graph_bench(
    step: Callable[[], object],
    *,
    warmup: int = 10,
    iters: int = 30,
    dirty: Callable[[], None] | None = None,
    verify: Callable[[], bool] | None = None,
) -> dict:
    """Warm, capture, validate, and time one replay-safe GPU step."""
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU available")

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(max(1, warmup)):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    try:
        times = _time_graph(step, max(1, iters), dirty, verify)
        mode = "cudagraph"
    except Exception as error:  # noqa: BLE001 - report an honest eager fallback
        times = _time_eager(step, max(1, iters))
        mode = f"eager ({type(error).__name__}: {error})"

    times = [value for value in times if value > 0]
    return {
        "mode": mode,
        "times_ms": times,
        "median_ms": statistics.median(times) if times else None,
    }
