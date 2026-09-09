"""Reusable CUDA/HIP graph timing harness for kernel micro-benchmarks."""

from __future__ import annotations

import statistics
from typing import Callable

import torch


class _CaptureInvalid(RuntimeError):
    """Raised when a captured graph does not reproduce a correct result."""


def _time_eager(step: Callable[[], object], iters: int) -> list[float]:
    """Per-iteration event timing WITHOUT graph capture (fallback path)."""
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
    """Capture ``step`` into a CUDA/HIP graph and time per-replay GPU execution."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    # Capture-validity guard: an uncaptured launch yields an empty graph whose replay is a fast no-op.
    if dirty is not None and verify is not None:
        dirty()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        if not verify():
            raise _CaptureInvalid(
                "graph replay did not recompute a correct result — the kernel was "
                "not captured (likely launched on a non-capture stream)"
            )

    # Replay warmups (steady state before timing).
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
    capture: bool = True,
    dirty: Callable[[], None] | None = None,
    verify: Callable[[], bool] | None = None,
) -> dict:
    """Benchmark ``step`` under CUDA/HIP graph replay (eager fallback)."""
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU available (torch.cuda.is_available() is False)")

    # Warm up on a side stream so JIT compile / autotune / workspace allocation complete BEFORE capture (those steps
    # are not capturable).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(max(1, warmup)):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    if capture:
        try:
            times = _time_graph(step, iters, dirty, verify)
            mode = "cudagraph"
        except Exception as e:  # noqa: BLE001 - fall back so a run always measures
            times = _time_eager(step, iters)
            mode = f"eager ({type(e).__name__}: {e})"
    else:
        times = _time_eager(step, iters)
        mode = "eager (capture disabled)"

    times = [t for t in times if t > 0]
    return {
        "mode": mode,
        "times_ms": times,
        "median_ms": statistics.median(times) if times else None,
        "mean_ms": statistics.mean(times) if times else None,
    }
