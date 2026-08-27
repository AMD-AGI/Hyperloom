"""Reusable CUDA/HIP graph timing harness for kernel micro-benchmarks.

Why this exists
---------------
A tiny kernel's wall time is dominated by HOST-side launch/dispatch overhead
(Python -> framework -> kernel launch), not by the GPU work itself. If the loop
benchmarks in eager mode, the agent ends up "optimizing" launch overhead, which
is meaningless — and real AMD serving runs these ops under a HIP graph anyway.

This harness captures ONE invocation of an operator into a CUDA/HIP graph and
then times graph *replays*. CUDA events bracket only the GPU stream timeline, so
the reported time is GPU execution, with the host replay-launch cost excluded —
the quantity that actually matters for a latency-bound kernel.

How to use it (operator-agnostic)
---------------------------------
Any operator plugs in by passing a zero-argument ``step`` closure that runs ONE
invocation on tensors the caller has ALREADY allocated. A graph replays the same
memory every time, so:

  * allocate all inputs (and, if you can, outputs) ONCE before calling;
  * ``step`` must be replay-safe: no host-side control flow that depends on the
    result, no per-call allocation that must differ between replays (internal
    allocations are fine — they are captured into the graph's private pool and
    reused on replay);
  * anything that must compile/autotune (e.g. JIT) happens during warmup, before
    capture — the harness warms up on a side stream first;
  * ``step`` must launch its kernel on the CURRENT stream. torch.cuda.graph
    captures a private capture stream; a kernel launched on the default/NULL
    stream (common for DSLs that manage their own stream) is NOT recorded, which
    yields a silently EMPTY graph that "replays" in a few microseconds
    regardless of problem size. See ``verify`` below to guard against this.

Capture-validity guard (``dirty`` / ``verify``)
-----------------------------------------------
Because an uncaptured launch produces a fast-but-empty graph, trusting the raw
replay time is dangerous. If the caller passes ``dirty`` and ``verify``, the
harness — after capture — corrupts the output via ``dirty()``, replays once, and
checks ``verify()``. If the replay did NOT recompute a correct result the graph
captured no real work, so the harness rejects it and falls back to eager timing
with a mode string that says so. This turns a silent mis-measurement into a loud,
honest one.

Example::

    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    ref = torch.softmax(x, dim=-1)
    result = cuda_graph_bench(
        lambda: my_op(x, out),
        warmup=10, iters=30,
        dirty=lambda: out.zero_(),
        verify=lambda: torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
    )
    for t in result["times_ms"]:
        print(f"wall_ms: {t:.6f}")

On a platform/op that cannot be captured (or fails the guard), it transparently
falls back to eager event timing so the driver still produces measurements (the
returned ``mode`` says which path ran and why).
"""

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
    """Capture ``step`` into a CUDA/HIP graph and time per-replay GPU execution.

    Raises on capture failure OR on a failed capture-validity guard so the caller
    can fall back to eager timing.
    """
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    # Capture-validity guard: an uncaptured launch yields an empty graph whose
    # replay is a fast no-op. Corrupt the output, replay, and confirm the graph
    # actually recomputed a correct result before we trust any timing.
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
    """Benchmark ``step`` under CUDA/HIP graph replay (eager fallback).

    Args:
        step: zero-arg closure running ONE operator invocation on pre-allocated
            tensors (see module docstring for the replay-safety contract). It
            must launch on the current stream.
        warmup: warmup invocations (on a side stream) before capture — this is
            where JIT compile / autotune / workspace allocation must happen.
        iters: number of timed graph replays.
        capture: set False to force the eager path (e.g. for A/B comparison).
        dirty: optional zero-arg closure that corrupts the output tensor(s)
            (e.g. ``lambda: out.zero_()``). Used with ``verify`` to prove the
            captured graph actually did work.
        verify: optional zero-arg predicate returning True iff the output is
            correct after a replay. When both ``dirty`` and ``verify`` are given,
            a graph that fails the check is rejected (falls back to eager).

    Returns:
        dict with ``mode`` ("cudagraph" or an "eager..." reason), ``times_ms``
        (per-iteration GPU time), and ``median_ms`` / ``mean_ms`` aggregates.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU available (torch.cuda.is_available() is False)")

    # Warm up on a side stream so JIT compile / autotune / workspace allocation
    # complete BEFORE capture (those steps are not capturable).
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
