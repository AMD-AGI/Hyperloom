# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Building the trace shapes the serving censuses read, so each test states what it is about and no more.

The censuses answer different questions off one file, and every one of their tests needs the same handful of
events to ask it: an annotated step on the device timeline, a kernel inside it, and the launch call that says the
kernel was dispatched rather than replayed. Spelling those out per test buries the case each one is making.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

#: The device-side annotation flavour. The host records the same steps, but its clock is not the kernels'.
GPU_ANNOTATION = "gpu_user_annotation"

#: The process id a device timeline uses; kernels and device annotations share it.
DEVICE_PID = 2


def step(name: str, ts: float, dur: float, *, cat: str = GPU_ANNOTATION) -> dict[str, Any]:
    """One annotated step, e.g. ``step[VERIFY bs=3]``."""
    return {"cat": cat, "name": name, "ts": ts, "dur": dur, "pid": DEVICE_PID, "tid": 7}


def kernel(name: str, ts: float, dur: float, *, correlation: int | None = None) -> dict[str, Any]:
    """One kernel. Give it a correlation only when a launch call will claim the same id."""
    event: dict[str, Any] = {"cat": "kernel", "name": name, "ts": ts, "dur": dur, "pid": DEVICE_PID, "tid": 7}
    if correlation is not None:
        event["args"] = {"correlation": correlation}
    return event


def launch(correlation: int) -> dict[str, Any]:
    """The runtime call that dispatched one kernel; a replayed kernel has none."""
    return {
        "cat": "cuda_runtime",
        "name": "hipLaunchKernel",
        "ts": 0.0,
        "dur": 1.0,
        "args": {"correlation": correlation},
    }


def write_trace(path: Path, events: list[dict[str, Any]], *, gzipped: bool | None = None) -> Path:
    """Write *events* as a capture; gzipped by default when the name says so."""
    if gzipped is None:
        gzipped = path.suffix == ".gz" or path.name.endswith(".json.gz")
    payload = json.dumps({"traceEvents": events})
    if gzipped:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def eager_step(name: str, *, start: float, count: int, first_correlation: int = 0) -> list[dict[str, Any]]:
    """A step whose every kernel carries its own launch call: nothing here came from a graph."""
    events: list[dict[str, Any]] = [step(name, start, 10.0 * count)]
    events.extend(kernel(f"eager_{i}", start + 1.0 + i, 5.0, correlation=first_correlation + i) for i in range(count))
    events.extend(launch(first_correlation + i) for i in range(count))
    return events


def graphed_step(name: str, *, start: float, count: int, first_correlation: int = 900) -> list[dict[str, Any]]:
    """A step whose kernels no launch call claims: every one of them was replayed."""
    events: list[dict[str, Any]] = [step(name, start, 10.0 * count)]
    events.extend(kernel(f"graphed_{i}", start + 1.0 + i, 5.0, correlation=first_correlation + i) for i in range(count))
    return events


__all__ = [
    "DEVICE_PID",
    "GPU_ANNOTATION",
    "eager_step",
    "graphed_step",
    "kernel",
    "launch",
    "step",
    "write_trace",
]
