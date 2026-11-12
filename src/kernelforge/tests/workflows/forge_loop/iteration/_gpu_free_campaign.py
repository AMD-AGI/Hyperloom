# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A whole campaign in this process, then what of the GPU runtime the process holds.

Run as a module, in a process of its own, by the test that keeps the campaign process off the GPU: a GPU hang ends
every process that has the GPU open, so only a campaign that never opens it outlives one.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import _make_loop, _unused_supervisor
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore
from kernelforge.workflows.forge_loop.iteration import attempt_verdict
from kernelforge.workflows.forge_loop.iteration import runner as runner_module

GPU_RUNTIME_LIBRARIES = ("libamdhip64", "libhsa-runtime64")
GPU_DEVICE_NODE = "/dev/kfd"

#: Correct in every revision; twice as fast once the kernel returns 2.
DRIVER = """\
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-mode", action="store_true")
    args, _ = parser.parse_known_args()
    if not args.bench_mode:
        print("SNR: 62.13 dB")
        return 0
    kernel = (Path(__file__).resolve().parent / "kernel.py").read_text()
    case_ms = 0.5 if "return 2" in kernel else 1.0
    print(f"case_ms: case {case_ms:.6f}")
    print(f"mean_ms: {case_ms:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


async def _agent(kernel_path, _history, session_sink):
    """The first session makes the kernel faster, the second makes it no faster than the first."""
    value = 2 if "return 1" in Path(kernel_path).read_text() else 3
    Path(kernel_path).write_text(f"def kernel():\n    return {value}\n")
    session_sink["plan"] = f"return {value}"
    session_sink["end_reason"] = "candidate_submitted"
    return f"return {value}"


def _gpu_runtime_held() -> dict[str, list[str]]:
    mappings = Path("/proc/self/maps").read_text().splitlines()
    libraries = {line.split()[-1] for line in mappings if any(name in line for name in GPU_RUNTIME_LIBRARIES)}
    descriptors = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target == GPU_DEVICE_NODE:
            descriptors.add(target)
    return {"libraries": sorted(libraries), "descriptors": sorted(descriptors)}


async def _runs_work():
    return None


def main(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(attempt_verdict, "probe_device", _runs_work)
        loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=2, driver_source=DRIVER)
        asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_agent, supervisor_fn=_unused_supervisor)))
    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]
    print(json.dumps({"decisions": decisions, **_gpu_runtime_held()}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
