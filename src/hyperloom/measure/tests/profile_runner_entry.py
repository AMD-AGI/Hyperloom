# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real runner and the real local backend, with only ``run_grid`` replaced: no server, but a torch trace on disk.

Started as a local sandbox template's command. ``run_grid`` is where the GPU run would happen; the stand-in writes what
a profiled round leaves (a run_dir with ``profile/torch_trace/<rank>.pt.trace.json.gz``), so the local backend's
collection of the traces, the runner's inventory and the signed manifest are all the shipped code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from hyperloom.measure import local
from hyperloom.measure.runner import run as runner_flow
from hyperloom.measure.tests.bundle_factory import lm_eval_at_its_pin
from hyperloom.orchestrator.actions.executors import VariantResult
from inference_suite.contracts import RunLayout

TRACE_NAME = "rank0.pt.trace.json.gz"


async def fake_run_grid(*, grid: list[Any], output_root: Path, **_: Any) -> list[VariantResult]:
    name = grid[0].name
    workspace = Path(output_root) / name / "suite_run_0"
    trace = RunLayout(workspace).profile_torch_trace / TRACE_NAME
    trace.parent.mkdir(parents=True)
    trace.write_bytes(b"stub trace bytes")
    return [
        VariantResult(
            name=name, extra_server_args="", extra_envs={}, status="succeeded", returncode=0, workspace=str(workspace)
        )
    ]


def main() -> int:
    root = Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
    local.run_grid = fake_run_grid  # type: ignore[assignment]
    sys.path.insert(0, str(lm_eval_at_its_pin(root / "lm-eval-pin")))
    return runner_flow.run(root / "in", root / "out", scratch=root / "work" / "scratch", spec_wait_sec=60)


if __name__ == "__main__":
    sys.exit(main())
