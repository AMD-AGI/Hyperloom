# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real runner and the real local backend, with only ``run_grid`` replaced: a measured round that warns.

Started as a local sandbox template's command. The stand-in leaves the report a measured round leaves and answers with
the ``nonfatal_warnings`` a real ``run_grid`` would put on the variant, so the path from ``run_grid`` through the signed
rounds file to the stored result is all shipped code.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from hyperloom.measure import local
from hyperloom.measure.runner import run as runner_flow
from hyperloom.orchestrator.actions.executors import VariantResult

WARNINGS = ["warmup_round_seed:1", "harvested_leaked_artifact:x", "warmup_round_seed:1"]


async def fake_run_grid(*, grid: list[Any], output_root: Path, **_: Any) -> list[VariantResult]:
    name = grid[0].name
    workspace = Path(output_root) / name / "benchmark_stub_20261014_000000"
    workspace.mkdir(parents=True)
    report = {"success": True, "throughput": {"output_throughput": 1234.5, "completed_requests": 8, "num_prompts": 8}}
    (workspace / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
    return [
        VariantResult(
            name=name,
            extra_server_args="",
            extra_envs={},
            status="succeeded",
            returncode=0,
            workspace=str(workspace),
            nonfatal_warnings=list(WARNINGS),
        )
    ]


def main() -> int:
    root = Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
    local.run_grid = fake_run_grid  # type: ignore[assignment]
    return runner_flow.run(root / "in", root / "out", scratch=root / "work" / "scratch", spec_wait_sec=60)


if __name__ == "__main__":
    sys.exit(main())
