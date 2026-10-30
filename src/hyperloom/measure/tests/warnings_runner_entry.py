# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real runner and the real local backend, with only ``run_grid`` and the machine's probes replaced: a measured round
that warns, on a machine that is the same wherever the test runs.

Started as a local sandbox template's command. The stand-in leaves the report a measured round leaves and answers with
the ``nonfatal_warnings`` a real ``run_grid`` would put on the variant, so the path from ``run_grid`` through the signed
rounds file to the stored result is all shipped code. The probes the runner reads the machine with (who holds its cards,
what they hold, what they are) answer for a machine whose cards nothing else uses: left to the host, a driver that is not
there adds ``card_occupancy_unknown`` to the result, and another process on the host's cards adds
``foreign_process_on_card``, and the warnings the test is about are no longer the only ones.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from hyperloom.measure import local
from hyperloom.measure.runner import hardware
from hyperloom.measure.runner import run as runner_flow
from hyperloom.orchestrator.actions.executors import VariantResult
from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import CardReading

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


def _idle_machine() -> None:
    """Answer the runner's probes for a machine on which nothing holds the cards, whatever the host is."""
    card_occupancy.read_holders = lambda cards: CardReading()  # type: ignore[assignment]
    hardware.busy_gpus = lambda visible: []  # type: ignore[assignment]
    hardware.read_gpu_facts = lambda visible: None  # type: ignore[assignment]


def main() -> int:
    root = Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
    local.run_grid = fake_run_grid  # type: ignore[assignment]
    _idle_machine()
    return runner_flow.run(root / "in", root / "out", scratch=root / "work" / "scratch", spec_wait_sec=60)


if __name__ == "__main__":
    sys.exit(main())
