# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real runner and the real local backend, with only the inference server replaced: a repro test with no GPU.

Started as a local sandbox template's command. ``run_grid`` is where the server would boot; the stand-in answers as a
boot-only round does (the server is "up" and nothing is measured), or fails the boot when the point says
``FAKE_BOOT: fail``. The runner unpacks the attachments, the backend runs the case's ``repro.sh`` and the manifest
signs what it leaves, all shipped code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

from hyperloom.measure import local
from hyperloom.measure.runner import run as runner_flow
from hyperloom.orchestrator.actions.executors import GridRecipe, VariantResult

PORT = 8123
CASE_TIMEOUT_SEC = 2.0
BOOT_FAILURE_LOG = "ModuleNotFoundError: No module named 'sglang_kernel'\n"


async def fake_run_grid(
    *, grid: list[Any], output_root: Path, recipe: GridRecipe, **kwargs: Any
) -> list[VariantResult]:
    name = grid[0].name
    workspace = Path(output_root) / name / "benchmark_stub_20261014_000000"
    workspace.mkdir(parents=True)
    envs = (yaml.safe_load(Path(recipe.base_yaml_path).read_text()).get("benchmark") or {}).get("envs") or {}
    if envs.get("FAKE_BOOT") == "fail":
        log = workspace / "server.log"
        log.write_text(BOOT_FAILURE_LOG)
        return [
            VariantResult(
                name=name,
                extra_server_args="",
                extra_envs={},
                status="failed",
                returncode=1,
                workspace=str(workspace),
                server_log_path=str(log),
                error=BOOT_FAILURE_LOG,
            )
        ]
    assert kwargs["launch"].lifecycle_boot_only is True, "a repro test boots the server and measures nothing"
    return [
        VariantResult(
            name=name,
            extra_server_args="",
            extra_envs={},
            status="succeeded",
            returncode=0,
            workspace=str(workspace),
            note="server_lifecycle_boot_only",
        )
    ]


def main() -> int:
    root = Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
    local.run_grid = fake_run_grid  # type: ignore[assignment]
    local._CASE_TIMEOUT_SEC = CASE_TIMEOUT_SEC  # a hanging case ends in seconds, not minutes
    local.resolve_lifecycle_params = lambda path: {  # type: ignore[assignment]
        "eligible": True,
        "port": PORT,
        "framework": "sglang",
        "reason": "",
    }
    local.teardown_lifecycle_server = lambda **kwargs: None  # type: ignore[assignment]
    return runner_flow.run(root / "in", root / "out", scratch=root / "work" / "scratch", spec_wait_sec=60)


if __name__ == "__main__":
    sys.exit(main())
