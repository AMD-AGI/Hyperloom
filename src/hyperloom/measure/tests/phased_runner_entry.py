# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real runner and the real local backend, with only ``run_grid`` replaced: the test of the kind
``HL_PHASE_SCENARIO`` names (``graded``, ``agentx`` or ``failing_boot``, see ``phase_rounds.Rounds``), whose rounds spend
``HL_PHASE_SCALE`` times the seconds a real round spends and leave the marks a launcher leaves beside each run's log.

Started as a local sandbox template's command. What the runner stamps, signs and the host lands is all shipped code.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from hyperloom.measure import local
from hyperloom.measure.runner import run as runner_flow
from hyperloom.measure.tests.phase_rounds import RealClock, Rounds, StagedGrid


def main() -> int:
    root = Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
    rounds = Rounds(RealClock(), float(os.environ["HL_PHASE_SCALE"]))
    local.run_grid = StagedGrid(getattr(rounds, os.environ["HL_PHASE_SCENARIO"])())  # type: ignore[assignment]
    local.run_grid_warmup_enabled = lambda: True  # type: ignore[assignment]
    local.resolve_lifecycle_params = lambda path: {"eligible": True, "port": 8123, "framework": "sglang"}  # type: ignore[assignment]

    def stop_server(**kwargs: Any) -> None:
        time.sleep(3.0 * float(os.environ["HL_PHASE_SCALE"]))

    local.teardown_lifecycle_server = stop_server  # type: ignore[assignment]
    return runner_flow.run(root / "in", root / "out", scratch=root / "work" / "scratch", spec_wait_sec=60)


if __name__ == "__main__":
    sys.exit(main())
