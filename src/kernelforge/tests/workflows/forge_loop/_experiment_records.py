# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read back the experiment records an ExperimentTracker persisted, for assertions."""

from __future__ import annotations

import json
from pathlib import Path

from kernelforge.workflows.forge_loop.campaign.schema import Experiment


def persisted_experiments(experiments_dir: str | Path) -> list[Experiment]:
    """Every experiment record under ``experiments_dir``.

    A record carries both an identity and a creation time; the other JSON the run writes beside them, and corrupt
    files, are skipped.
    """
    experiments = []
    for path in sorted(Path(experiments_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("experiment_id") and payload.get("created_at"):
            experiments.append(Experiment.from_dict(payload))
    return experiments
