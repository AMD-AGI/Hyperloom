# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The request shape an AgentX round replayed, read from the aiperf export it leaves in its workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.common.agentx_workload import find_aiperf_artifact

_SHAPE_PERCENTILES = ("avg", "p50", "p75", "p90", "p99")
_AIPERF_SUMMARY = "profile_export_aiperf.json"


def measured_corpus_shape(workspace: Path | None) -> dict[str, Any]:
    """The ISL/OSL percentiles the round replayed, from aiperf's export; ``{}`` when it does not carry both."""
    path = None if workspace is None else find_aiperf_artifact(workspace, _AIPERF_SUMMARY)
    if path is None:
        return {}
    try:
        export = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(export, dict):
        return {}
    shape: dict[str, Any] = {}
    for key, name in (("isl", "input_sequence_length"), ("osl", "output_sequence_length")):
        metric = export.get(name)
        if not isinstance(metric, dict):
            return {}
        shape[key] = {p: int(metric[p]) for p in _SHAPE_PERCENTILES if isinstance(metric.get(p), (int, float))}
    return {**shape, "source": "measured"} if all(shape.values()) else {}
