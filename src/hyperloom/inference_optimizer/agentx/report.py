# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an AgentX round replayed, read from the aiperf export it leaves in its workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.common.agentx_workload import find_aiperf_artifact

_SHAPE_PERCENTILES = ("avg", "p50", "p75", "p90", "p99")
_AIPERF_SUMMARY = "profile_export_aiperf.json"


def _aiperf_summary(workspace: Path | None) -> dict[str, Any]:
    """aiperf's summary export of the round in ``workspace``; ``{}`` when there is none that parses."""
    path = None if workspace is None else find_aiperf_artifact(workspace, _AIPERF_SUMMARY)
    if path is None:
        return {}
    try:
        export = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return export if isinstance(export, dict) else {}


def measured_corpus_shape(workspace: Path | None) -> dict[str, Any]:
    """The ISL/OSL percentiles the round replayed, from aiperf's export; ``{}`` when it does not carry both."""
    export = _aiperf_summary(workspace)
    shape: dict[str, Any] = {}
    for key, name in (("isl", "input_sequence_length"), ("osl", "output_sequence_length")):
        metric = export.get(name)
        if not isinstance(metric, dict):
            return {}
        shape[key] = {p: int(metric[p]) for p in _SHAPE_PERCENTILES if isinstance(metric.get(p), (int, float))}
    return {**shape, "source": "measured"} if all(shape.values()) else {}


def replayed_trace(workspace: Path | None) -> dict[str, Any]:
    """The corpus the round replayed and its entry count, under the ``workload_spec`` keys GEAK reads, with the mean
    request shape when aiperf measured one; ``{}`` when aiperf's export names no corpus."""
    dataset = (_aiperf_summary(workspace).get("metadata") or {}).get("dataset") or {}
    corpus, entries = dataset.get("loader"), dataset.get("num_dataset_entries")
    if not isinstance(corpus, str) or not corpus or isinstance(entries, bool) or not isinstance(entries, int):
        return {}
    trace = {"corpus": corpus, "canonical_corpus": corpus, "num_entries": entries, "canonical_num_entries": entries}
    shape = measured_corpus_shape(workspace)
    if "avg" in shape.get("isl", {}) and "avg" in shape.get("osl", {}):
        trace.update(observed_isl=shape["isl"]["avg"], observed_osl=shape["osl"]["avg"])
    return trace
