# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""External baseline comparison layer.

This subpackage owns the path from the user's ``--compare-against-gpu``
flag to persisted-on-disk InferenceX-measured reference points that the
final report can render alongside the run's own measured throughput.

Design boundary: nothing here ever writes to ``SharedState``, participates
in the orchestrator's scoring / Objective, or gates any KEEP/REVERT
decision. On a successful match :func:`target_analyzer.analyze` does write a
measured ``competitor_target.json`` (``source`` = the API URL) that the
gap advisory surfaces to specialists as *direction, not a gate* — so
any reference number reaching a prompt is always API-measured, never
LLM-authored. Consumers:

* :class:`hyperloom.orchestrator.actions.executors.TargetAnalysisExecutor`
  — calls :func:`target_analyzer.analyze` once per session.
* :class:`hyperloom.orchestrator.actions.executors.ReportExecutor`
  — reads ``target_analysis/target_baseline.json`` to render the report
  section in ``final.md``.
* the gap advisory — reads the measured ``competitor_target.json``.

Source of upstream data: the InferenceX public benchmarks API
(https://inferencex.semianalysis.com/api/v1/benchmarks).
"""

from .inferencex_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_SEC,
    InferenceXFetchError,
)
from .target_analyzer import KNOWN_INFERENCEX_MODELS, analyze, to_inferencex_name
from .types import BaselinePoint, BaselineQuery, BaselineSummary


__all__ = [
    "BaselinePoint",
    "BaselineQuery",
    "BaselineSummary",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT_SEC",
    "InferenceXFetchError",
    "KNOWN_INFERENCEX_MODELS",
    "analyze",
    "to_inferencex_name",
]
