# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""External baseline comparison layer (report-only).

This subpackage owns the path from the user's ``--compare-against-gpu``
flag to a persisted-on-disk reference data point that the final report
can render alongside the run's own measured throughput.

Strict design boundary: nothing here ever writes to ``SharedState``,
never participates in the orchestrator's scoring or Objective, never
shows up in any agent's system prompt. The only consumers are:

* :class:`inference_optimizer.orchestrator.action_executors.TargetAnalysisExecutor`
  — calls :func:`target_analyzer.analyze` once per session.
* :class:`inference_optimizer.orchestrator.action_executors.ReportExecutor`
  — reads ``target_analysis/target_baseline.json`` to render an advisory
  section in ``final.md``.

Source of upstream data: the InferenceX public benchmarks API
(https://inferencex.semianalysis.com/api/v1/benchmarks). The exact same
endpoint that the SaFE apiserver's ``apiserver/pkg/handlers/inferencex``
proxies — so this client and that handler stay observationally
equivalent (modulo authentication, which we don't need against the
public endpoint).
"""

from .inferencex_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_SEC,
    InferenceXFetchError,
    fetch_rows,
)
from .name_mapping import KNOWN_INFERENCEX_MODELS, to_inferencex_name
from .target_analyzer import analyze
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
    "fetch_rows",
    "to_inferencex_name",
]
