# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The per-op shares of a TraceLens analysis, for the tools that run one (``tracelens``, ``roofline``).

The shares come from the ``analysis.json`` beside the analysis report, read by the single flow's own loader. A tool
that produced no table, or whose request does not name the commit the profile ran on, reports no shares: they are
never guessed. A report TraceLens marked degraded (its deterministic fallback, or a poor trace) carries no shares
either: the decomposition behind them failed, so the one row it keeps is a kernel name, not an op.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from hyperloom_multibranch.contracts.notebook.records import MAX_OP_SHARES
from hyperloom_multibranch.toolbox.tools.base import ToolOutput, ToolRequest

log = logging.getLogger(__name__)

ANALYSIS_JSON = "analysis.json"
COMMIT_PARAM = "commit_id"
PROFILE_REQUEST_PARAM = "profile_request_id"
_REPORT_KEYS = ("trace_report_path", "analysis_md_path")
DEGRADED_MODE = "deterministic-fallback"
POOR_QUALITY = "poor"
DEGRADED_NOTICE = "TraceLens report is degraded, so no op shares were taken from it"


def _report_dir(result: Mapping[str, Any]) -> Path | None:
    for key in _REPORT_KEYS:
        text = str(result.get(key) or "").strip()
        if text:
            return Path(text).parent
    return None


def degraded_reason(result: Mapping[str, Any]) -> str:
    """Why the analysis ``result`` points at is degraded, from its ``report_info``; empty when it is not (or unreadable)."""
    directory = _report_dir(result)
    if directory is None:
        return ""
    try:
        report = json.loads((directory / ANALYSIS_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    info = report.get("report_info") if isinstance(report, dict) else None
    if not isinstance(info, dict):
        return ""
    mode, quality = str(info.get("mode") or ""), str(info.get("trace_quality") or "")
    if mode != DEGRADED_MODE and quality != POOR_QUALITY:
        return ""
    warnings = info.get("warnings")
    detail = " ".join(str(w) for w in warnings) if isinstance(warnings, list) else ""
    return " ".join(f"mode={mode or '?'} trace_quality={quality or '?'}: {detail}".split())


def read_op_shares(result: Mapping[str, Any], limit: int = MAX_OP_SHARES) -> dict[str, float]:
    """``percent_of_total`` per op name of the analysis ``result`` points at, the ``limit`` largest only.

    An op the table lists in several rows has the rows' shares summed. Empty when the analysis wrote no table,
    or a degraded one.
    """
    directory = _report_dir(result)
    if directory is None or degraded_reason(result):
        return {}
    try:
        from hyperloom.orchestrator.trace_analysis.analysis_json import load_report_tasks

        rows = load_report_tasks(directory / ANALYSIS_JSON)
    except ImportError:
        log.warning("op shares not read: the TraceLens reader is not importable", exc_info=True)
        return {}
    shares: dict[str, float] = {}
    for row in rows:
        name, share = str(row.get("name") or "").strip(), row.get("gpu_pct")
        if name and isinstance(share, (int, float)) and math.isfinite(share):
            shares[name] = shares.get(name, 0.0) + float(share)
    largest = sorted(shares.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return dict(largest)


def with_op_shares(output: ToolOutput, request: ToolRequest) -> ToolOutput:
    """``output`` carrying the shares of its analysis, on the commit the request names; unchanged when unknown."""
    commit = str(request.params.get(COMMIT_PARAM) or "").strip()
    shares = read_op_shares(output.result) if commit and output.ok else {}
    if not shares:
        reason = degraded_reason(output.result) if commit and output.ok else ""
        return replace(output, summary=f"{output.summary} | {DEGRADED_NOTICE} ({reason})") if reason else output
    profile = str(request.params.get(PROFILE_REQUEST_PARAM) or "").strip()
    return replace(output, commit_id=commit, profile_request_id=profile, op_shares=shares)
