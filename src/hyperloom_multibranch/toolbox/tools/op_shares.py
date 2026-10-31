# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The per-op shares of a TraceLens analysis, for the tools that run one (``tracelens``, ``roofline``).

The shares come from a table of every op's share of the total GPU time. The ``report`` route of ``tracelens`` names its own
(``kernel_shares_path``: the kernels of a graded wave, the decode and prefill windows weighted by the time each takes,
:mod:`~.phase_mix`): TraceLens' per-op durations are wrong on a decode window after a graph replay. The agent route and the
roofline name the kernels of the one window they ran on
(:func:`~.tracelens_report.with_window_shares`). An analysis that names none has TraceLens' ``ops_summary.csv`` (the
``perf_report_csvs`` folder beside the report), the complete per-op table its perf report writes deterministically from the
trace. Neither is the
``analysis.json`` task list, which holds only the few optimisation tasks the report recommends (a few percent of the time)
and would hide GEMM and attention from the Amdahl ceiling. A tool that produced no table, or whose request does not name the
commit the profile ran on, reports no shares: they are never guessed. A report TraceLens marked degraded (its deterministic
fallback, or a poor trace) carries no shares either: the decomposition behind them failed, so the one row it keeps is a
kernel name, not an op. So does one the report route itself found hollow (``window_degraded_reason``).
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from hyperloom_multibranch.contracts.blackboard.records import MAX_OP_SHARES
from hyperloom_multibranch.toolbox.tools.base import ToolOutput, ToolRequest

log = logging.getLogger(__name__)

ANALYSIS_JSON = "analysis.json"
OPS_SUMMARY = "perf_report_csvs/ops_summary.csv"
NAME_COLUMN, SHARE_COLUMN = "name", "Percentage (%)"
COMMIT_PARAM = "commit_id"
PROFILE_REQUEST_PARAM = "profile_request_id"
_REPORT_KEYS = ("trace_report_path", "analysis_md_path")
#: The reason the report route gives when its own check, not TraceLens', found the report hollow.
OWN_DEGRADED_KEY = "window_degraded_reason"
#: The report route's table of the window's kernels and their shares; empty when the window gives none.
KERNEL_SHARES_KEY = "kernel_shares_path"
#: ruppell's table of the profile's GPU time by module and step kind, beside the profile; absent when ruppell did not
#: trace it (``hyperloom.measure.ruppell``). The op shares never come from it.
MODULE_SHARES_KEY = "module_shares_path"
#: How many GPU events of the window the report route left out of its kernel shares, and the time they claim.
EXCLUDED_EVENTS_KEY, EXCLUDED_CLAIMED_KEY = "excluded_events", "excluded_claimed_ms"
#: What the report route says of a graded mix in which one window stands for the wave alone (``phase_mix``).
PHASE_NOTE_KEY = "phase_note"
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
    """Why the analysis ``result`` is degraded: TraceLens' ``report_info``, else the report route's own finding; empty if not."""
    return _tracelens_reason(result) or str(result.get(OWN_DEGRADED_KEY) or "").strip()


def _tracelens_reason(result: Mapping[str, Any]) -> str:
    """The degraded reason of the ``report_info`` in the ``analysis.json`` beside the report; empty when it is not (or unreadable)."""
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


def _table_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, ValueError, csv.Error):
        log.warning("op shares not read: %s is unreadable", path.name, exc_info=True)
        return []


def _share_table(result: Mapping[str, Any]) -> Path | None:
    """The table the shares are read from: the one the report route names (even when it names none), else the ops summary."""
    if KERNEL_SHARES_KEY in result:
        text = str(result[KERNEL_SHARES_KEY] or "").strip()
        return Path(text) if text else None
    directory = _report_dir(result)
    return directory / OPS_SUMMARY if directory is not None else None


def read_op_shares(result: Mapping[str, Any], limit: int = MAX_OP_SHARES) -> dict[str, float]:
    """``Percentage (%)`` per name of the table the analysis ``result`` points at, the ``limit`` largest only.

    A name the table lists in several rows has the rows' shares summed. Empty when the analysis wrote no table, or a
    degraded one.
    """
    table = _share_table(result)
    if table is None or degraded_reason(result):
        return {}
    shares: dict[str, float] = {}
    for row in _table_rows(table):
        name = str(row.get(NAME_COLUMN) or "").strip()
        try:
            share = float(row.get(SHARE_COLUMN) or "")
        except ValueError:
            continue
        if name and math.isfinite(share):
            shares[name] = shares.get(name, 0.0) + share
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
    return replace(output, summary=_with_notes(output), commit_id=commit, profile_request_id=profile, op_shares=shares)


def _with_notes(output: ToolOutput) -> str:
    """The summary, saying which window stands for the wave alone and how many GPU events the shares leave out as
    misreported, when the analysis says so."""
    summary = output.summary
    note = str(output.result.get(PHASE_NOTE_KEY) or "").strip()
    if note:
        summary = f"{summary} | {note}"
    excluded = output.result.get(EXCLUDED_EVENTS_KEY)
    if excluded:
        claimed = output.result.get(EXCLUDED_CLAIMED_KEY)
        summary = f"{summary} | {excluded} GPU events the profiler misreported (claiming {claimed} ms) are left out of the shares"
    return summary
