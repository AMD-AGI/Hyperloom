# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The columns of a search-variant row: what it measured, what it was launched with, why it was refused, and where its artifacts are."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.common.prompt_safety import flatten_for_prompt as _flatten_for_prompt

# Width budget for artifact anchors.
_ANCHOR_MAX_CHARS = 100

# Anchors are dropped in this order while the set is over the width budget.
_ANCHOR_DROP_ORDER = ("log=", "ws=")

# A rejection reason, truncated to its actionable tail.
_REASON_CHARS = 120

# Reasons the gain column already conveys.
_REASONS_IMPLIED_BY_GAIN = ("not_keep", "gain_below_threshold")


def _is_measurement(value: Any) -> bool:
    """Whether a value is a recorded measurement; a bool is a flag, not a number this row can report."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _last_path_segments(raw: Any) -> str:
    """The last two segments of an artifact path; the full path is available via ``get_failure(fid)``."""
    text = str(raw or "").strip()
    return "/".join(PurePosixPath(text).parts[-2:]) if text else ""


def format_measurement(entry: dict[str, Any]) -> str:
    """The gain the variant moved, the axis that gain was taken on, and the throughput it reached.

    A REVERT carries no ``gain_pct`` by design; its graded movement is ``graded_gain_pct``. Only a row with
    neither was never measured on the graded axis -- printing "no_meas" for a guard refusal sends readers
    hunting for a broken extractor.
    """
    gain = entry.get("gain_pct")
    graded_gain = entry.get("graded_gain_pct")
    if _is_measurement(gain):
        gain_s = f"{gain:+.2f}%"
    elif _is_measurement(graded_gain):
        gain_s = f"{graded_gain:+.2f}%"
    else:
        gain_s = " no_meas"
    # The gain column is meaningless without the axis it was taken on.
    graded_obj = str(entry.get("graded_objective") or "").strip()
    graded_obj_s = f" [{graded_obj}]" if graded_obj and graded_obj != GRADED_OUTPUT else ""
    tput = entry.get("tput")
    tput_s = f" (tput={tput:.1f})" if isinstance(tput, (int, float)) and tput > 0 else ""
    return f"{gain_s:>9}{graded_obj_s}{tput_s}"


def format_config(entry: dict[str, Any]) -> str:
    """The server args and envs the variant was launched with."""
    args = str(entry.get("extra_server_args") or "").strip() or "(no-flag)"
    envs = entry.get("extra_envs") or {}
    envs_s = " " + " ".join(f"{k}={v}" for k, v in sorted(envs.items())) if envs else ""
    return f"{args}{envs_s}"


def format_diagnosis(entry: dict[str, Any]) -> str:
    """Why the variant failed or was refused, empty when the gain column already says it."""
    parts: list[str] = []
    error_class = str(entry.get("error_class") or "").strip()
    if error_class:
        parts.append(f"err={error_class}")
    reason = str(entry.get("reason") or "").strip()
    if reason and reason not in _REASONS_IMPLIED_BY_GAIN:
        # error_excerpt on variant rows is tail-1200 (boot assertion at end); flatten+defang prevents
        # section-header injection from untrusted log text.
        body = _flatten_for_prompt(str(entry.get("error_excerpt") or reason))
        parts.append(f"reason={body[-_REASON_CHARS:]}")
    return "  " + " ".join(parts) if parts else ""


def format_anchors(entry: dict[str, Any]) -> str:
    """The row's artifact anchors, dropped lowest-value-first until they fit the width budget."""
    labelled = (
        ("ws=", _last_path_segments(entry.get("workspace"))),
        ("log=", _last_path_segments(entry.get("server_log_path"))),
    )
    anchors = [f"{label}{value}" for label, value in labelled if value]
    rendered = ("  " + " ".join(anchors)) if anchors else ""
    for prefix in _ANCHOR_DROP_ORDER:
        if len(rendered) <= _ANCHOR_MAX_CHARS:
            break
        anchors = [a for a in anchors if not a.startswith(prefix)]
        rendered = ("  " + " ".join(anchors)) if anchors else ""
    return rendered
