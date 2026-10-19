# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Evidence blocks read out of the bypass route's ``analysis.md`` report.

That report, held as ``last_trace_analyze["analysis_md_text"]``, is the only
place the per-kernel operand args appear. The TraceLens route's report has
another layout and parses to nothing here; its window and operator split come
from :mod:`~hyperloom.orchestrator.predictor.sidecars`. Each block is returned
complete or not at all: the predictor renders an absent block as a missing
paragraph, and was never trained on a half-filled one.

The headings, the summary rows' labels and the P-item column header are the
renderer's own constants, from ``hyperloom.orchestrator.trace_analysis._analysis_md``:
a summary row is matched by its label and a P-item cell by its column's name.
"""

from __future__ import annotations

import re
from typing import Any

from hyperloom.orchestrator.trace_analysis._analysis_md import (
    ATTRIBUTION_LABEL,
    DASH,
    EXEC_SUMMARY_HEADING,
    EXPOSED_COMM_LABEL,
    GPU_BUSY_LABEL,
    GPU_IDLE_LABEL,
    P_ITEM_COLUMNS,
    SYSTEM_SIGNALS_HEADING,
    TOP_BOTTLENECK_LABEL,
    TOTAL_GPU_TIME_LABEL,
)

_P_ITEM_HEADER = [cell.strip() for cell in P_ITEM_COLUMNS.strip().strip("|").split("|")]
_P_ITEM_CELLS = len(_P_ITEM_HEADER)
_P_ITEM_COLUMN = {name: index for index, name in enumerate(_P_ITEM_HEADER)}

_ROW_RE = re.compile(r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<value>[^|]*?)\s*\|", re.MULTILINE)
_P_ITEM_HEADING_RE = re.compile(r"^###\s+P\d+:\s*(?P<category>.+?)\s+kernels\s*$", re.MULTILINE)
_NEXT_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")
#: Operand shapes are joined with ``<br>`` so the cell wraps; in a prompt the tag is noise.
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HEADER_LABELS = frozenset({"metric", "signal", "rank", "operation"})


def _cell(raw: Any) -> str | None:
    """A cell's text, or ``None`` for an empty cell or the renderer's dash."""
    if raw is None:
        return None
    text = _BR_RE.sub(", ", str(raw)).strip()
    text = re.sub(r"(,\s*)+", ", ", text).strip().strip(",").strip()
    return None if not text or text == DASH else text


def _number(raw: Any) -> float | None:
    """The leading number in a cell, ignoring a ``%`` or unit suffix such as ``263.980 ms``."""
    text = _cell(raw)
    match = _NUM_RE.search(text.replace(",", "")) if text else None
    return float(match.group(1)) if match else None


def _section_rows(text: str, heading: str) -> dict[str, str]:
    """Two-column rows of the table under ``heading``, by label; scoped so two tables never overwrite each other."""
    start = text.find(heading)
    if start < 0:
        return {}
    body = text[start + len(heading) :]
    end = _NEXT_HEADING_RE.search(body)
    rows: dict[str, str] = {}
    for match in _ROW_RE.finditer(body[: end.start()] if end else body):
        label = match.group("label").strip()
        if label and not set(label) <= {"-", ":"} and label.lower() not in _HEADER_LABELS:
            rows[label] = match.group("value").strip()
    return rows


def parse_window(text: str) -> dict[str, Any] | None:
    """Window-relative timings, or ``None`` unless all four are present."""
    summary = _section_rows(text, EXEC_SUMMARY_HEADING)
    signals = _section_rows(text, SYSTEM_SIGNALS_HEADING)
    block = {
        "total_gpu_time_ms": _number(summary.get(TOTAL_GPU_TIME_LABEL)),
        "gpu_busy_pct": _number(summary.get(GPU_BUSY_LABEL)),
        "gpu_idle_pct": _number(summary.get(GPU_IDLE_LABEL)),
        "exposed_comm_pct": _number(signals.get(EXPOSED_COMM_LABEL)),
    }
    return None if any(value is None for value in block.values()) else block


def _p_item_groups(text: str) -> list[tuple[str | None, list[dict[str, Any]]]]:
    """Per-P-item ``(category, rows)`` in report order; a row with the wrong cell count is skipped, not misread."""
    headings = list(_P_ITEM_HEADING_RE.finditer(text))
    groups: list[tuple[str | None, list[dict[str, Any]]]] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        rows: list[dict[str, Any]] = []
        for line in text[heading.end() : end].splitlines():
            line = line.strip()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not line.startswith("|") or len(cells) != _P_ITEM_CELLS:
                continue
            if cells == _P_ITEM_HEADER or set(cells[0]) <= {"-", ":"}:
                continue
            count = _number(cells[_P_ITEM_COLUMN["Count"]])
            rows.append(
                {
                    "name": _cell(cells[_P_ITEM_COLUMN["Operation"]]),
                    "time_us": _number(cells[_P_ITEM_COLUMN["Time (us)"]]),
                    "gpu_pct": _number(cells[_P_ITEM_COLUMN["GPU%"]]),
                    "call_count": None if count is None else int(count),
                    "args": _cell(cells[_P_ITEM_COLUMN["Args"]]),
                }
            )
        groups.append((_cell(heading.group("category")), rows))
    return groups


def parse_operators(text: str) -> dict[str, Any] | None:
    """Operator distribution summed from the P-item tables, or ``None`` when no category carries a share.

    ``attribution_pct`` stays ``None`` when the report has no attribution column;
    the predictor reads ``None`` as unmeasured and a number at or below 90 as a
    warning, so it must not become ``0.0``.
    """
    category_pct: dict[str, float] = {}
    for category, rows in _p_item_groups(text):
        if category:
            share = sum(row["gpu_pct"] for row in rows if row["gpu_pct"] is not None)
            category_pct[category] = round(category_pct.get(category, 0.0) + share, 2)
    if not category_pct:
        return None
    summary = _section_rows(text, EXEC_SUMMARY_HEADING)
    return {
        "top_bottleneck_category": _cell(summary.get(TOP_BOTTLENECK_LABEL)),
        "attribution_pct": _number(summary.get(ATTRIBUTION_LABEL)),
        "category_pct": category_pct,
        "top3_cumulative_pct": round(sum(sorted(category_pct.values(), reverse=True)[:3]), 2),
    }


def p_item_index(text: str) -> dict[str, dict[str, Any]]:
    """Operation name to its first P-item row: the args and call count ``hot_kernels_top15`` lacks."""
    index: dict[str, dict[str, Any]] = {}
    for _, rows in _p_item_groups(text):
        for row in rows:
            if row["name"] and row["name"] not in index:
                index[row["name"]] = row
    return index
