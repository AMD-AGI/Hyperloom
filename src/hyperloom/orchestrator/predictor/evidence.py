# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the predictor's evidence blocks out of the canonical ``analysis.md``.

``SharedState`` carries the roofline numbers as fields, but the window-relative
timings, the per-category operator split and the per-kernel ``args`` / call
counts only exist inside the rendered report. That report is already in memory
as ``last_trace_analyze["analysis_md_text"]``, so this module parses the string
rather than walking ``category_data/`` on disk.

Every block is returned complete or not at all. A half-filled block would let a
consumer describe a profiling window whose duration it does not know, and a
prompt shape that never occurred in a corpus is worse than an honestly absent
one.

``attribution_pct`` is the documented exception: the deterministic route leaves
op-attribution coverage unset, so ``None`` is a real value meaning "this report
has no attribution column" and must survive rather than become ``0.0``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mirrored from ``src/hyperloom/agents/kernel/tools/_analysis_md.py``.
#
# That module is the renderer and the single source of truth, but it cannot be
# imported from here: it resolves its own sibling with a bare
# ``from _kernel_category import canonical_category``, which only works when the
# tools directory is on ``sys.path``. Importing it as a package module raises
# ``ModuleNotFoundError``.
#
# ``tests/test_predictor_evidence.py::test_mirrored_constants_match_the_renderer``
# reads the renderer's source and fails when these drift, so a heading or column
# rename breaks loudly here instead of silently returning ``None`` for a block.
# ---------------------------------------------------------------------------

#: Placeholder the renderer emits for a cell it does not model.
DASH = "\u2014"

EXEC_SUMMARY_HEADING = "## Executive Summary"
SYSTEM_SIGNALS_HEADING = "## System-Level Signals"
TOP_HOT_KERNELS_HEADING = "## Top Hot Kernels"

P_ITEM_COLUMNS = (
    "| Operation | Time (us) | GPU% | %E2E | Count | FLOPS/Byte | "
    "Efficiency | Bound | Args | Source File | Kernel Path (launcher) |"
)

#: Executive Summary row labels this module reads.
_LABEL_TOTAL_GPU_TIME = "Total GPU Time"
_LABEL_GPU_BUSY = "GPU Busy %"
_LABEL_GPU_IDLE = "GPU Idle %"
_LABEL_TOP_BOTTLENECK = "Top Bottleneck Category"
_LABEL_ATTRIBUTION = "Op-attribution Coverage"

#: System-Level Signals row label this module reads.
_SIGNAL_EXPOSED_COMM = "Exposed communication"

# ---------------------------------------------------------------------------

#: ``| label | value |`` — the first two cells of any table row.
_ROW_RE = re.compile(r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<value>[^|]*?)\s*\|", re.MULTILINE)

#: ``### P3: gemm kernels`` — a per-P-item group heading.
_P_ITEM_HEADING_RE = re.compile(r"^###\s+P\d+:\s*(?P<category>.+?)\s+kernels\s*$", re.MULTILINE)

_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")

#: Number of cells in a P-item data row (``P_ITEM_COLUMNS`` minus the two empty
#: strings ``str.split('|')`` puts either side of a leading/trailing pipe).
_P_ITEM_CELLS = 11


#: Markdown-table line breaks. Upstream puts one between each operand shape so
#: the cell renders on several lines; in a prose prompt the literal tag is noise
#: no consumer was trained on, so it becomes a separator.
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _cell(raw: Any) -> str | None:
    """Return a table cell's text, or ``None`` when the renderer wrote a dash."""
    if raw is None:
        return None
    text = _BR_RE.sub(", ", str(raw)).strip()
    text = re.sub(r"(,\s*)+", ", ", text).strip().strip(",").strip()
    if not text or text == DASH:
        return None
    return text


def _number(raw: Any) -> float | None:
    """Return the leading number in a cell, ignoring a ``%`` or unit suffix.

    ``Total GPU Time`` renders as ``263.980 ms`` and percentages render as
    ``71.40%``, so the suffix has to come off before the float conversion. A
    parser that does not strip it drops the value and, with it, the whole block.

    Args:
        raw (Any): The raw cell text.

    Returns:
        float | None: The parsed number, or ``None`` when the cell is a dash,
            empty, or carries no number.
    """
    text = _cell(raw)
    if text is None:
        return None
    match = _NUM_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _int(raw: Any) -> int | None:
    """Return a cell's value as an int, for counts the renderer writes as text.

    ``call_count`` goes through the renderer's ``_text`` rather than ``_num``,
    so it arrives as ``1440`` and not ``1440.0``. Keeping it an int stops a
    launch count from being rendered downstream as a float.

    Args:
        raw (Any): The raw cell text.

    Returns:
        int | None: The parsed count, or ``None`` when absent or non-numeric.
    """
    value = _number(raw)
    return None if value is None else int(value)


def _int(raw: Any) -> int | None:
    """Return a cell as an integer — launch counts are counts, not measurements."""
    value = _number(raw)
    return None if value is None else int(value)


def _section(text: str, heading: str) -> str:
    """Return the body between ``heading`` and the next heading of any level.

    Scoping matters because ``_ROW_RE`` only anchors on a row's first two
    cells: a three-column System-Level Signals row matches the same pattern as
    a two-column Executive Summary row, so an unscoped parse would let the two
    tables overwrite each other.

    Args:
        text (str): The full report.
        heading (str): The exact heading line to start from.

    Returns:
        str: The section body, or ``""`` when the heading is absent.
    """
    start = text.find(heading)
    if start < 0:
        return ""
    start += len(heading)
    rest = text[start:]
    nxt = re.search(r"^#{1,6}\s", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _rows(section: str) -> dict[str, str]:
    """Map a section's table rows by their first cell, skipping header rows."""
    out: dict[str, str] = {}
    for match in _ROW_RE.finditer(section):
        label = match.group("label").strip()
        if not label or set(label) <= {"-", ":"}:
            continue
        if label.lower() in ("metric", "signal", "rank", "operation"):
            continue
        out[label] = match.group("value").strip()
    return out


def parse_window(text: str) -> dict[str, Any] | None:
    """Window-relative timings, or ``None`` when the report cannot fill them all.

    Args:
        text (str): The ``analysis.md`` body.

    Returns:
        dict[str, Any] | None: ``total_gpu_time_ms`` / ``gpu_busy_pct`` /
            ``gpu_idle_pct`` / ``exposed_comm_pct``, or ``None`` if any is
            missing.
    """
    exec_rows = _rows(_section(text, EXEC_SUMMARY_HEADING))
    signal_rows = _rows(_section(text, SYSTEM_SIGNALS_HEADING))

    block = {
        "total_gpu_time_ms": _number(exec_rows.get(_LABEL_TOTAL_GPU_TIME)),
        "gpu_busy_pct": _number(exec_rows.get(_LABEL_GPU_BUSY)),
        "gpu_idle_pct": _number(exec_rows.get(_LABEL_GPU_IDLE)),
        "exposed_comm_pct": _number(signal_rows.get(_SIGNAL_EXPOSED_COMM)),
    }
    missing = [k for k, v in block.items() if v is None]
    if missing:
        log.debug("predictor_evidence: dropping window block, missing %s", missing)
        return None
    return block


def parse_p_items(text: str) -> list[dict[str, Any]]:
    """Per-P-item groups with their data rows.

    Args:
        text (str): The ``analysis.md`` body.

    Returns:
        list[dict[str, Any]]: ``{"category": str, "rows": [...]}`` in report
            order. Rows whose cell count disagrees with ``P_ITEM_COLUMNS`` are
            skipped rather than positionally misread — an ``Args`` cell holding
            a literal pipe is the realistic way that happens.
    """
    groups: list[dict[str, Any]] = []
    headings = list(_P_ITEM_HEADING_RE.finditer(text))
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[match.end() : end]
        rows: list[dict[str, Any]] = []
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != _P_ITEM_CELLS:
                continue
            if cells[0].lower() == "operation" or set(cells[0]) <= {"-", ":"}:
                continue
            rows.append(
                {
                    "name": _cell(cells[0]),
                    "time_us": _number(cells[1]),
                    "gpu_pct": _number(cells[2]),
                    "e2e_pct": _number(cells[3]),
                    "call_count": _int(cells[4]),
                    "flops_per_byte": _number(cells[5]),
                    "efficiency_percent": _number(cells[6]),
                    "bound_type": _cell(cells[7]),
                    "args": _cell(cells[8]),
                    "source_file": _cell(cells[9]),
                    "kernel_path": _cell(cells[10]),
                }
            )
        groups.append({"category": _cell(match.group("category")), "rows": rows})
    return groups


def parse_operators(text: str) -> dict[str, Any] | None:
    """Operator distribution aggregated from the per-P-item tables.

    ``category_pct`` is the sum of each group's row ``gpu_pct``; the renderer
    groups by canonical category already, so no re-canonicalisation is needed.

    ``attribution_pct`` stays nullable on purpose — see the module docstring.

    Args:
        text (str): The ``analysis.md`` body.

    Returns:
        dict[str, Any] | None: ``top_bottleneck_category`` / ``attribution_pct``
            / ``category_pct`` / ``top3_cumulative_pct``, or ``None`` when no
            category carries a share.
    """
    category_pct: dict[str, float] = {}
    for group in parse_p_items(text):
        category = group.get("category")
        if not category:
            continue
        share = sum(row["gpu_pct"] for row in group["rows"] if row.get("gpu_pct") is not None)
        category_pct[category] = round(category_pct.get(category, 0.0) + share, 2)

    if not category_pct:
        log.debug("predictor_evidence: dropping operators block, no per-category shares")
        return None

    exec_rows = _rows(_section(text, EXEC_SUMMARY_HEADING))
    top3 = sorted(category_pct.values(), reverse=True)[:3]
    return {
        "top_bottleneck_category": _cell(exec_rows.get(_LABEL_TOP_BOTTLENECK)),
        "attribution_pct": _number(exec_rows.get(_LABEL_ATTRIBUTION)),
        "category_pct": category_pct,
        "top3_cumulative_pct": round(sum(top3), 2),
    }


def p_item_index(text: str) -> dict[str, dict[str, Any]]:
    """Map operation name to its P-item row, for enriching hot-kernel entries.

    ``hot_kernels_top15`` carries neither ``args`` nor a launch count; the
    P-item tables carry both. First occurrence wins, matching report order.

    Args:
        text (str): The ``analysis.md`` body.

    Returns:
        dict[str, dict[str, Any]]: Operation name to row.
    """
    index: dict[str, dict[str, Any]] = {}
    for group in parse_p_items(text):
        for row in group["rows"]:
            name = row.get("name")
            if name and name not in index:
                index[name] = row
    return index
