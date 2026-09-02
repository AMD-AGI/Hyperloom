# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the predictor's ``analysis.md`` evidence parser.

The parser mirrors headings and column names from the canonical renderer
``agents/kernel/tools/_analysis_md.py``, which cannot be imported as a package
module: it resolves a sibling with a bare ``from _kernel_category import ...``.
``test_mirrored_constants_match_the_renderer`` loads the renderer by path and
compares the values, so a heading rename fails here rather than silently
turning an evidence block into ``None``.

The round-trip tests drive the real renderer instead of hand-written markdown.
A fixture string would keep passing after the renderer changed shape, which is
the one failure this parser exists to catch early.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.predictor import evidence as ev

_TOOLS_DIR = (
    Path(__file__).resolve().parents[3] / "hyperloom" / "agents" / "kernel" / "tools"
)


def _load_renderer():
    """Import ``_analysis_md`` by path, with its bare-import sibling reachable."""
    tools = str(_TOOLS_DIR)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "_analysis_md_under_test", _TOOLS_DIR / "_analysis_md.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.skip(f"cannot load renderer at {_TOOLS_DIR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(
    *,
    exec_summary: dict[str, Any] | None = None,
    system_signals: dict[str, Any] | None = None,
    p_items: list[dict[str, Any]] | None = None,
    hot_kernels: list[dict[str, Any]] | None = None,
) -> str:
    renderer = _load_renderer()
    return renderer.render_report(
        route="deterministic",
        model_name="Qwen-Qwen3-8B",
        provenance_detail="",
        exec_summary=exec_summary if exec_summary is not None else {},
        system_signals=system_signals if system_signals is not None else {},
        idle_threshold=15.0,
        hot_kernels=hot_kernels or [],
        p_items=p_items or [],
    )


_FULL_EXEC = {
    "total_gpu_time_ms": 263.98,
    "gpu_busy_pct": 71.4,
    "gpu_idle_pct": 22.1,
    "gpu_memcpy_ms": 1.2,
    "top_bottleneck_category": "gemm",
    "attribution_pct": None,
}
_FULL_SIGNALS = {"idle_pct": 22.1, "exposed_comm_pct": 6.5, "exposed_memcpy_pct": 0.4}
_FULL_P_ITEMS = [
    {
        "rank": 1,
        "category": "gemm",
        "rows": [
            {
                "name": "torch_gemm",
                "time_us": 38.2,
                "gpu_pct": 14.1,
                "e2e_pct": 9.1,
                "call_count": 1440,
                "flops_per_byte": 118.4,
                "efficiency_percent": 61.0,
                "bound_type": "compute",
                "args": ["16x4096x12288", "bf16"],
                "source_file": "tuned_gemm.py",
                "kernel_path": "vllm/gemm",
            }
        ],
    },
    {
        "rank": 2,
        "category": "attention",
        "rows": [
            {
                "name": "fmha_fwd",
                "time_us": 21.0,
                "gpu_pct": 27.5,
                "e2e_pct": 17.0,
                "call_count": 720,
                "flops_per_byte": 44.1,
                "efficiency_percent": 52.0,
                "bound_type": "memory",
                "args": "bs64",
                "source_file": "attn.py",
                "kernel_path": "aiter/fmha",
            }
        ],
    },
]


def test_mirrored_constants_match_the_renderer():
    """The duplicated headings/columns must equal the renderer's own values."""
    renderer = _load_renderer()
    assert ev.DASH == renderer.DASH
    assert ev.EXEC_SUMMARY_HEADING == renderer.EXEC_SUMMARY_HEADING
    assert ev.SYSTEM_SIGNALS_HEADING == renderer.SYSTEM_SIGNALS_HEADING
    assert ev.TOP_HOT_KERNELS_HEADING == renderer.TOP_HOT_KERNELS_HEADING
    assert ev.P_ITEM_COLUMNS == renderer.P_ITEM_COLUMNS


def test_mirrored_row_labels_are_still_emitted():
    """Every Executive Summary / Signals label the parser reads must be rendered."""
    text = _render(exec_summary=_FULL_EXEC, system_signals=_FULL_SIGNALS)
    for label in (
        ev._LABEL_TOTAL_GPU_TIME,
        ev._LABEL_GPU_BUSY,
        ev._LABEL_GPU_IDLE,
        ev._LABEL_TOP_BOTTLENECK,
        ev._LABEL_ATTRIBUTION,
        ev._SIGNAL_EXPOSED_COMM,
    ):
        assert f"| {label} |" in text, f"renderer no longer emits {label!r}"


def test_window_block_strips_the_ms_suffix():
    """``Total GPU Time`` renders as ``263.980 ms``; the unit must not defeat it."""
    text = _render(exec_summary=_FULL_EXEC, system_signals=_FULL_SIGNALS)
    assert "263.980 ms" in text
    assert ev.parse_window(text) == {
        "total_gpu_time_ms": 263.98,
        "gpu_busy_pct": 71.4,
        "gpu_idle_pct": 22.1,
        "exposed_comm_pct": 6.5,
    }


def test_window_block_is_all_or_nothing():
    """One missing member drops the block rather than emitting a partial one."""
    exec_summary = dict(_FULL_EXEC)
    exec_summary["gpu_idle_pct"] = None
    assert ev.parse_window(_render(exec_summary=exec_summary, system_signals=_FULL_SIGNALS)) is None

    signals = dict(_FULL_SIGNALS)
    signals["exposed_comm_pct"] = None
    assert ev.parse_window(_render(exec_summary=_FULL_EXEC, system_signals=signals)) is None


def test_exposed_comm_comes_from_the_signals_table_not_the_summary():
    """Section scoping: the two tables share ``_ROW_RE``'s two-cell shape."""
    text = _render(exec_summary=_FULL_EXEC, system_signals=_FULL_SIGNALS)
    exec_body = ev._section(text, ev.EXEC_SUMMARY_HEADING)
    assert ev._SIGNAL_EXPOSED_COMM not in exec_body
    assert ev.parse_window(text)["exposed_comm_pct"] == 6.5


def test_operators_aggregate_category_shares():
    """``category_pct`` sums P-item rows; the renderer canonicalises the label."""
    block = ev.parse_operators(
        _render(exec_summary=_FULL_EXEC, system_signals=_FULL_SIGNALS, p_items=_FULL_P_ITEMS)
    )
    assert block is not None
    assert sorted(block["category_pct"].values()) == [14.1, 27.5]
    assert block["top3_cumulative_pct"] == 41.6
    assert block["top_bottleneck_category"]


def test_null_attribution_survives_as_none():
    """``None`` means "no attribution column"; ``0.0`` means "nothing attributed"."""
    block = ev.parse_operators(
        _render(exec_summary=_FULL_EXEC, system_signals=_FULL_SIGNALS, p_items=_FULL_P_ITEMS)
    )
    assert block["attribution_pct"] is None

    exec_summary = dict(_FULL_EXEC)
    exec_summary["attribution_pct"] = 0.0
    zero = ev.parse_operators(
        _render(exec_summary=exec_summary, system_signals=_FULL_SIGNALS, p_items=_FULL_P_ITEMS)
    )
    assert zero["attribution_pct"] == 0.0


def test_operators_block_absent_without_p_items():
    """No per-category share means no operator-distribution block."""
    assert (
        ev.parse_operators(_render(exec_summary=_FULL_EXEC, system_signals=_FULL_SIGNALS)) is None
    )


def test_p_item_index_supplies_args_and_launch_count():
    """The two hot-kernel fields ``hot_kernels_top15`` does not carry."""
    index = ev.p_item_index(_render(exec_summary=_FULL_EXEC, p_items=_FULL_P_ITEMS))
    assert set(index) == {"torch_gemm", "fmha_fwd"}
    row = index["torch_gemm"]
    assert row["args"] == "16x4096x12288 bf16"
    assert row["call_count"] == 1440
    assert isinstance(row["call_count"], int), "a launch count is a count, not a measurement"


def test_p_item_row_with_a_pipe_in_args_is_skipped_not_misread():
    """A stray pipe shifts every later cell; dropping the row beats lying."""
    p_items = [
        {
            "rank": 1,
            "category": "gemm",
            "rows": [
                {
                    "name": "bad_row",
                    "time_us": 1.0,
                    "gpu_pct": 1.0,
                    "e2e_pct": 1.0,
                    "call_count": 1,
                    "flops_per_byte": 1.0,
                    "efficiency_percent": 1.0,
                    "bound_type": "compute",
                    "args": "a | b",
                    "source_file": "x.py",
                    "kernel_path": "p",
                },
                dict(_FULL_P_ITEMS[0]["rows"][0]),
            ],
        }
    ]
    index = ev.p_item_index(_render(exec_summary=_FULL_EXEC, p_items=p_items))
    assert "bad_row" not in index
    assert "torch_gemm" in index


def test_empty_and_dash_only_reports_degrade_quietly():
    """Absence must read as absence, not as zeroes."""
    assert ev.parse_window("") is None
    assert ev.parse_operators("") is None
    assert ev.parse_p_items("") == []

    dashes = _render()
    assert ev.parse_window(dashes) is None
    assert ev.parse_operators(dashes) is None
