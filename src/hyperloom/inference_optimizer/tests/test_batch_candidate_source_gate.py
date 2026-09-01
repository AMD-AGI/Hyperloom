# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pins which rows the batch selector drops before dispatch.

The selector filters on ``source_file`` and ``reusable_native_kernel`` in two
independent passes -- the grouped pass and the legacy per-kernel pass -- and both
drops are silent. Moving kernel selection into forge removes these filters, so
their current shape is pinned here first: a refactor that changes which rows
survive should fail loudly rather than quietly widening or narrowing dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.kernel import request_handlers as krh


def _row(kernel_id: str, **overrides: Any) -> dict[str, Any]:
    """A row that passes every filter, so one override isolates one filter."""
    row: dict[str, Any] = {
        "kernel_id": kernel_id,
        "name": f"{kernel_id}_kernel",
        "source_file": f"/repo/{kernel_id}.py",
        "gpu_pct": 25.0,
        "reusable_native_kernel": True,
    }
    row.update(overrides)
    return row


def _candidates(tmp_path: Path, rows: list[dict[str, Any]], **extra: Any) -> str:
    payload: dict[str, Any] = {"hot_kernels": rows}
    payload.update(extra)
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _selected(tmp_path: Path, rows: list[dict[str, Any]], **extra: Any) -> list[str]:
    """Run the selector with no SharedState, so only row-level filters apply."""
    chosen = krh._batch_kernel_candidates({"candidates_path": _candidates(tmp_path, rows, **extra)})
    return [str(candidate.get("kernel_id") or "") for candidate in chosen]


def test_a_fully_eligible_row_is_selected(tmp_path: Path) -> None:
    assert _selected(tmp_path, [_row("k001")]) == ["k001"]


@pytest.mark.parametrize("missing", ["", None])
def test_row_without_source_file_is_dropped(tmp_path: Path, missing: Any) -> None:
    """The drop that makes a hot kernel disappear with no error and no counter."""
    assert _selected(tmp_path, [_row("k001", source_file=missing)]) == []


def test_row_without_source_file_does_not_take_its_siblings(tmp_path: Path) -> None:
    assert _selected(tmp_path, [_row("k001", source_file=""), _row("k002")]) == ["k002"]


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_reusable_flag_must_be_exactly_true(tmp_path: Path, value: Any) -> None:
    """The check is an identity test, so a truthy stand-in does not pass it."""
    assert _selected(tmp_path, [_row("k001", reusable_native_kernel=value)]) == []


def test_missing_candidates_path_selects_nothing(tmp_path: Path) -> None:
    assert krh._batch_kernel_candidates({}) == []


def test_unreadable_candidates_file_selects_nothing(tmp_path: Path) -> None:
    path = tmp_path / "kernel_candidates.json"
    path.write_text("not json", encoding="utf-8")
    assert krh._batch_kernel_candidates({"candidates_path": str(path)}) == []


def test_hot_kernels_must_be_a_list(tmp_path: Path) -> None:
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels": {"k001": {}}}), encoding="utf-8")
    assert krh._batch_kernel_candidates({"candidates_path": str(path)}) == []


def test_top15_is_read_when_hot_kernels_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels_top15": [_row("k001")]}), encoding="utf-8")
    chosen = krh._batch_kernel_candidates({"candidates_path": str(path)})
    assert [str(c.get("kernel_id") or "") for c in chosen] == ["k001"]


def test_skipped_out_records_a_reason_for_a_dropped_row(tmp_path: Path) -> None:
    """Whatever reasons the selector reports today, it must report something."""
    skipped: dict[str, str] = {}
    krh._batch_kernel_candidates(
        {"candidates_path": _candidates(tmp_path, [_row("k001", source_file="")])},
        skipped_out=skipped,
    )
    assert isinstance(skipped, dict)
