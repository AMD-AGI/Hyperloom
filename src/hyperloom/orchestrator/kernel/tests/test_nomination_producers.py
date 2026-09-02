# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The three producer helpers that hand forge its ``--auto`` brief.

These are the Hyperloom half of the nomination contract: project the candidate
list into the manifest forge reads, derive the rewrite lane's budget from the
time actually left, and write the request that ties them together. They are the
seam the ``auto=true`` handler composes; here each is exercised in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.kernel import nomination_request as nr
from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.state.shared_state import SharedState


def _row(kernel_id: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kernel_id": kernel_id,
        "name": f"{kernel_id}_kernel",
        "gpu_pct": 10.0,
        "source_file": f"/repo/{kernel_id}.py",
        "reusable_native_kernel": True,
        "skip_reason": "",
    }
    row.update(overrides)
    return row


def _candidates(session_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = session_dir / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels": rows}), encoding="utf-8")
    return path


def _state(session_dir: Path, *, max_minutes: float | None = 600.0, **fields: Any) -> SharedState:
    state = SharedState.load_or_init(session_dir)
    if max_minutes is not None:
        state.max_minutes = max_minutes
    for key, value in fields.items():
        setattr(state, key, value)
    state.save(session_dir)
    return state


# --------------------------------------------------------------------------- #
# 12a — manifest emission
# --------------------------------------------------------------------------- #
def test_manifest_keeps_the_unroutable_row_and_merges_history(tmp_path: Path) -> None:
    rows = [
        _row("k001", gpu_pct=30.0),
        _row("k002", source_file="", reusable_native_kernel=False, skip_reason="source file not resolved"),
    ]
    candidates = _candidates(tmp_path, rows)
    _state(
        tmp_path,
        rejected_kernel_ids=["k002"],
        # The ledger is keyed by a stable task key; the manifest re-indexes it by
        # the ordinal ``current_kernel_id`` each entry currently occupies.
        kernel_opt_task_attempts={"task-a": {"current_kernel_id": "k001", "attempts": 2}},
    )
    manifest_path = krh._write_forge_candidate_manifest(
        {"candidates_path": str(candidates)}, session_dir=tmp_path
    )
    assert manifest_path is not None
    assert manifest_path.name == "forge_candidate_manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {entry["kernel_id"]: entry for entry in document["hot_kernels"]}
    # The superset: the unroutable row survives, and session history is merged.
    assert set(by_id) == {"k001", "k002"}
    assert by_id["k002"]["reason_class"] == "source_not_resolved"
    assert by_id["k002"]["rejected"] is True
    assert by_id["k001"]["attempts"] == 2


def test_manifest_is_readable_by_the_forge_consumer(tmp_path: Path) -> None:
    from kernelforge import nomination as nom

    candidates = _candidates(tmp_path, [_row("k001", gpu_pct=30.0)])
    _state(tmp_path)
    manifest_path = krh._write_forge_candidate_manifest(
        {"candidates_path": str(candidates)}, session_dir=tmp_path
    )
    parsed = nom.read_candidates(manifest_path)
    assert [candidate.kernel_name for candidate in parsed] == ["k001_kernel"]


def test_manifest_is_none_without_a_candidate_path(tmp_path: Path) -> None:
    _state(tmp_path)
    assert krh._write_forge_candidate_manifest({}, session_dir=tmp_path) is None


# --------------------------------------------------------------------------- #
# 12b — rewrite lane budget
# --------------------------------------------------------------------------- #
def test_rewrite_budget_funds_targets_from_remaining_time(tmp_path: Path) -> None:
    # 600 min - 5 min reserve = 595 min; rewrite gets 50% = ~17850s;
    # max_targets = 17850 // 4500 = 3.
    state = _state(tmp_path, max_minutes=600.0)
    allocation = krh._nomination_lane_budget(state)
    assert allocation.lane == "rewrite"
    assert allocation.is_fundable
    assert allocation.max_targets == allocation.budget_sec // 4500


def test_unbounded_session_is_not_fundable(tmp_path: Path) -> None:
    state = _state(tmp_path, max_minutes=None)
    allocation = krh._nomination_lane_budget(state)
    assert allocation.budget_sec == 0
    assert allocation.max_targets == 0
    assert not allocation.is_fundable


# --------------------------------------------------------------------------- #
# 12c — nomination request producer
# --------------------------------------------------------------------------- #
def test_request_points_candidates_path_at_the_manifest(tmp_path: Path) -> None:
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", gpu_pct=30.0)])
    state = _state(tmp_path, last_profile_trace=str(trace))
    manifest_path = krh._write_forge_candidate_manifest(
        {"candidates_path": str(candidates)}, session_dir=tmp_path
    )
    allocation = krh._nomination_lane_budget(state)
    request_path = krh._write_nomination_request(
        {},
        session_dir=tmp_path,
        manifest_path=manifest_path,
        allocation=allocation,
    )
    assert request_path.name == "forge_nomination_input.json"
    request = nr.read_request(request_path)
    assert request.lane == "rewrite"
    # The load-bearing decision: forge reads the MANIFEST, not the raw artifact.
    assert Path(request.candidates_path) == manifest_path.resolve()
    assert Path(request.trace_path) == trace.resolve()
    assert request.lane_budget_sec == allocation.budget_sec
    assert request.max_kernels == allocation.max_targets
    assert request.protocol_version == 1


def test_request_rejects_a_missing_trace(tmp_path: Path) -> None:
    candidates = _candidates(tmp_path, [_row("k001")])
    state = _state(tmp_path)  # no last_profile_trace → trace resolves to ""
    manifest_path = krh._write_forge_candidate_manifest(
        {"candidates_path": str(candidates)}, session_dir=tmp_path
    )
    allocation = krh._nomination_lane_budget(state)
    with pytest.raises(nr.NominationRequestError, match="trace_path"):
        krh._write_nomination_request(
            {}, session_dir=tmp_path, manifest_path=manifest_path, allocation=allocation
        )
