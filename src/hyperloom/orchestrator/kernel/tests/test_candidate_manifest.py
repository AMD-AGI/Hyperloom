# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The handover list must keep what dispatch throws away."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.kernel import candidate_manifest as cm


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


def _artifact(tmp_path: Path, rows: list[Any], key: str = "hot_kernels") -> Path:
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps({key: rows}), encoding="utf-8")
    return path


def _build(tmp_path: Path, rows: list[Any], **kwargs: Any) -> tuple[dict[str, Any], cm.ManifestStats]:
    return cm.build_manifest(_artifact(tmp_path, rows), **kwargs)


def test_a_resolved_row_is_carried_with_its_fields(tmp_path: Path) -> None:
    document, stats = _build(tmp_path, [_row("k001", gpu_pct=24.5, call_count=7, duration_us=1234.0)])
    (entry,) = document["hot_kernels"]
    assert entry["kernel_id"] == "k001"
    assert entry["kernel_name"] == "k001_kernel"
    assert entry["gpu_pct"] == 24.5
    assert entry["call_count"] == 7
    assert entry["duration_us"] == 1234.0
    assert entry["reason_class"] == "resolved"
    assert stats.resolved == 1
    assert stats.undispatchable_gpu_pct == 0.0


def test_an_undispatchable_row_is_kept_not_dropped(tmp_path: Path) -> None:
    """The whole reason this exists: dispatch drops these before forge sees them."""
    rows = [
        _row("k001"),
        _row(
            "k002", source_file="", reusable_native_kernel=False, skip_reason="source file not resolved", gpu_pct=15.0
        ),
    ]
    document, stats = _build(tmp_path, rows)
    classes = {entry["kernel_id"]: entry["reason_class"] for entry in document["hot_kernels"]}
    assert classes == {"k001": "resolved", "k002": "source_not_resolved"}
    assert stats.total == 2
    assert stats.resolved == 1
    assert stats.undispatchable_gpu_pct == 15.0


def test_unsalvageable_and_recoverable_rows_are_distinguished(tmp_path: Path) -> None:
    rows = [
        _row("k001", source_file="", reusable_native_kernel=False, skip_reason="source file not resolved"),
        _row("k002", reusable_native_kernel=False, skip_reason="vendor binary (no rewritable source)"),
        _row("k003", reusable_native_kernel=False, skip_reason="launch API, not a kernel (no rewritable body)"),
    ]
    document, _ = _build(tmp_path, rows)
    classes = [entry["reason_class"] for entry in document["hot_kernels"]]
    assert classes == ["source_not_resolved", "vendor_binary", "launch_api_only"]


def test_session_history_is_merged_in(tmp_path: Path) -> None:
    """Attempt counts and rejections live in orchestrator state, not the artifact."""
    document, stats = _build(
        tmp_path,
        [_row("k001"), _row("k002")],
        rejected_kernel_ids=["k002"],
        attempts_by_kernel_id={"k001": 2},
    )
    by_id = {entry["kernel_id"]: entry for entry in document["hot_kernels"]}
    assert by_id["k001"]["attempts"] == 2
    assert by_id["k001"]["rejected"] is False
    assert by_id["k002"]["attempts"] == 0
    assert by_id["k002"]["rejected"] is True
    assert stats.rejected == 1


def test_attempts_from_a_ledger_entry_are_read(tmp_path: Path) -> None:
    document, _ = _build(tmp_path, [_row("k001")], attempts_by_kernel_id={"k001": {"attempts": 3}})
    assert document["hot_kernels"][0]["attempts"] == 3


@pytest.mark.parametrize("value", [None, "many", -4, True, {"attempts": "x"}])
def test_an_unusable_attempt_count_reads_as_zero(tmp_path: Path, value: Any) -> None:
    document, _ = _build(tmp_path, [_row("k001")], attempts_by_kernel_id={"k001": value})
    assert document["hot_kernels"][0]["attempts"] == 0


@pytest.mark.parametrize("rejected", [None, "k002", 7])
def test_a_non_collection_rejected_argument_means_none(tmp_path: Path, rejected: Any) -> None:
    document, _ = _build(tmp_path, [_row("k001")], rejected_kernel_ids=rejected)
    assert document["hot_kernels"][0]["rejected"] is False


def test_resolution_method_and_reason_are_carried(tmp_path: Path) -> None:
    """Passed through, not recomputed, so the two views cannot disagree."""
    document, _ = _build(
        tmp_path,
        [_row("k001", source_resolution_method="active_finder", source_resolution_reason="symbol matched")],
    )
    entry = document["hot_kernels"][0]
    assert entry["resolution_method"] == "active_finder"
    assert entry["resolution_reason"] == "symbol matched"


def test_a_row_without_any_identity_is_dropped(tmp_path: Path) -> None:
    """Nothing could be reported back against it, so it could never be a patch."""
    document, stats = _build(tmp_path, [{"gpu_pct": 30.0}, _row("k001")])
    assert [entry["kernel_id"] for entry in document["hot_kernels"]] == ["k001"]
    assert stats.total == 1


def test_a_row_with_only_a_name_keeps_it_as_the_key(tmp_path: Path) -> None:
    document, _ = _build(tmp_path, [{"name": "bare_kernel", "gpu_pct": 1.0}])
    entry = document["hot_kernels"][0]
    assert entry["kernel_name"] == "bare_kernel"
    assert entry["kernel_id"] == ""


def test_a_row_with_only_an_id_falls_back_for_the_name(tmp_path: Path) -> None:
    document, _ = _build(tmp_path, [{"kernel_id": "k009", "gpu_pct": 1.0}])
    assert document["hot_kernels"][0]["kernel_name"] == "k009"


def test_non_dict_rows_are_skipped(tmp_path: Path) -> None:
    document, stats = _build(tmp_path, [_row("k001"), "junk", None, 7])
    assert stats.total == 1


def test_non_finite_measurements_read_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "kernel_candidates.json"
    path.write_text('{"hot_kernels": [{"kernel_id": "k001", "gpu_pct": null, "duration_us": "x"}]}', "utf-8")
    document, _ = cm.build_manifest(path)
    entry = document["hot_kernels"][0]
    assert entry["gpu_pct"] == 0.0
    assert entry["duration_us"] == 0.0


def test_the_top15_key_is_accepted(tmp_path: Path) -> None:
    document, _ = cm.build_manifest(_artifact(tmp_path, [_row("k001")], key="hot_kernels_top15"))
    assert len(document["hot_kernels"]) == 1


def test_an_empty_row_array_is_valid(tmp_path: Path) -> None:
    """A trace that found nothing is a fact worth telling, not an error."""
    document, stats = _build(tmp_path, [])
    assert document["hot_kernels"] == []
    assert stats.total == 0


def test_a_missing_row_array_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps({"routable_kernels": []}), encoding="utf-8")
    with pytest.raises(cm.CandidateManifestError, match="no hot_kernels array"):
        cm.build_manifest(path)


def test_an_unreadable_artifact_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "kernel_candidates.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(cm.CandidateManifestError, match="could not read"):
        cm.build_manifest(path)


def test_a_non_object_artifact_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "kernel_candidates.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(cm.CandidateManifestError, match="must be a JSON object"):
        cm.build_manifest(path)


def test_trace_provenance_is_recorded(tmp_path: Path) -> None:
    document, _ = _build(tmp_path, [_row("k001")], trace_path="/t/decode.json", trace_captured_after="abc123")
    assert document["trace_path"] == "/t/decode.json"
    assert document["trace_captured_after"] == "abc123"


def test_the_manifest_carries_its_version(tmp_path: Path) -> None:
    document, _ = _build(tmp_path, [_row("k001")])
    assert document["manifest_version"] == cm.MANIFEST_VERSION


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    document, _ = _build(tmp_path, [_row("k001")])
    path = cm.write_manifest(tmp_path / "attempt", document)
    assert path.name == cm.MANIFEST_FILENAME
    assert json.loads(path.read_text(encoding="utf-8")) == document


def test_the_forge_consumer_can_read_what_we_write(tmp_path: Path) -> None:
    """Producer and consumer are mirrored modules; this is the seam between them."""
    from kernelforge import nomination as nom

    document, _ = _build(
        tmp_path,
        [
            _row("k001", gpu_pct=30.0),
            _row("k002", source_file="", reusable_native_kernel=False, skip_reason="source file not resolved"),
        ],
        rejected_kernel_ids=["k002"],
    )
    path = cm.write_manifest(tmp_path / "attempt", document)
    candidates = nom.read_candidates(path)
    assert [candidate.kernel_name for candidate in candidates] == ["k001_kernel", "k002_kernel"]
    assert candidates[0].is_resolved is True
    assert candidates[1].is_resolved is False
    assert candidates[1].reason_class == "source_not_resolved"
    assert candidates[1].rejected is True
