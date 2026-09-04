# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Producer half of the nomination contract: build, persist, round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import nomination_request as nr


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, str]:
    """Two real files, because the builder refuses paths that do not exist."""
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = tmp_path / "kernel_candidates.json"
    candidates.write_text('{"hot_kernels": []}', encoding="utf-8")
    return {"trace_path": str(trace), "candidates_path": str(candidates)}


def _build(inputs: dict[str, str], **overrides: object) -> nr.NominationRequest:
    """Build a valid request, overriding one field at a time."""
    kwargs: dict[str, object] = {
        "lane": nr.LANE_REWRITE,
        "lane_budget_sec": 6000,
        "max_kernels": 1,
        **inputs,
    }
    kwargs.update(overrides)
    return nr.build_request(**kwargs)  # type: ignore[arg-type]


def test_build_resolves_paths_and_keeps_scalars(inputs: dict[str, str]) -> None:
    request = _build(inputs)
    assert request.lane == nr.LANE_REWRITE
    assert request.lane_budget_sec == 6000
    assert request.max_kernels == 1
    assert request.protocol_version == nr.PROTOCOL_VERSION
    assert Path(request.trace_path).is_absolute()
    assert Path(request.candidates_path).is_absolute()


def test_unknown_lane_is_rejected(inputs: dict[str, str]) -> None:
    with pytest.raises(nr.NominationRequestError, match="unknown lane"):
        _build(inputs, lane="collective")


@pytest.mark.parametrize("field", ["trace_path", "candidates_path"])
def test_missing_input_file_is_rejected(inputs: dict[str, str], tmp_path: Path, field: str) -> None:
    with pytest.raises(nr.NominationRequestError, match=field):
        _build(inputs, **{field: str(tmp_path / "absent.json")})


@pytest.mark.parametrize("field", ["trace_path", "candidates_path"])
def test_empty_input_path_is_rejected(inputs: dict[str, str], field: str) -> None:
    with pytest.raises(nr.NominationRequestError, match=f"{field} is required"):
        _build(inputs, **{field: ""})


def test_directory_is_not_accepted_for_trace(inputs: dict[str, str], tmp_path: Path) -> None:
    """A session dir holds several traces; picking one is the producer's job."""
    with pytest.raises(nr.NominationRequestError, match="trace_path is not a file"):
        _build(inputs, trace_path=str(tmp_path))


@pytest.mark.parametrize("field", ["lane_budget_sec", "max_kernels"])
@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_scalars_are_rejected(inputs: dict[str, str], field: str, value: int) -> None:
    with pytest.raises(nr.NominationRequestError, match=f"{field} must be positive"):
        _build(inputs, **{field: value})


@pytest.mark.parametrize("field", ["lane_budget_sec", "max_kernels"])
def test_non_integer_scalars_are_rejected(inputs: dict[str, str], field: str) -> None:
    with pytest.raises(nr.NominationRequestError, match=f"{field} must be an integer"):
        _build(inputs, **{field: "many"})


def test_write_then_read_round_trips(inputs: dict[str, str], tmp_path: Path) -> None:
    request = _build(inputs, trace_captured_after="abc1234")
    path = nr.write_request(tmp_path / "attempt", request)
    assert path.name == nr.REQUEST_FILENAME
    assert nr.read_request(path) == request


def test_written_payload_carries_the_protocol_version(inputs: dict[str, str], tmp_path: Path) -> None:
    path = nr.write_request(tmp_path, _build(inputs))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol_version"] == nr.PROTOCOL_VERSION


def test_unknown_protocol_version_is_refused(inputs: dict[str, str], tmp_path: Path) -> None:
    """A reader that guesses at an unknown shape is worse than one that stops."""
    path = nr.write_request(tmp_path, _build(inputs))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol_version"] = nr.PROTOCOL_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(nr.NominationRequestError, match="unsupported nomination protocol"):
        nr.read_request(path)


def test_unreadable_request_is_refused(tmp_path: Path) -> None:
    path = tmp_path / nr.REQUEST_FILENAME
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(nr.NominationRequestError, match="could not read"):
        nr.read_request(path)


def test_non_object_request_is_refused(tmp_path: Path) -> None:
    path = tmp_path / nr.REQUEST_FILENAME
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(nr.NominationRequestError, match="must be a JSON object"):
        nr.read_request(path)


def test_an_unknown_request_field_is_refused(inputs: dict[str, str], tmp_path: Path) -> None:
    """An extra field the reader does not understand is a version skew, not a
    field to swallow silently (defect 5)."""
    path = nr.write_request(tmp_path, _build(inputs))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["gpu_capability_hint"] = "gfx950"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(nr.NominationRequestError, match="unknown nomination request field"):
        nr.read_request(path)
