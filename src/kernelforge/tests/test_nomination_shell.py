# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consumer half of the nomination contract: read, validate, delegate, count."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge import nomination as nom


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _request_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_version": nom.PROTOCOL_VERSION,
        "lane": nom.LANE_REWRITE,
        "trace_path": "/tmp/decode.trace.json",
        "candidates_path": "/tmp/kernel_candidates.json",
        "lane_budget_sec": 6000,
        "max_kernels": 2,
        "trace_captured_after": "abc1234",
    }
    payload.update(overrides)
    return payload


def _candidates_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"hot_kernels": rows}


def _row(name: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "kernel_name": name,
        "source_file": f"/repo/{name}.py",
        "gpu_pct": 1.0,
        "reason_class": "resolved",
        "attempts": 0,
        "rejected": False,
    }
    row.update(overrides)
    return row


def test_read_request_parses_every_field(tmp_path: Path) -> None:
    path = _write(tmp_path / "req.json", _request_payload())
    request = nom.read_request(path)
    assert request.lane == nom.LANE_REWRITE
    assert request.lane_budget_sec == 6000
    assert request.max_kernels == 2
    assert request.trace_captured_after == "abc1234"


def test_protocol_mismatch_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "req.json", _request_payload(protocol_version=nom.PROTOCOL_VERSION + 1))
    with pytest.raises(nom.NominationError, match="unsupported nomination protocol"):
        nom.read_request(path)


def test_unknown_lane_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "req.json", _request_payload(lane="collective"))
    with pytest.raises(nom.NominationError, match="unknown lane"):
        nom.read_request(path)


@pytest.mark.parametrize("field_name", ["lane_budget_sec", "max_kernels"])
def test_non_positive_scalars_are_refused(tmp_path: Path, field_name: str) -> None:
    path = _write(tmp_path / "req.json", _request_payload(**{field_name: 0}))
    with pytest.raises(nom.NominationError, match=f"{field_name} must be positive"):
        nom.read_request(path)


def test_an_unknown_request_field_is_refused(tmp_path: Path) -> None:
    """A field Hyperloom writes that this build does not read must not be
    swallowed: silently dropping it hides a version skew (defect 5)."""
    path = _write(tmp_path / "req.json", _request_payload(gpu_capability_hint="gfx950"))
    with pytest.raises(nom.NominationError, match="unknown nomination request field"):
        nom.read_request(path)


def test_unreadable_request_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "req.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(nom.NominationError, match="could not read nomination request"):
        nom.read_request(path)


def test_read_candidates_keeps_unresolved_rows(tmp_path: Path) -> None:
    """The unresolved rows are exactly what a real nominator is meant to rescue."""
    path = _write(
        tmp_path / "cand.json",
        _candidates_payload(
            [
                _row("hot", gpu_pct=15.0),
                _row("blind", source_file="", reason_class="graph_replay_no_args"),
            ]
        ),
    )
    candidates = nom.read_candidates(path)
    assert [candidate.kernel_name for candidate in candidates] == ["hot", "blind"]
    assert candidates[0].is_resolved is True
    assert candidates[1].is_resolved is False
    assert candidates[1].reason_class == "graph_replay_no_args"


def test_read_candidates_accepts_legacy_name_key(tmp_path: Path) -> None:
    path = _write(tmp_path / "cand.json", _candidates_payload([{"name": "legacy", "source_file": "/repo/a.py"}]))
    assert nom.read_candidates(path)[0].kernel_name == "legacy"


def test_read_candidates_drops_nameless_and_non_dict_rows(tmp_path: Path) -> None:
    path = _write(tmp_path / "cand.json", _candidates_payload([_row("keep"), {"source_file": "/x.py"}, "junk"]))  # type: ignore[list-item]
    assert [candidate.kernel_name for candidate in nom.read_candidates(path)] == ["keep"]


def test_missing_hot_kernels_array_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "cand.json", {"routable_kernels": []})
    with pytest.raises(nom.NominationError, match="no hot_kernels array"):
        nom.read_candidates(path)


def test_non_finite_gpu_pct_ranks_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "cand.json"
    path.write_text('{"hot_kernels": [{"kernel_name": "nan", "source_file": "/a.py", "gpu_pct": null}]}', "utf-8")
    assert nom.read_candidates(path)[0].gpu_pct == 0.0


def test_stub_picks_hottest_resolved_and_splits_budget(tmp_path: Path) -> None:
    request = nom.read_request(_write(tmp_path / "req.json", _request_payload(max_kernels=2)))
    candidates = nom.read_candidates(
        _write(
            tmp_path / "cand.json",
            _candidates_payload(
                [
                    _row("cold", gpu_pct=1.0),
                    _row("hottest", gpu_pct=30.0),
                    _row("warm", gpu_pct=10.0),
                ]
            ),
        )
    )
    targets = nom.nominate(request, candidates)
    assert [target.kernel_name for target in targets] == ["hottest", "warm"]
    assert [target.budget_sec for target in targets] == [3000, 3000]


def test_stub_honours_the_ceiling(tmp_path: Path) -> None:
    request = nom.read_request(_write(tmp_path / "req.json", _request_payload(max_kernels=1)))
    candidates = nom.read_candidates(
        _write(tmp_path / "cand.json", _candidates_payload([_row("a", gpu_pct=5.0), _row("b", gpu_pct=9.0)]))
    )
    targets = nom.nominate(request, candidates)
    assert [target.kernel_name for target in targets] == ["b"]
    assert targets[0].budget_sec == 6000


def test_stub_skips_unresolved_and_rejected(tmp_path: Path) -> None:
    request = nom.read_request(_write(tmp_path / "req.json", _request_payload()))
    candidates = nom.read_candidates(
        _write(
            tmp_path / "cand.json",
            _candidates_payload(
                [
                    _row("blind", source_file="", gpu_pct=99.0),
                    _row("banned", rejected=True, gpu_pct=98.0),
                    _row("usable", gpu_pct=2.0),
                ]
            ),
        )
    )
    assert [target.kernel_name for target in nom.nominate(request, candidates)] == ["usable"]


def test_empty_nomination_is_a_valid_outcome(tmp_path: Path) -> None:
    """No eligible row is a result, not a failure; the caller must not raise."""
    request = nom.read_request(_write(tmp_path / "req.json", _request_payload()))
    candidates = nom.read_candidates(
        _write(tmp_path / "cand.json", _candidates_payload([_row("blind", source_file="")]))
    )
    assert nom.nominate(request, candidates) == []


def test_summary_counts_seen_resolved_and_selected(tmp_path: Path) -> None:
    request = nom.read_request(_write(tmp_path / "req.json", _request_payload(max_kernels=1)))
    candidates = nom.read_candidates(
        _write(
            tmp_path / "cand.json",
            _candidates_payload([_row("a", gpu_pct=5.0), _row("b", gpu_pct=9.0), _row("blind", source_file="")]),
        )
    )
    targets = nom.nominate(request, candidates)
    summary = nom.summarize(candidates, targets)
    assert summary.to_dict() == {"candidates_seen": 3, "resolved": 2, "selected": 1}
