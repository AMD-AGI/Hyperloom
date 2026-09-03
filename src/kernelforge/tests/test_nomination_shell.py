# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consumer half of the nomination contract: read, validate, delegate, count."""

from __future__ import annotations

import json
from dataclasses import fields
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


def _candidates_payload(rows: list[dict[str, object]], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {"manifest_version": nom.MANIFEST_VERSION, "hot_kernels": rows}
    payload.update(overrides)
    return payload


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


def test_producer_output_round_trips_through_the_consumer(tmp_path: Path) -> None:
    """The consumer's hand-maintained ``_KNOWN_REQUEST_KEYS`` must accept every key
    the real producer writes. The producer derives its key set from the dataclass
    fields, so if a new field is added there but forgotten here, this round-trip of
    genuine producer output would over-reject it -- catching the drift the
    strict-key guard (defect 5) would otherwise turn into a silent skew (defect 5).
    """
    from hyperloom.orchestrator.kernel import nomination_request as producer

    # Every key the producer serializes -- exactly its dataclass fields, not a
    # hand-copied literal -- must be understood by the consumer's allowed-key set.
    producer_keys = {f.name for f in fields(producer.NominationRequest)}
    assert producer_keys <= nom._KNOWN_REQUEST_KEYS, (
        f"producer writes fields the consumer would reject: {sorted(producer_keys - nom._KNOWN_REQUEST_KEYS)}"
    )

    # And prove it end to end: a request the producer actually writes reads back
    # cleanly through the consumer, no "unknown field" rejection.
    request = producer.build_request(
        lane=producer.LANE_REWRITE,
        trace_path=str(_write(tmp_path / "trace.json", {})),
        candidates_path=str(_write(tmp_path / "cand.json", {"hot_kernels": []})),
        lane_budget_sec=6000,
        max_kernels=2,
    )
    path = producer.write_request(tmp_path, request)
    parsed = nom.read_request(path)
    assert parsed.lane == nom.LANE_REWRITE
    assert parsed.max_kernels == 2


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
                _row("blind", source_file="", reason_class="source_not_resolved"),
            ]
        ),
    )
    candidates = nom.read_candidates(path)
    assert [candidate.kernel_name for candidate in candidates] == ["hot", "blind"]
    assert candidates[0].is_resolved is True
    assert candidates[1].is_resolved is False
    assert candidates[1].reason_class == "source_not_resolved"


def test_read_candidates_refuses_a_row_missing_required_fields(tmp_path: Path) -> None:
    """A row the producer did not fill in is a skew, not something to guess at.

    The legacy singular ``name`` key is one such gap: ``kernel_name`` is the
    accounting identity, so a row carrying only the old spelling is refused
    rather than silently read under a different key.
    """
    path = _write(tmp_path / "cand.json", _candidates_payload([{"name": "legacy", "source_file": "/repo/a.py"}]))
    with pytest.raises(nom.NominationError, match="missing required field"):
        nom.read_candidates(path)


def test_read_candidates_refuses_an_unknown_reason_class(tmp_path: Path) -> None:
    """An unknown class means the producer classifies on rules this build lacks."""
    path = _write(tmp_path / "cand.json", _candidates_payload([_row("hot", reason_class="invented_class")]))
    with pytest.raises(nom.NominationError, match="unknown reason_class"):
        nom.read_candidates(path)


def test_every_class_the_producer_can_emit_is_accepted(tmp_path: Path) -> None:
    """The two sets are a literal on each side, so only a test ties them.

    Driving the consumer's own set would pass however far it drifted; the
    classes come from the producing contract instead.
    """
    from hyperloom.common.kernel_source_contract import KNOWN_REASON_CLASSES as produced

    rows = [_row(f"k{index}", reason_class=name) for index, name in enumerate(sorted(produced))]
    path = _write(tmp_path / "cand.json", _candidates_payload(rows))
    assert len(nom.read_candidates(path)) == len(produced)


def test_read_candidates_refuses_a_non_dict_row(tmp_path: Path) -> None:
    """A row shape this build cannot read is a skew, so it must not be dropped."""
    path = _write(tmp_path / "cand.json", _candidates_payload([_row("keep"), "junk"]))  # type: ignore[list-item]
    with pytest.raises(nom.NominationError, match="candidate row 1 is not a JSON object"):
        nom.read_candidates(path)


def test_read_candidates_refuses_a_nameless_row(tmp_path: Path) -> None:
    """The name is what a patch is reported back against; a row without one
    would vanish with no error and no counter."""
    path = _write(tmp_path / "cand.json", _candidates_payload([_row("keep"), _row("", source_file="/x.py")]))
    with pytest.raises(nom.NominationError, match="candidate row 1 has no kernel name"):
        nom.read_candidates(path)


def test_read_candidates_accepts_the_known_manifest_version(tmp_path: Path) -> None:
    path = _write(tmp_path / "cand.json", _candidates_payload([_row("hot")]))
    assert [candidate.kernel_name for candidate in nom.read_candidates(path)] == ["hot"]


def test_unknown_manifest_version_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "cand.json", _candidates_payload([_row("hot")], manifest_version=nom.MANIFEST_VERSION + 1))
    with pytest.raises(nom.NominationError, match="unsupported candidate manifest"):
        nom.read_candidates(path)


def test_absent_manifest_version_is_refused(tmp_path: Path) -> None:
    """The producer always writes the version, so its absence is a skew too --
    the same call ``read_request`` makes for ``protocol_version``."""
    path = _write(tmp_path / "cand.json", {"hot_kernels": [_row("hot")]})
    with pytest.raises(nom.NominationError, match="unsupported candidate manifest None"):
        nom.read_candidates(path)


def test_producer_manifest_round_trips_through_the_consumer(tmp_path: Path) -> None:
    """``candidate_manifest`` claims the consumer refuses a version it does not
    know, which only holds while the two constants agree."""
    from hyperloom.orchestrator.kernel import candidate_manifest as producer

    assert producer.MANIFEST_VERSION == nom.MANIFEST_VERSION
    document, _stats = producer.build_manifest(
        _write(tmp_path / "kernel_candidates.json", {"hot_kernels": [{"name": "hot", "source_file": "/repo/hot.py"}]})
    )
    candidates = nom.read_candidates(producer.write_manifest(tmp_path, document))
    assert [candidate.kernel_name for candidate in candidates] == ["hot"]


def test_missing_hot_kernels_array_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "cand.json", {"manifest_version": nom.MANIFEST_VERSION, "routable_kernels": []})
    with pytest.raises(nom.NominationError, match="no hot_kernels array"):
        nom.read_candidates(path)


def test_a_non_object_candidate_list_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "cand.json", [_row("hot")])
    with pytest.raises(nom.NominationError, match="candidate list must be a JSON object"):
        nom.read_candidates(path)


def test_non_finite_gpu_pct_ranks_as_zero(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "cand.json",
        _candidates_payload([_row("nan", gpu_pct=None)]),
    )
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


def test_stub_summary_line_describes_the_filter_it_applies() -> None:
    """The summary is the only description of the seam's selection rule, so it
    has to name the two properties the filter actually reads."""
    from kernelforge.nomination.stub import nominate_from_candidates

    assert (nominate_from_candidates.__doc__ or "").splitlines()[0] == (
        "Take the hottest already-resolved, unrejected rows and split the budget evenly."
    )


def test_stub_still_picks_a_row_that_was_already_tried(tmp_path: Path) -> None:
    """Attempt counts are orchestrator knowledge the stub deliberately ignores."""
    request = nom.read_request(_write(tmp_path / "req.json", _request_payload(max_kernels=1)))
    candidates = nom.read_candidates(
        _write(tmp_path / "cand.json", _candidates_payload([_row("tried", attempts=4, gpu_pct=9.0)]))
    )
    assert [target.kernel_name for target in nom.nominate(request, candidates)] == ["tried"]


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
