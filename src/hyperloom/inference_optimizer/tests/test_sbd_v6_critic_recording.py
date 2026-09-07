# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``critic`` key carries the critic agent's own run.

Unlike the robustness section, the v4 source here was real, so these tests
include the parity check that has to pass before the v4 writer is retired: the
same iteration recorded both ways must produce the same row. What differs is
only what the v4 writer did *besides* recording the fact -- minting operation
and artifact entities on the side -- which is not this key's business.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.v6_critic import collect_v6_critic
from hyperloom.inference_optimizer.breakdown.recorder import instrument
from hyperloom.inference_optimizer.breakdown.recorder.assembler import assemble_parts
from hyperloom.inference_optimizer.breakdown.recorder.critic_out import record_critic_iteration

_REVIEW = {
    "verdict": "approve",
    "summary": "the flag pays for itself",
    "ts": "2026-09-01T04:00:00+00:00",
}
_EMIT = {"topic": "backends:flag_X", "ts": "2026-09-01T04:00:01+00:00", "kb_writes": [{"kind": "point"}]}
_REQUEST = {"context": {"phase": "framework", "macro_cycle": 3}}


def _iterations(session_dir: Path) -> list[dict]:
    return collect_v6_critic(assemble_parts(session_dir, warnings=[]).get("critic"))["iterations"]


def _record(session_dir: Path, *, iter_n: int, **overrides) -> None:
    kwargs = dict(
        iter_n=iter_n,
        request=_REQUEST,
        judge_bundle={"proposals": []},
        review=_REVIEW,
        emit=_EMIT,
        workdir=session_dir / "critic-workdir" / f"{iter_n:06d}",
    )
    kwargs.update(overrides)
    record_critic_iteration(session_dir, **kwargs)


def test_an_iteration_carries_its_verdict_and_its_artifacts(tmp_path: Path) -> None:
    _record(tmp_path, iter_n=1)

    (row,) = _iterations(tmp_path)
    assert row["iter"] == 1
    assert row["verdict"] == "approve"
    assert row["topic"] == "backends:flag_X"
    assert row["summary"] == "the flag pays for itself"
    assert row["phase"] == "FRAMEWORK"
    assert row["macro_cycle"] == 3
    assert row["review_path"].endswith("review.json")
    assert row["kb_writes"] == [{"kind": "point"}]


def test_the_recorded_row_matches_what_the_v4_writer_produced(tmp_path: Path) -> None:
    """The parity gate: the fact must survive the writer being retired."""
    v6_dir = tmp_path / "v6"
    v4_dir = tmp_path / "v4"
    for session_dir in (v6_dir, v4_dir):
        session_dir.mkdir()
    _record(v6_dir, iter_n=2, workdir=v6_dir / "critic-workdir" / "000002")
    instrument.record_critic_iteration(
        v4_dir,
        iter_n=2,
        request=_REQUEST,
        judge_bundle={"proposals": []},
        review=_REVIEW,
        emit=_EMIT,
        workdir=v4_dir / "critic-workdir" / "000002",
    )

    (recorded,) = _iterations(v6_dir)
    (legacy,) = assemble_parts(v4_dir, warnings=[])["critic_robustness"]["critic_iterations"]
    assert recorded == legacy


def test_a_resumed_session_does_not_overwrite_an_earlier_iteration(tmp_path: Path) -> None:
    """``iter`` is reused across a resume, so it cannot be the key."""
    _record(tmp_path, iter_n=0, emit={**_EMIT, "topic": "first pass"})
    _record(tmp_path, iter_n=0, emit={**_EMIT, "topic": "after the resume"})

    assert [r["topic"] for r in _iterations(tmp_path)] == ["first pass", "after the resume"]


def test_re_recording_the_same_iteration_stays_one_row(tmp_path: Path) -> None:
    _record(tmp_path, iter_n=4)
    _record(tmp_path, iter_n=4)

    assert len(_iterations(tmp_path)) == 1


def test_iterations_are_ordered_as_the_agent_ran_them(tmp_path: Path) -> None:
    _record(tmp_path, iter_n=3)
    _record(tmp_path, iter_n=1)
    _record(tmp_path, iter_n=2)

    assert [r["iter"] for r in _iterations(tmp_path)] == [1, 2, 3]


def test_a_session_the_critic_never_reviewed_reports_no_iterations(tmp_path: Path) -> None:
    assert _iterations(tmp_path) == []
