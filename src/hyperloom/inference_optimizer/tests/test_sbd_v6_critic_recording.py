# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``critic`` key carries the critic agent's own run.

The v4 writer this replaces was real, so these rows had to reproduce it exactly
before it could be retired; the operation and artifact entities it also minted on
the side were never this key's business.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.v6_critic import collect_v6_critic
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


def test_a_resumed_session_does_not_overwrite_an_earlier_iteration(tmp_path: Path) -> None:
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
