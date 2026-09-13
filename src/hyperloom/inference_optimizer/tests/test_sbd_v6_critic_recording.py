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


def _emit(topic: str, body: str) -> dict:
    """An emitted envelope shaped as the agent writes ``emit.json``."""
    return {
        "kind": "coordinator_inbox",
        "kb_writes": [{"kind": "point"}],
        "intent_envelope": {"intents": [{"intent_type": "send_message", "payload": {"topic": topic, "body_md": body}}]},
    }


_REVIEW = {
    "review_verdicts": [
        {"target_proposal_msg_id": "p1", "verdict": "approve", "reasoning": "the flag pays for itself"},
    ]
}
_EMIT = _emit("backends:flag_X", "the flag pays for itself")
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
    assert row["verdict"] == "1 approve"
    assert row["verdict_counts"] == {"approve": 1}
    assert row["topic"] == "backends:flag_X"
    assert row["summary"] == "the flag pays for itself"
    assert row["phase"] == "FRAMEWORK"
    assert row["macro_cycle"] == 3
    assert row["review_path"].endswith("review.json")
    assert row["kb_writes"] == [{"kind": "point"}]
    assert row["ts"]


def test_an_iteration_reports_how_its_rulings_fell(tmp_path: Path) -> None:
    review = {
        "review_verdicts": [
            {"target_proposal_msg_id": "p1", "verdict": "approve"},
            {"target_proposal_msg_id": "p2", "verdict": "reject"},
            {"target_proposal_msg_id": "p3", "verdict": "approve"},
        ]
    }
    _record(tmp_path, iter_n=1, review=review)

    (row,) = _iterations(tmp_path)
    assert row["verdict_counts"] == {"approve": 2, "reject": 1}
    assert row["verdict"] == "2 approve, 1 reject"


def test_a_pass_that_only_spoke_rules_on_nothing(tmp_path: Path) -> None:
    """A heartbeat turn is the common case and must not read as a verdict."""
    _record(tmp_path, iter_n=1, review={"review_verdicts": []}, emit=_emit("heartbeat", "ok (critic)"))

    (row,) = _iterations(tmp_path)
    assert row["verdict"] == ""
    assert row["verdict_counts"] == {}
    assert row["topic"] == "heartbeat"
    assert row["summary"] == "ok (critic)"


def test_a_resumed_session_does_not_overwrite_an_earlier_iteration(tmp_path: Path) -> None:
    _record(tmp_path, iter_n=0, emit=_emit("first pass", "one"))
    _record(tmp_path, iter_n=0, emit=_emit("after the resume", "two"))

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
