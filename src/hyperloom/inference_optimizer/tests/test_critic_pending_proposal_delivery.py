# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Durable, at-least-once-until-decided delivery of proposals to the Critic.

``ConversationCollaborator._augment_critic_inbox_with_pending`` re-presents every
still-undecided proposal from the durable ``pending_proposals`` registry until it
is decided, independent of the tail/cursor. These tests pin that behaviour AND
that the re-presented row is parseable by the real Critic inbox parser.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from hyperloom.agents.critic.runtime.inbox_parser import parse_inbox_prompt
from hyperloom.orchestrator.bus.message_bus import Message
from hyperloom.orchestrator.loop.conversation import ConversationCollaborator
from hyperloom.orchestrator.loop.coordinator import (
    PendingProposal,
    _format_inbox_event,
)

_PROP_ID = "deadbeef01"


def _msg(seq: int, topic: str, payload: dict, *, msg_id: str | None = None) -> Message:
    m = Message.new("coordinator", "*", topic, payload)
    m.seq = seq
    if msg_id:
        m.msg_id = msg_id
    return m


def _make_conv(*, decided: bool):
    """A ConversationCollaborator wired to a minimal fake Coordinator.

    The bus holds one buried proposal (``seq=5``); the durable registry tracks
    it as an (un)decided ``framework_agent`` proposal.
    """
    prop = _msg(
        5,
        "proposal",
        {
            "action_name": "framework_agent",
            "framework_agent_candidate_id": "https://github.com/x/y/pull/1",
            "needs_review": True,
        },
        msg_id=_PROP_ID,
    )

    class _FakeBus:
        async def lookup_by_id(self, mid: str):
            return prop if mid == _PROP_ID else None

    coord = types.SimpleNamespace(
        state=types.SimpleNamespace(
            pending_proposals={
                _PROP_ID: PendingProposal(
                    proposal_msg_id=_PROP_ID,
                    from_agent="coordinator",
                    action_name="framework_agent",
                    predicted_gain_pct=0.0,
                    payload=dict(prop.payload),
                    decided=decided,
                )
            }
        ),
        bus=_FakeBus(),
    )
    return ConversationCollaborator(coord), prop


def _tail_noise(n: int = 40) -> list[Message]:
    """``n`` later observation messages that crowd the proposal out of the tail."""
    return [_msg(s, "observation", {"kind": "noise", "n": s}) for s in range(6, 6 + n)]


def test_buried_undecided_proposal_is_re_presented() -> None:
    conv, prop = _make_conv(decided=False)
    tail = _tail_noise()[-20:]  # mirrors the inbox cap; proposal not present
    assert all(m.topic != "proposal" for m in tail)

    merged = asyncio.run(conv._augment_critic_inbox_with_pending(list(tail)))

    proposal_rows = [m for m in merged if m.topic == "proposal"]
    assert len(proposal_rows) == 1
    assert proposal_rows[0].msg_id == prop.msg_id
    # Re-sorted by seq so "newest last" holds — the buried proposal leads.
    assert [m.seq for m in merged] == sorted(m.seq for m in merged)
    assert merged[0].seq == 5


def test_re_presented_proposal_is_parseable_by_critic() -> None:
    conv, _prop = _make_conv(decided=False)
    merged = asyncio.run(
        conv._augment_critic_inbox_with_pending(list(_tail_noise()[-20:]))
    )
    lines = ["=== Inbox for critic (newest last) ==="]
    lines += [f"  {_format_inbox_event(m)}" for m in merged]
    parsed = parse_inbox_prompt("\n".join(lines))
    assert len(parsed.proposals) == 1


def test_decided_proposal_is_not_re_presented() -> None:
    conv, _prop = _make_conv(decided=True)
    tail = _tail_noise()[-20:]
    merged = asyncio.run(conv._augment_critic_inbox_with_pending(list(tail)))
    assert all(m.topic != "proposal" for m in merged)
    assert len(merged) == len(tail)


def test_already_visible_proposal_is_not_duplicated() -> None:
    conv, prop = _make_conv(decided=False)
    # Proposal already in the rendered tail — must not be added twice.
    merged = asyncio.run(conv._augment_critic_inbox_with_pending([prop]))
    assert sum(1 for m in merged if m.msg_id == prop.msg_id) == 1


def test_no_pending_returns_tail_unchanged() -> None:
    conv, _prop = _make_conv(decided=True)
    tail = _tail_noise(3)
    out = asyncio.run(conv._augment_critic_inbox_with_pending(list(tail)))
    assert out == tail


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
