# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Critic per-variant ``verdict_map`` tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.loop.coordinator import (
    Coordinator,
    CoordinatorState,
    PendingProposal,
)
from hyperloom.inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
    IntentValidationError,
    validate_envelope,
)
from hyperloom.orchestrator.loop.coordinator_helpers import (
    collapse_verdict_map,
    collapse_verdicts,
    proceedable_variant_names,
    verdict_held_to_its_rule,
    verdict_map_entry_grounds,
    verdict_map_entry_held_to_its_rule,
)
from hyperloom.orchestrator.policy.gate import (
    INTEGRATE_PATCH_PERMISSIVE_VERDICTS,
    PolicyDenied,
    PolicyGate,
    REVIEW_VERDICTS,
)
from hyperloom.orchestrator.specialists.patch_safety import (
    QUANTITATIVE_CLAIM_REASON_CODE,
    cross_domain_rule_descriptors,
)


# 1. intent_parser — envelope schema accepts verdict OR verdict_map
def _envelope(**payload: Any) -> dict[str, Any]:
    return {
        "intents": [
            {
                "intent_type": "review_verdict",
                "payload": payload,
            }
        ],
    }


def test_intent_parser_accepts_legacy_single_verdict():
    intents = validate_envelope(
        _envelope(
            target_proposal_msg_id="msg-1",
            verdict="approve",
            reasoning="ok",
        )
    )
    assert len(intents) == 1
    assert intents[0].type is IntentType.REVIEW_VERDICT
    assert intents[0].payload["verdict"] == "approve"


def test_intent_parser_accepts_per_variant_verdict_map():
    intents = validate_envelope(
        _envelope(
            target_proposal_msg_id="msg-1",
            verdict_map={
                "v_a": {"verdict": "approve", "rationale": "looks promising"},
                "v_b": {"verdict": "reject", "rationale": "kb says no"},
            },
        )
    )
    assert intents[0].payload["verdict_map"]["v_a"]["verdict"] == "approve"


def test_intent_parser_rejects_both_verdict_and_verdict_map():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(
            _envelope(
                target_proposal_msg_id="msg-1",
                verdict="approve",
                verdict_map={"v_a": {"verdict": "reject"}},
            )
        )
    assert "mutually exclusive" in str(exc.value)


def test_intent_parser_rejects_neither_verdict_nor_verdict_map():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(_envelope(target_proposal_msg_id="msg-1"))
    assert "must include either" in str(exc.value)


def test_intent_parser_rejects_empty_verdict_map():
    with pytest.raises(IntentValidationError):
        validate_envelope(
            _envelope(
                target_proposal_msg_id="msg-1",
                verdict_map={},
            )
        )


def test_intent_parser_rejects_verdict_map_entry_missing_verdict():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(
            _envelope(
                target_proposal_msg_id="msg-1",
                verdict_map={"v_a": {"rationale": "no verdict key"}},
            )
        )
    assert "missing required 'verdict'" in str(exc.value)


def test_intent_parser_rejects_non_dict_verdict_map_entry():
    with pytest.raises(IntentValidationError):
        validate_envelope(
            _envelope(
                target_proposal_msg_id="msg-1",
                verdict_map={"v_a": "approve"},
            )
        )


# 2. PolicyGate — verdict_map content validation
@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


def _critic_intent(**payload: Any) -> Intent:
    return Intent(type=IntentType.REVIEW_VERDICT, payload=payload)


def test_policy_gate_accepts_legacy_single_verdict(gate):
    gate.validate_intent(
        "critic",
        _critic_intent(
            target_proposal_msg_id="msg-1",
            verdict="approve",
        ),
    )


def test_policy_gate_accepts_per_variant_verdict_map(gate):
    gate.validate_intent(
        "critic",
        _critic_intent(
            target_proposal_msg_id="msg-1",
            verdict_map={
                "v_a": {"verdict": "approve"},
                "v_b": {"verdict": "reject", "rationale": "no"},
            },
        ),
    )


def test_policy_gate_rejects_when_both_present(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic",
            _critic_intent(
                target_proposal_msg_id="msg-1",
                verdict="approve",
                verdict_map={"v_a": {"verdict": "approve"}},
            ),
        )
    assert exc.value.rule == "payload"
    # Either "exactly one" (PolicyGate) or "mutually exclusive" (intent_parser) is valid.
    msg = str(exc.value) + " " + (exc.value.hint or "")
    assert "exactly one" in msg or "mutually exclusive" in msg


def test_policy_gate_rejects_when_neither_present(gate):
    """Defense in depth — PolicyGate still rejects even though intent_parser fires first."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic",
            _critic_intent(
                target_proposal_msg_id="msg-1",
            ),
        )
    assert exc.value.rule == "payload"


def test_the_per_variant_shape_the_gate_teaches_carries_the_cited_rule(gate):
    """The gate's hint is where the per-variant entry shape is spelled out for
    the emitter, so it is where a variant learns it can name the rule its
    verdict rests on rather than leaving that to prose."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("critic", _critic_intent(target_proposal_msg_id="msg-1"))

    assert "failure_reason_code" in (exc.value.hint or "")


def test_policy_gate_rejects_unknown_per_variant_verdict(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic",
            _critic_intent(
                target_proposal_msg_id="msg-1",
                verdict_map={
                    "v_a": {"verdict": "approve"},
                    "v_b": {"verdict": "obliterate"},  # not a valid verdict
                },
            ),
        )
    assert "verdict_map" in str(exc.value)
    assert "obliterate" in str(exc.value)


def test_policy_gate_review_verdicts_vocab_contains_canonical_set():
    """REVIEW_VERDICTS must cover the five canonical strings the Critic prompt documents."""
    for v in ("approve", "reject", "redirect", "advise", "needs_review"):
        assert v in REVIEW_VERDICTS


# 3. Coordinator helpers — _handle_review_verdict dispatch
@dataclass
class _BareSharedState:
    """SharedState double exposing the fields the review-verdict handler touches."""

    recipe_kb_session_id: str = "sid-test"
    save_count: int = 0
    # Empty string means "nothing in flight"; the auto-roofline dispatch gate is a no-op.
    auto_roofline_pending_task_id: str = ""

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1


@dataclass
class _BusMessage:
    from_agent: str
    to_agent: str
    topic: str
    payload: dict[str, Any]
    in_reply_to: str = ""
    priority: int = 1
    msg_id: str = ""


class _StubBus:
    """MessageBus double — captures every appended message."""

    def __init__(self) -> None:
        self.messages: list[_BusMessage] = []

    async def append_and_seq(self, msg: Any) -> Any:  # noqa: ANN401
        self.messages.append(
            _BusMessage(
                from_agent=getattr(msg, "from_agent", ""),
                to_agent=getattr(msg, "to_agent", ""),
                topic=getattr(msg, "topic", ""),
                payload=dict(getattr(msg, "payload", {}) or {}),
                in_reply_to=getattr(msg, "in_reply_to", "") or "",
                priority=int(getattr(msg, "priority", 1) or 1),
                msg_id=getattr(msg, "msg_id", ""),
            )
        )
        return None


class _StubRecipeKB:
    enabled: bool = True

    def __init__(self) -> None:
        self.verify_calls: list[dict[str, Any]] = []

    def verify(self, **kwargs: Any) -> None:
        self.verify_calls.append(dict(kwargs))


@pytest.fixture
def coord(tmp_path: Path):
    """Coordinator-shaped object with just enough plumbing for the review-verdict path."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareSharedState()
    c.state = CoordinatorState()
    c.recipe_kb = _StubRecipeKB()
    c.bus = _StubBus()
    c._record_observation = AsyncMock()  # type: ignore[method-assign]
    materialise_calls: list[tuple[PendingProposal, set[str] | None]] = []
    c._materialise_calls = materialise_calls  # type: ignore[attr-defined]

    async def _mat(
        pending: PendingProposal,
        *,
        approved_variant_names: set[str] | None = None,
    ) -> None:
        materialise_calls.append((pending, approved_variant_names))

    c._materialize_approved_proposal = _mat  # type: ignore[method-assign]
    return c


def _seed_explore_proposal(
    coord: Coordinator,
    *,
    msg_id: str = "msg-1",
    variants: list[str] | None = None,
) -> PendingProposal:
    variants = variants or ["v_a", "v_b", "v_c", "v_d"]
    grid = [{"name": vn, "extra_args": f"--flag-{vn}"} for vn in variants]
    pending = PendingProposal(
        proposal_msg_id=msg_id,
        from_agent="orchestration",
        action_name="explore",
        predicted_gain_pct=1.0,
        payload={"action_name": "explore", "params": {"grid": grid}},
    )
    coord.state.pending_proposals[msg_id] = pending
    return pending


# routing
@pytest.mark.asyncio
async def test_legacy_single_verdict_still_materialises_whole_proposal(coord):
    """Non-grid actions keep the single-verdict path: approve materialises the whole proposal."""
    pending = PendingProposal(
        proposal_msg_id="msg-kernel",
        from_agent="orchestration",
        action_name="kernel_opt",
        predicted_gain_pct=2.0,
        payload={"action_name": "kernel_opt", "params": {}},
    )
    coord.state.pending_proposals["msg-kernel"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={"target_proposal_msg_id": "msg-kernel", "verdict": "approve"},
    )
    await coord._handle_review_verdict("critic", intent)
    bus_msgs = [m for m in coord.bus.messages if m.topic == "review_verdict"]
    assert len(bus_msgs) == 1
    assert bus_msgs[0].payload["verdict"] == "approve"
    assert "verdict_map" not in bus_msgs[0].payload
    assert len(coord._materialise_calls) == 1
    assert coord._materialise_calls[0][1] is None
    assert pending.decided is True
    assert pending.verdict == "approve"


@pytest.mark.asyncio
async def test_verdict_map_collapses_to_summary_single_verdict(coord):
    """A mixed map proceeds on the approved subset, not the whole grid."""
    pending = _seed_explore_proposal(coord)
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict_map": {
                "v_a": {"verdict": "approve"},
                "v_b": {"verdict": "reject", "rationale": "kb says no"},
                "v_c": {"verdict": "reject", "rationale": "duplicate"},
                "v_d": {"verdict": "reject"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.decided is True
    assert pending.verdict == "approve"
    assert len(coord._materialise_calls) == 1
    assert coord._materialise_calls[0][1] == {"v_a"}
    bus_msgs = [m for m in coord.bus.messages if m.topic == "review_verdict"]
    assert len(bus_msgs) == 1
    assert bus_msgs[0].payload["verdict"] == "approve"
    assert "verdict_map" not in bus_msgs[0].payload


@pytest.mark.asyncio
async def test_verdict_map_mixed_collapse_logs_audit(coord, caplog):
    """A mixed map still logs the collapse, and materialises only the proceedable names."""
    import logging

    pending = _seed_explore_proposal(coord)
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict_map": {
                "v_a": {"verdict": "approve"},
                "v_b": {"verdict": "reject"},
            },
        },
    )
    with caplog.at_level(logging.WARNING, logger="hyperloom.orchestrator.loop.intent_router"):
        await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "approve"
    assert len(coord._materialise_calls) == 1
    assert coord._materialise_calls[0][1] == {"v_a"}
    assert any("review_verdict collapse" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_verdict_map_all_rejected_collapses_to_reject(coord):
    pending = _seed_explore_proposal(coord)
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict_map": {
                "v_a": {"verdict": "reject"},
                "v_b": {"verdict": "reject"},
                "v_c": {"verdict": "reject"},
                "v_d": {"verdict": "reject"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_a_genuine_reject_does_not_sink_advised_siblings(coord):
    """A substantive reject on one variant must not discard siblings the Critic advised through."""
    pending = _seed_explore_proposal(coord, msg_id="msg-partial", variants=["v_a", "v_b", "v_c"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict_map": {
                "v_a": {"verdict": "advise", "rationale": "worth a look"},
                "v_b": {"verdict": "reject", "rationale": "kb says no"},
                "v_c": {"verdict": "approve"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "approve"
    assert len(coord._materialise_calls) == 1
    assert coord._materialise_calls[0][1] == {"v_a", "v_c"}


@pytest.mark.asyncio
async def test_verdict_for_unknown_proposal_logs_observation(coord):
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "ghost-msg",
            "verdict": "approve",
        },
    )
    await coord._handle_review_verdict("critic", intent)
    coord._record_observation.assert_awaited()
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_single_verdict_rebroadcast_carries_full_advisory_fieldset(coord):
    """The rebroadcast payload and the compact inbox line both flow through the one serializer, carrying the full advisory field set."""
    from hyperloom.orchestrator.loop.coordinator import _format_inbox_event
    from hyperloom.orchestrator.bus.message_bus import Message

    pending = PendingProposal(
        proposal_msg_id="msg-adv",
        from_agent="orchestration",
        action_name="kernel_opt",
        predicted_gain_pct=2.0,
        payload={"action_name": "kernel_opt", "params": {}},
    )
    coord.state.pending_proposals["msg-adv"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-adv",
            "verdict": "advise",
            "reasoning": "proceed with care",
            # An empty entry must be dropped by the serializer.
            "required_evidence": ["bench at conc=64", "isolate decode path", ""],
            "risks": [{"severity": "high", "text": "may regress decode"}],
            "advice_text": "tune --max-running-requests",
            "alternative_action": "explore",
            "notes": ["see kb recipe r12"],
            "kb_evidence": ["kb://r12"],
            "packet_evidence": ["pkt://9"],
        },
    )
    await coord._handle_review_verdict("critic", intent)

    bus_msgs = [m for m in coord.bus.messages if m.topic == "review_verdict"]
    assert len(bus_msgs) == 1
    payload = bus_msgs[0].payload
    assert payload["verdict"] == "advise"
    assert payload["required_evidence"] == ["bench at conc=64", "isolate decode path"]
    assert payload["risks"] == [{"severity": "high", "text": "may regress decode"}]
    assert payload["advice_text"] == "tune --max-running-requests"
    assert payload["alternative_action"] == "explore"
    assert payload["notes"] == ["see kb recipe r12"]
    assert payload["kb_evidence"] == ["kb://r12"]
    assert payload["packet_evidence"] == ["pkt://9"]
    # advise materialises like approve.
    assert len(coord._materialise_calls) == 1

    msg = Message.new("critic", "orchestration", "review_verdict", payload)
    msg.seq = 5
    line = _format_inbox_event(msg)
    assert "\n" not in line
    assert "verdict='advise'" in line
    assert "required_evidence[2]=" in line
    assert "risks=1" in line
    assert "advice=" in line


@pytest.mark.asyncio
async def test_single_verdict_without_advisory_keeps_bare_payload(coord):
    """A verdict with no advisory fields rebroadcasts only verdict/reasoning, and the inbox line stays minimal."""
    from hyperloom.orchestrator.loop.coordinator import _format_inbox_event
    from hyperloom.orchestrator.bus.message_bus import Message

    pending = PendingProposal(
        proposal_msg_id="msg-bare",
        from_agent="orchestration",
        action_name="kernel_opt",
        predicted_gain_pct=2.0,
        payload={"action_name": "kernel_opt", "params": {}},
    )
    coord.state.pending_proposals["msg-bare"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-bare",
            "verdict": "approve",
            "reasoning": "looks good",
        },
    )
    await coord._handle_review_verdict("critic", intent)
    payload = [m for m in coord.bus.messages if m.topic == "review_verdict"][0].payload
    for key in ("required_evidence", "risks", "advice_text", "notes"):
        assert key not in payload

    msg = Message.new("critic", "orchestration", "review_verdict", payload)
    msg.seq = 6
    line = _format_inbox_event(msg)
    assert "required_evidence" not in line
    assert "risks=" not in line
    assert "advice=" not in line


# 3b. A reject on a rule that asked for advice is held to that rule
@pytest.mark.asyncio
async def test_reject_on_an_advisory_only_rule_is_held_to_advise(coord):
    """The quantitative-claim rule declares ``advise``; a reject citing it must not end the proposal."""
    pending = PendingProposal(
        proposal_msg_id="msg-held",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-1"}},
    )
    coord.state.pending_proposals["msg-held"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-held",
            "verdict": "reject",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
            "reasoning": "proposal carried confidence",
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "advise"
    # advise materialises, so the round keeps the proposal.
    assert len(coord._materialise_calls) == 1
    assert [m for m in coord.bus.messages if m.topic == "review_verdict"][0].payload["verdict"] == "advise"


@pytest.mark.asyncio
async def test_a_held_reject_is_recorded_not_silently_corrected(coord, caplog):
    """The downgrade leaves both a log line and an observation, so prompt drift stays visible."""
    import logging

    _seed_explore_proposal(coord, msg_id="msg-audit")
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-audit",
            "verdict": "reject",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
        },
    )
    with caplog.at_level(logging.WARNING, logger="hyperloom.orchestrator.loop.intent_router"):
        await coord._handle_review_verdict("critic", intent)
    assert any("held to its rule" in r.getMessage() for r in caplog.records)
    kinds = [call.args[2].get("kind") for call in coord._record_observation.await_args_list]
    assert "verdict_downgraded_to_rule_verdict" in kinds


@pytest.mark.asyncio
async def test_a_rule_named_only_in_prose_still_holds_the_verdict(coord):
    """Field shape: the Critic names its rule in ``reasoning``, never in ``failure_reason_code``.

    ``failure_reason_code`` is an input descriptor — nothing on the output side
    requires it — so a verdict that cites the rule in prose is in contract and
    must move the same way one carrying the field does.
    """
    pending = PendingProposal(
        proposal_msg_id="msg-prose",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {}},
    )
    coord.state.pending_proposals["msg-prose"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-prose",
            "verdict": "reject",
            "reasoning": (
                f"{QUANTITATIVE_CLAIM_REASON_CODE}: the proposal payload carries "
                "the forbidden predicted_gain_pct field."
            ),
            "risks": [
                {
                    "severity": "blocker",
                    "summary": "Specialist proposal payload contains a prohibited quantitative claim field.",
                }
            ],
            "notes": ["Resubmit without predicted_gain_pct or other prohibited quantitative ranking fields."],
            "packet_evidence": ["payload.predicted_gain_pct"],
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "advise"
    assert len(coord._materialise_calls) == 1
    kinds = [call.args[2].get("kind") for call in coord._record_observation.await_args_list]
    assert "verdict_downgraded_to_rule_verdict" in kinds


@pytest.mark.asyncio
async def test_prose_that_cites_no_rule_leaves_the_reject_alone(coord):
    """Prose is scanned for a rule citation, not read for sentiment; an ordinary reject stands."""
    pending = PendingProposal(
        proposal_msg_id="msg-prose-plain",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-4"}},
    )
    coord.state.pending_proposals["msg-prose-plain"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-prose-plain",
            "verdict": "reject",
            "reasoning": "the patch rewrites a kernel with no before/after benchmark to stand on.",
            "notes": ["predicted_gain_pct was not the problem here."],
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_a_declared_reject_code_outranks_an_advisory_one_in_prose(coord):
    """The field is the Critic's explicit citation; a citation in prose cannot soften a rule it declared ``reject``."""
    pending = PendingProposal(
        proposal_msg_id="msg-both",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-5"}},
    )
    coord.state.pending_proposals["msg-both"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-both",
            "verdict": "reject",
            "failure_reason_code": "specialist_patch_not_grounded",
            "reasoning": f"{QUANTITATIVE_CLAIM_REASON_CODE}: the payload carries predicted_gain_pct.",
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@dataclass
class _PatchVerdictSharedState(_BareSharedState):
    """Adds the patch-verdict mirror the integrate_patch gate reads."""

    patch_verdicts: dict[str, str] = field(default_factory=dict)

    def record_specialist_patch_verdict(self, specialist_task_id: str, verdict: str) -> None:
        self.patch_verdicts[specialist_task_id] = verdict.strip().lower()


@pytest.mark.asyncio
async def test_a_held_reject_is_not_a_landing_permit(coord):
    """``advise`` is an integrate_patch permit, so mirroring the held verdict
    turned "the Critic rejected this patch" into "the Critic waved it through"
    -- over a formatting rule, and irreversibly."""
    coord.shared_state = _PatchVerdictSharedState()
    pending = PendingProposal(
        proposal_msg_id="msg-permit",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-permit"}},
    )
    coord.state.pending_proposals["msg-permit"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-permit",
            "verdict": "reject",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "advise"
    assert len(coord._materialise_calls) == 1
    assert coord.shared_state.patch_verdicts["t-permit"] == "reject"
    assert coord.shared_state.patch_verdicts["t-permit"] not in INTEGRATE_PATCH_PERMISSIVE_VERDICTS


@pytest.mark.asyncio
async def test_a_held_variant_mirrors_the_verdict_the_critic_wrote(coord):
    """The mirror is a landing permit for the specialist's patches, and the
    per-variant path reaches it the same way the single one does: what the
    Critic wrote is mirrored, whatever the hold made of it for this round."""
    coord.shared_state = _PatchVerdictSharedState()
    pending = PendingProposal(
        proposal_msg_id="msg-map-permit",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-map"}},
    )
    coord.state.pending_proposals["msg-map-permit"] = pending
    entry = {"verdict": "reject", "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE}
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-permit",
            "verdict_map": {"v_a": dict(entry), "v_b": dict(entry)},
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "advise"
    assert len(coord._materialise_calls) == 1
    assert coord.shared_state.patch_verdicts["t-map"] == "reject"


@pytest.mark.asyncio
async def test_an_unheld_verdict_still_mirrors_itself(coord):
    """The mirror only diverges from the acted-on verdict when a hold moved it."""
    coord.shared_state = _PatchVerdictSharedState()
    pending = PendingProposal(
        proposal_msg_id="msg-plain-permit",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-plain"}},
    )
    coord.state.pending_proposals["msg-plain-permit"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={"target_proposal_msg_id": "msg-plain-permit", "verdict": "advise"},
    )
    await coord._handle_review_verdict("critic", intent)

    assert coord.shared_state.patch_verdicts["t-plain"] == "advise"


@pytest.mark.asyncio
async def test_a_reject_that_also_names_a_second_risk_is_not_held(coord):
    """The hold answers "the whole reject was this one rule"; a verdict that
    also refuses on its own merits keeps both halves of the sentence."""
    pending = PendingProposal(
        proposal_msg_id="msg-mixed",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-mixed"}},
    )
    coord.state.pending_proposals["msg-mixed"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-mixed",
            "verdict": "reject",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
            "risks": [
                {"severity": "minor", "summary": "payload carries a self-reported gain."},
                {"severity": "blocker", "summary": "the patch can only be rolled back by hand."},
            ],
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_a_reject_still_asking_for_evidence_is_not_held(coord):
    """An outstanding evidence request is a ground of its own: the Critic is
    not complaining about a field, it is saying it cannot judge yet."""
    pending = PendingProposal(
        proposal_msg_id="msg-eviden",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-evid"}},
    )
    coord.state.pending_proposals["msg-eviden"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-eviden",
            "verdict": "reject",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
            "required_evidence": ["matched_benchmark"],
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_a_rule_the_critic_cleared_is_not_the_grounds_for_its_reject(coord):
    """The prose scan used to fire on any word-bounded mention, so a Critic that
    checked the advisory rule, found it clean, and refused for a real reason had
    that reason read as the formatting complaint -- and the proposal ran."""
    pending = PendingProposal(
        proposal_msg_id="msg-cleared",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-cleared"}},
    )
    coord.state.pending_proposals["msg-cleared"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-cleared",
            "verdict": "reject",
            "reasoning": (
                f"Checked {QUANTITATIVE_CLAIM_REASON_CODE}: clean. Rejecting because the "
                "patch rewrites a kernel with no before/after benchmark to stand on."
            ),
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.parametrize(
    "reasoning",
    [
        pytest.param(
            f"- {QUANTITATIVE_CLAIM_REASON_CODE}: clean.\n- rollback: absent. Rejecting on the rollback.",
            id="enumerated_checklist",
        ),
        pytest.param(
            f"> {QUANTITATIVE_CLAIM_REASON_CODE}: proposal_set[*] must not carry a self-reported gain field.\n"
            "The proposal is clean on that rule; it is refused for lack of a rollback.",
            id="quoted_rule_text",
        ),
        pytest.param(
            f"```\n{QUANTITATIVE_CLAIM_REASON_CODE}: proposal_set[*] must not carry a self-reported gain field.\n"
            "```\nRefused for lack of a rollback.",
            id="fenced_rule_text",
        ),
        pytest.param(
            f"Refused for lack of a rollback. For reference:\n{QUANTITATIVE_CLAIM_REASON_CODE}: not at issue here.",
            id="cited_after_the_grounds",
        ),
    ],
)
def test_a_rule_a_verdict_enumerates_or_quotes_is_not_the_ground_it_rests_on(reasoning):
    """A model that walks the rule list, or quotes a rule to say it does not
    apply, writes the code in exactly the shapes a citation would take. Reading
    one of those as grounds dispatches a proposal the Critic refused, so the
    scan only fires on a citation opening the verdict's own prose."""
    entry = {"verdict": "reject", "reasoning": reasoning}

    assert verdict_held_to_its_rule(entry, action_name="specialist") == ("reject", "")


def test_a_citation_opening_the_verdict_still_holds_it():
    """The shape the field verdict used stays readable: the code opens the
    prose and a colon introduces the finding."""
    entry = {
        "verdict": "reject",
        "reasoning": f"`{QUANTITATIVE_CLAIM_REASON_CODE}`: the payload carries predicted_gain_pct.",
    }

    assert verdict_held_to_its_rule(entry, action_name="specialist") == (
        "advise",
        QUANTITATIVE_CLAIM_REASON_CODE,
    )


def test_a_citation_in_either_field_the_entry_states_its_grounds_in_is_read():
    """The two prose keys are the same speaker's grounds for the same verdict --
    ``reasoning`` is how a single verdict spells them and ``rationale`` how a
    variant does -- and nothing establishes a priority between them. Stopping at
    the first key that says anything let a one-line pointer decide whether the
    citation beside it was read at all."""
    entry = {
        "verdict": "reject",
        "reasoning": "See the per-variant notes.",
        "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: the variant carries a self-reported gain.",
    }

    assert verdict_held_to_its_rule(entry, action_name="specialist") == (
        "advise",
        QUANTITATIVE_CLAIM_REASON_CODE,
    )


def test_a_rule_named_in_a_remediation_note_is_not_a_citation():
    """``notes`` is where a model writes what to do next, including "this is not
    a <code> problem"; grounds are stated in ``reasoning``."""
    entry = {
        "verdict": "reject",
        "reasoning": "the benchmark is not comparable with the baseline.",
        "notes": [f"{QUANTITATIVE_CLAIM_REASON_CODE}: nothing to fix on that front."],
    }
    assert verdict_held_to_its_rule(entry, action_name="specialist") == ("reject", "")


@pytest.mark.parametrize(
    "reasoning",
    [
        pytest.param(
            f"{QUANTITATIVE_CLAIM_REASON_CODE}_v2: a successor rule, not this one.",
            id="a_longer_identifier_the_code_only_starts",
        ),
        pytest.param(
            f"{QUANTITATIVE_CLAIM_REASON_CODE}\u00a0: a non-breaking space, not the gap a citation leaves.",
            id="a_space_that_is_not_the_gap_a_citation_leaves",
        ),
    ],
)
def test_a_code_the_colon_does_not_follow_is_not_a_citation(reasoning):
    """Nothing may sit between the code and the colon but a backtick and an
    ASCII gap, so an identifier the code merely starts is not a citation. The
    gap stays ASCII on purpose: a line has already been through ``splitlines``,
    which leaves only exotic unicode spaces for a wider class to add, and
    stretching the scan to reach them would buy a guess at the cost of reading
    citations that were never made.

    The other end of the anchor -- that nothing but whitespace or a backtick may
    precede the code -- is what
    ``test_a_rule_a_verdict_enumerates_or_quotes_is_not_the_ground_it_rests_on``
    covers, through the list and quote markers a model actually writes."""
    entry = {"verdict": "reject", "reasoning": reasoning}

    assert verdict_held_to_its_rule(entry, action_name="specialist") == ("reject", "")


@pytest.mark.parametrize(
    "risks",
    [
        pytest.param("the patch does not apply and there is no rollback plan", id="one_sentence_not_a_list"),
        pytest.param({"severity": "blocker", "summary": "the patch does not apply"}, id="one_risk_not_in_a_list"),
    ],
)
def test_findings_stated_outside_the_shape_that_counts_them_are_not_one_ground(risks):
    """The hold is confined to a reject naming at most one risk, which means
    counting the ``risks`` list. A verdict that states its risks in some other
    shape has stated grounds the count cannot be read off, and reading them as
    one would hold the whole reject to whichever rule the prose cites."""
    entry = {
        "verdict": "reject",
        "reasoning": f"{QUANTITATIVE_CLAIM_REASON_CODE}: the payload carries predicted_gain_pct.",
        "risks": risks,
    }

    assert verdict_held_to_its_rule(entry, action_name="specialist") == ("reject", "")


@pytest.mark.parametrize(
    "findings",
    [
        pytest.param(
            {"risks": [{"severity": "blocker", "summary": "the payload carries a self-reported gain."}, ""]},
            id="a_ground_stated_beside_an_empty_slot_is_one_ground",
        ),
        pytest.param({"required_evidence": [""]}, id="an_empty_slot_is_not_evidence_the_verdict_still_wants"),
    ],
)
def test_an_empty_findings_slot_states_nothing(findings):
    """A verdict serialised with its list slots padded out has stated what is
    in them, not how many there are -- the same reading
    ``serialize_verdict_advisory`` takes of the field set downstream. Counting
    an empty slot as a ground would withhold the downgrade from the shape the
    hold was built for."""
    entry = {
        "verdict": "reject",
        "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
        **findings,
    }

    assert verdict_held_to_its_rule(entry, action_name="specialist") == (
        "advise",
        QUANTITATIVE_CLAIM_REASON_CODE,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "findings",
    [
        pytest.param({"risks": 1}, id="a_count_where_the_risks_go"),
        pytest.param({"required_evidence": 1}, id="a_count_where_the_evidence_requests_go"),
    ],
)
async def test_a_verdict_whose_findings_cannot_be_counted_still_decides_its_proposal(coord, findings):
    """A number where the schema puts a list used to raise ``TypeError`` out of
    the ground count. That reached the router's catch-all, which recorded the
    exception and dropped the intent, leaving the proposal undecided for the
    rest of the session. A count is not a citation the hold can act on, so the
    verdict the Critic wrote is what the proposal is decided on."""
    pending = PendingProposal(
        proposal_msg_id="msg-uncountable",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-uncountable"}},
    )
    coord.state.pending_proposals["msg-uncountable"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-uncountable",
            "verdict": "reject",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
            **findings,
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert (pending.decided, pending.verdict) == (True, "reject")
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_a_held_reject_never_lands_the_patch_it_rejected(coord):
    """The rules the hold enforces are about specialist proposal payloads, and
    ``advise`` means "dispatch may proceed": holding an ``integrate_patch``
    reject would execute the patch the Critic refused, with the propose-time
    PolicyGate patch gate already behind it."""
    coord.shared_state = _PatchVerdictSharedState()
    pending = PendingProposal(
        proposal_msg_id="msg-patch",
        from_agent="orchestration",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={"action_name": "integrate_patch", "params": {"specialist_task_id": "t-patch"}},
    )
    coord.state.pending_proposals["msg-patch"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-patch",
            "verdict": "reject",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "reject"
    assert coord._materialise_calls == []
    assert coord.shared_state.patch_verdicts["t-patch"] == "reject"


@pytest.mark.asyncio
@pytest.mark.parametrize("action_name", ["integrate_patch", "kernel_opt", "sweep"])
async def test_a_reject_of_a_proposal_the_rules_do_not_govern_stands(coord, action_name):
    """Every advisory rule is about a specialist-authored proposal payload, so a
    reject of anything else cannot rest on one however the verdict is worded."""
    pending = PendingProposal(
        proposal_msg_id="msg-ungoverned",
        from_agent="orchestration",
        action_name=action_name,
        predicted_gain_pct=0.0,
        payload={"action_name": action_name, "params": {}},
    )
    coord.state.pending_proposals["msg-ungoverned"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-ungoverned",
            "verdict": "reject",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_a_cross_domain_hint_reject_is_held_too(coord):
    """Every rule declaring ``advise`` is covered, not just the quantitative-claim one."""
    reason_code = cross_domain_rule_descriptors()[0]["failure_reason_code"]
    pending = PendingProposal(
        proposal_msg_id="msg-xd",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-2"}},
    )
    coord.state.pending_proposals["msg-xd"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-xd",
            "verdict": "reject",
            "failure_reason_code": reason_code,
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "advise"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload_extra",
    [
        pytest.param({}, id="no_reason_code"),
        # A code no rule declares as advisory — e.g. the safety hard guard,
        # which critic.md keeps at ``reject``.
        pytest.param({"failure_reason_code": "specialist_patch_not_grounded"}, id="code_outside_the_advisory_set"),
    ],
)
async def test_a_substantive_reject_still_rejects(coord, payload_extra):
    """The backstop is scoped to rules that declared ``advise``; every other reject stands."""
    pending = PendingProposal(
        proposal_msg_id="msg-real",
        from_agent="orchestration",
        action_name="specialist",
        predicted_gain_pct=0.0,
        payload={"action_name": "specialist", "params": {"task_id": "t-3"}},
    )
    coord.state.pending_proposals["msg-real"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-real",
            "verdict": "reject",
            **payload_extra,
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_one_variant_held_to_advise_does_not_out_rank_its_siblings(coord):
    """The hold runs per variant before the collapse, so an advisory-only reject cannot discard the set."""
    pending = _seed_explore_proposal(coord, msg_id="msg-grid", variants=["v_a", "v_b"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-grid",
            "verdict_map": {
                "v_a": {"verdict": "advise", "rationale": "worth a look"},
                "v_b": {
                    "verdict": "reject",
                    "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
                },
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)
    # Without the per-entry hold, reject out-ranks advise and the set is lost.
    assert pending.verdict == "advise"
    assert len(coord._materialise_calls) == 1
    assert coord._materialise_calls[0][1] == {"v_a", "v_b"}


@pytest.mark.asyncio
async def test_a_variant_citing_its_rule_in_the_key_the_entry_carries_is_held(coord):
    """A ``verdict_map`` entry is ``{verdict, rationale?}`` — the shape PolicyGate
    documents and every fixture uses. Scanning ``reasoning`` there left the
    per-variant hold unreachable on the path it was written for."""
    pending = _seed_explore_proposal(coord, msg_id="msg-rationale", variants=["v_a", "v_b"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-rationale",
            "verdict_map": {
                "v_a": {"verdict": "needs_review", "rationale": "no prior on this flag"},
                "v_b": {
                    "verdict": "reject",
                    "rationale": (f"{QUANTITATIVE_CLAIM_REASON_CODE}: the variant carries a self-reported gain."),
                },
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "advise"
    assert len(coord._materialise_calls) == 1
    assert coord._materialise_calls[0][1] == {"v_b"}


@pytest.mark.asyncio
async def test_a_grid_rejected_only_on_advisory_rules_survives(coord):
    """A whole map rejected on advisory-only grounds collapses to advise, not reject."""
    pending = _seed_explore_proposal(coord, msg_id="msg-grid-all", variants=["v_a", "v_b"])
    entry = {"verdict": "reject", "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE}
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-grid-all",
            "verdict_map": {"v_a": dict(entry), "v_b": dict(entry)},
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.verdict == "advise"
    assert len(coord._materialise_calls) == 1
    assert coord._materialise_calls[0][1] == {"v_a", "v_b"}


@pytest.mark.asyncio
async def test_a_variant_reject_keeps_the_grounds_the_payload_states_for_it(coord):
    """A ``verdict_map`` entry is ``{verdict, rationale?, failure_reason_code?}``
    -- it has no slot for ``required_evidence`` or ``risks``, which the schema
    puts on the payload. Reading the hold's "rests on one ground" guard against
    the entry alone made it structurally unfireable: a reject stating blockers
    and asking for evidence was downgraded and its variant dispatched."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-grounds", variants=["v_a"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-grounds",
            "required_evidence": ["matched benchmark", "rollback plan"],
            "risks": [
                {"severity": "blocker", "summary": "the patch does not apply to the base checkout."},
                {"severity": "blocker", "summary": "there is no rollback plan."},
                {"severity": "minor", "summary": "the variant carries a self-reported gain."},
            ],
            "verdict_map": {
                "v_a": {
                    "verdict": "reject",
                    "rationale": (
                        f"{QUANTITATIVE_CLAIM_REASON_CODE}: the variant carries a self-reported "
                        "gain, and the patch it rests on does not apply."
                    ),
                },
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_a_reject_code_the_payload_declares_outranks_a_variants_advisory_prose(coord):
    """Declared-code precedence has to reach the batch path too: a
    ``failure_reason_code`` naming a rule that asked for a reject is the
    Critic's own citation, and a variant naming an advisory rule in prose
    cannot soften it."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-declared", variants=["v_a"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-declared",
            "failure_reason_code": "specialist_patch_not_grounded",
            "verdict_map": {
                "v_a": {
                    "verdict": "reject",
                    "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: the variant carries a self-reported gain.",
                },
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_the_one_risk_a_batch_states_is_not_the_rule_its_variants_cite(coord):
    """A single stated risk is allowed beside a citation because on the single
    path both come from one statement by one author. A batch states its risks
    for the whole set and the citation belongs to the entry, so counting them
    as one ground identifies them -- and a set refused for supplying no
    rollback plan anywhere was dispatched on a formatting rule instead."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-one-blocker", variants=["v_a", "v_b"])
    entry = {
        "verdict": "reject",
        "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: this variant carries a self-reported gain field.",
    }
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-one-blocker",
            "risks": [{"severity": "blocker", "summary": "no variant in this set supplies a rollback plan."}],
            "verdict_map": {"v_a": dict(entry), "v_b": dict(entry)},
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert (pending.verdict, len(coord._materialise_calls)) == ("reject", 0)


@pytest.mark.asyncio
async def test_a_variant_resting_only_on_the_cited_rule_still_gives_up_its_reject(coord):
    """A finding is what holds a variant, not the batch's whole advisory
    context: remediation notes and evidence pointers say what to do next and
    where to look, so a variant resting on nothing but its cited rule still
    gives up its reject."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-only", variants=["v_a", "v_b"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-only",
            "notes": ["Resubmit without predicted_gain_pct."],
            "packet_evidence": ["proposal_set[0].predicted_gain_pct"],
            "verdict_map": {
                "v_a": {"verdict": "needs_review", "rationale": "no prior on this flag"},
                "v_b": {
                    "verdict": "reject",
                    "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: the variant carries a self-reported gain.",
                },
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "advise"
    assert len(coord._materialise_calls) == 1
    kinds = [call.args[2].get("kind") for call in coord._record_observation.await_args_list]
    assert "verdict_downgraded_to_rule_verdict" in kinds


@pytest.mark.asyncio
async def test_the_verdicts_own_prose_does_not_supply_a_variants_citation(coord):
    """Grounds stated once for the whole review are inherited; a citation is
    not. The payload's prose speaks for the batch, and reading it as one
    variant's grounds would downgrade a reject whose own rationale refuses on
    something else entirely."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-prose", variants=["v_a"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-prose",
            "reasoning": f"{QUANTITATIVE_CLAIM_REASON_CODE}: two of the variants carry a self-reported gain.",
            "verdict_map": {
                "v_a": {
                    "verdict": "reject",
                    "rationale": "this variant has no rollback plan and its patch does not apply.",
                },
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_the_batchs_declared_advisory_code_does_not_supply_a_variants_citation(coord):
    """The batch's citation is not the variant's either. A declared code
    outranks prose, so inheriting one would not merely compete with the
    variant's own rationale — it would stop it being read at all, and downgrade
    a reject that refuses on grounds no advisory rule ever asked advice for."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-field", variants=["v_a"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-field",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
            "verdict_map": {
                "v_a": {
                    "verdict": "reject",
                    "rationale": "this variant has no rollback plan and its patch does not apply.",
                },
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert pending.verdict == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rejected", "risks"),
    [
        pytest.param(
            "v_a",
            [
                {"severity": "blocker", "summary": "no variant in this set gives a rollback plan (v_b included)."},
                {"severity": "blocker", "summary": "none name the active path the flag touches; v_b is typical."},
            ],
            id="prose_naming_a_sibling_as_its_example",
        ),
        pytest.param(
            "v_a",
            [
                {"severity": "blocker", "summary": "v_a: no rollback plan."},
                {"severity": "blocker", "summary": "v_a does not name the active path the flag touches."},
            ],
            id="prose_naming_the_rejected_variant",
        ),
        pytest.param(
            "v_a",
            [
                {"severity": "blocker", "summary": "the second variant gives no rollback plan; nor do the rest."},
                {"severity": "blocker", "summary": "none name the active path the flag touches."},
            ],
            id="prose_naming_no_key_at_all",
        ),
        pytest.param(
            "naïve",
            [
                {"severity": "blocker", "summary": "naïve and v_b: neither gives a rollback plan."},
                {"severity": "blocker", "summary": "neither names the active path the flag touches."},
            ],
            id="prose_naming_a_key_no_json_escape_survives",
        ),
    ],
)
async def test_a_batch_blocker_binds_every_variant_whatever_name_its_prose_carries(coord, rejected, risks):
    """A batch states its blockers once, in prose written for the whole set,
    and nothing asks the Critic to file one under a variant's key. Reading a
    name in that prose as "about that one, not you" removed grounds the Critic
    stated and dispatched a set it refused; reading it the other way would let
    the hold turn on a spelling. Neither is read: the batch's grounds hold
    every variant that states none of its own."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-example", variants=[rejected, "v_b", "v_c"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-example",
            "risks": risks,
            "verdict_map": {
                rejected: {
                    "verdict": "reject",
                    "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: carries a self-reported gain field.",
                },
                "v_b": {"verdict": "needs_review", "rationale": "re-run once the patch applies"},
                "v_c": {"verdict": "needs_review", "rationale": "no prior on this flag"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert (pending.verdict, len(coord._materialise_calls)) == ("reject", 0)


@pytest.mark.asyncio
async def test_a_batch_finding_is_not_disowned_by_a_sibling_named_in_a_neighbouring_field(coord):
    """A finding is a dict of several fields, and only one of them states what
    the finding *is*. A remediation pointing at how a sibling was fixed, or an
    evidence request phrased by example, says nothing about who the finding
    binds."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-fields", variants=["v_a", "v_b"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-fields",
            "required_evidence": ["a rollback plan of the kind v_b supplies"],
            "risks": [
                {
                    "severity": "blocker",
                    "summary": "the patch does not apply to the base checkout.",
                    "required_fix": "rebase it the way v_b was rebased.",
                },
            ],
            "verdict_map": {
                "v_a": {
                    "verdict": "reject",
                    "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: carries a self-reported gain field.",
                },
                "v_b": {"verdict": "needs_review", "rationale": "re-run once the patch applies"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert (pending.verdict, len(coord._materialise_calls)) == ("reject", 0)


@pytest.mark.asyncio
async def test_grounds_stated_in_a_shape_the_schema_does_not_use_are_not_read_as_one(coord):
    """``risks`` is a list in the schema. A verdict that states it as one
    sentence has stated grounds whose count cannot be read off, and "the patch
    does not apply and there is no rollback plan" is two of them -- so it
    cannot be counted as the single ground the hold is confined to."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-scalar", variants=["v_a", "v_b"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-scalar",
            "risks": "the patch does not apply and there is no rollback plan",
            "verdict_map": {
                "v_a": {
                    "verdict": "reject",
                    "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: carries a self-reported gain field.",
                },
                "v_b": {"verdict": "needs_review", "rationale": "no prior on this flag"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert (pending.verdict, len(coord._materialise_calls)) == ("reject", 0)


@pytest.mark.asyncio
async def test_evidence_the_batch_still_wants_holds_a_variant_resting_on_a_cited_rule(coord):
    """``required_evidence`` is the stronger of the two inherited findings --
    one item is a ground where ``risks`` needs two -- and a batch states it in
    the same place, for variants whose own shape has no slot for it."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-evidence", variants=["v_a", "v_b"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-evidence",
            "required_evidence": ["a matched benchmark for every variant here"],
            "verdict_map": {
                "v_a": {
                    "verdict": "reject",
                    "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: carries a self-reported gain field.",
                },
                "v_b": {"verdict": "needs_review", "rationale": "no prior on this flag"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert (pending.verdict, len(coord._materialise_calls)) == ("reject", 0)


@pytest.mark.asyncio
async def test_a_variant_that_states_a_finding_of_its_own_still_answers_for_the_batchs(coord):
    """Filing a risk under a variant's key says that risk is about it; it does
    not say the blocker the batch filed under no key is about someone else.
    Letting the entry's own list stand in for the batch's would hand a variant
    its downgrade back by writing one minor risk next to it."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-both", variants=["v_a", "v_b"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-both",
            "risks": [{"severity": "blocker", "summary": "nothing here has a rollback plan."}],
            "verdict_map": {
                "v_a": {
                    "verdict": "reject",
                    "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: carries a self-reported gain field.",
                    "risks": [{"severity": "minor", "summary": "carries a self-reported gain field."}],
                },
                "v_b": {"verdict": "needs_review", "rationale": "no prior on this flag"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert (pending.verdict, len(coord._materialise_calls)) == ("reject", 0)


@pytest.mark.asyncio
async def test_a_variant_stating_no_grounds_does_not_borrow_the_batchs_advisory_citation(coord):
    """An entry that states nothing but its verdict has cited no rule. The
    batch's advisory code is the batch's citation, and lending it to that
    entry would downgrade a reject on grounds nobody wrote down."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-bare", variants=["v_a", "v_b"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-bare",
            "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
            "verdict_map": {
                "v_a": {"verdict": "reject"},
                "v_b": {"verdict": "needs_review", "rationale": "no prior on this flag"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert (pending.verdict, len(coord._materialise_calls)) == ("reject", 0)


@pytest.mark.asyncio
async def test_a_batch_code_no_rule_declares_withholds_the_downgrade(coord):
    """A code the Critic filled in with something no rule defines is not a
    citation the hold can act on, and it is not permission to dispatch either:
    it withholds the downgrade, at the cost this reading accepts -- the set
    keeps the reject it would otherwise have given up."""
    pending = _seed_explore_proposal(coord, msg_id="msg-map-junk", variants=["v_a", "v_b"])
    entry = {
        "verdict": "reject",
        "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: carries a self-reported gain field.",
    }
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-map-junk",
            "failure_reason_code": "N/A",
            "verdict_map": {"v_a": dict(entry), "v_b": dict(entry)},
        },
    )
    await coord._handle_review_verdict("critic", intent)

    assert (pending.verdict, len(coord._materialise_calls)) == ("reject", 0)


@pytest.mark.parametrize(
    ("entry", "payload", "expected"),
    [
        pytest.param(
            {"verdict": "reject"},
            {"risks": [{"severity": "blocker"}], "required_evidence": ["a bench"]},
            {"verdict": "reject"},
            id="the_batchs_findings_are_not_moved_onto_the_entry",
        ),
        pytest.param(
            {"verdict": "reject", "failure_reason_code": "variant_code"},
            {"failure_reason_code": "payload_code"},
            {"verdict": "reject", "failure_reason_code": "variant_code"},
            id="the_entrys_own_statement_of_a_ground_wins",
        ),
        pytest.param(
            {"verdict": "reject", "rationale": "the variant's own grounds"},
            {"reasoning": "the batch's grounds"},
            {"verdict": "reject", "rationale": "the variant's own grounds"},
            id="prose_is_not_inherited",
        ),
        pytest.param(
            {"verdict": "reject", "rationale": "the variant's own grounds"},
            {"failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE},
            {"verdict": "reject", "rationale": "the variant's own grounds"},
            id="a_code_that_could_soften_the_reject_is_not_inherited_either",
        ),
        pytest.param(
            {"verdict": "reject", "rationale": "the variant's own grounds"},
            {"failure_reason_code": "specialist_patch_not_grounded"},
            {
                "verdict": "reject",
                "rationale": "the variant's own grounds",
                "failure_reason_code": "specialist_patch_not_grounded",
            },
            id="a_code_that_can_only_hold_the_reject_is_inherited",
        ),
        pytest.param(
            {"verdict": "reject", "risks": [{"summary": "its own"}]},
            {"risks": [{"summary": "the batch's"}]},
            {"verdict": "reject", "risks": [{"summary": "its own"}]},
            id="an_entrys_own_findings_are_left_as_the_entry_stated_them",
        ),
        pytest.param("reject", {"failure_reason_code": "payload_code"}, {}, id="a_non_dict_entry_states_nothing"),
    ],
)
def test_an_entry_inherits_only_a_code_that_can_hold_its_reject(entry, payload, expected):
    """A finding the batch states is read where it is stated, as a hold on the
    whole set; moving one onto an entry would put it back in the count the
    ``<= 1`` allowance runs on, which is what identified the batch's ground
    with the entry's rule."""
    assert verdict_map_entry_grounds(entry, payload) == expected


def test_a_batch_that_states_nothing_readable_leaves_its_entry_on_its_own_grounds():
    """Both halves of the batch reading -- what an entry inherits and what
    holds it -- ask a payload what it states, so both have to answer for one
    that is not a payload at all. It states no findings and no code, which
    leaves the entry judged on what it wrote itself."""
    entry = {
        "verdict": "reject",
        "rationale": f"{QUANTITATIVE_CLAIM_REASON_CODE}: carries a self-reported gain field.",
    }

    assert verdict_map_entry_held_to_its_rule(entry, "not a payload", action_name="specialist") == (
        "advise",
        QUANTITATIVE_CLAIM_REASON_CODE,
    )


def test_a_map_of_verdicts_none_of_which_decides_asks_for_review():
    """``redirect`` is a legal verdict with no place in the collapse order, so a
    map carrying only those decides nothing and must fall back to review."""
    assert collapse_verdicts(["redirect", ""]) == "needs_review"


def test_an_empty_map_asks_for_review():
    assert collapse_verdicts([]) == "needs_review"


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        pytest.param(["approve", "reject", "advise", "needs_review"], "approve", id="one_approve_carries_it"),
        pytest.param(["reject", "advise", "needs_review"], "reject", id="reject_outranks_advice"),
        pytest.param(["advise", "needs_review"], "advise", id="advice_outranks_more_review"),
    ],
)
def test_the_collapse_keeps_its_priority_order(verdicts, expected):
    assert collapse_verdicts(verdicts) == expected


def test_proceedable_names_drop_rejects_and_blank_keys():
    assert proceedable_variant_names(
        {"v_a": "approve", "v_b": "reject", "": "advise", "v_c": "advise", "v_d": "needs_review"}
    ) == {"v_a", "v_c"}


def test_collapse_verdict_map_keeps_proceedable_siblings():
    summary, names = collapse_verdict_map({"v_a": "advise", "v_b": "reject", "v_c": "approve"})
    assert summary == "approve"
    assert names == {"v_a", "v_c"}


def test_collapse_verdict_map_advise_survives_a_sibling_reject():
    summary, names = collapse_verdict_map({"v_a": "advise", "v_b": "reject"})
    assert summary == "advise"
    assert names == {"v_a"}


def test_collapse_verdict_map_with_no_proceedable_variant_stays_reject():
    summary, names = collapse_verdict_map({"v_a": "reject", "v_b": "needs_review"})
    assert summary == "reject"
    assert names is None


# 4. _materialize_approved_proposal — filter semantics (unit)
@pytest.mark.asyncio
async def test_materialize_filter_drops_rejected_variants(tmp_path: Path):
    """Pin the ``approved_variant_names`` filter contract independently."""
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = _BareSharedState()
    coord.state = CoordinatorState()
    coord.recipe_kb = _StubRecipeKB()
    coord.bus = _StubBus()
    coord._record_observation = AsyncMock()  # type: ignore[method-assign]

    create_calls: list[dict[str, Any]] = []

    class _StubTaskRegistry:
        async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
            create_calls.append(dict(kwargs))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-explore-filtered",
                    kind=kwargs["kind"],
                    state="queued",
                    params=kwargs["params"],
                    idempotency_key=kwargs["idempotency_key"],
                ),
                False,
            )

    coord.tasks = _StubTaskRegistry()
    pending = _seed_explore_proposal(
        coord,
        variants=["v_a", "v_b", "v_c", "v_d"],
    )

    class _MoreState(_BareSharedState):
        baseline_config_path: str = ""
        baseline_tput: float = 1000.0
        synergy_attempted: list[str] = field(default_factory=list)
        backends_search: dict = field(default_factory=dict)
        params_search: dict = field(default_factory=dict)
        current_best: dict = field(default_factory=dict)

    coord.shared_state = _MoreState()
    await coord._materialize_approved_proposal(
        pending,
        approved_variant_names={"v_a", "v_c"},
    )
    assert create_calls, "materialize must enqueue a task"
    grid = create_calls[0]["params"]["grid"]
    names = [v["name"] for v in grid]
    assert names == ["v_a", "v_c"]
    assert create_calls[0]["params"]["critic_filtered_count"] == 2


@pytest.mark.asyncio
async def test_materialize_filter_skips_when_no_variant_survives(tmp_path: Path):
    """A filter that matches nothing must not enqueue an empty explore grid."""
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.state = CoordinatorState()
    coord.recipe_kb = _StubRecipeKB()
    coord.bus = _StubBus()
    coord._record_observation = AsyncMock()  # type: ignore[method-assign]
    create_calls: list[dict[str, Any]] = []

    class _StubTaskRegistry:
        async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
            create_calls.append(dict(kwargs))
            raise AssertionError("empty filtered grid must not create a task")

    coord.tasks = _StubTaskRegistry()
    pending = _seed_explore_proposal(coord, variants=["v_a", "v_b"])

    class _MoreState(_BareSharedState):
        baseline_config_path: str = ""
        baseline_tput: float = 1000.0
        synergy_attempted: list[str] = field(default_factory=list)
        backends_search: dict = field(default_factory=dict)
        params_search: dict = field(default_factory=dict)
        current_best: dict = field(default_factory=dict)

    coord.shared_state = _MoreState()
    await coord._materialize_approved_proposal(pending, approved_variant_names={"no-such-variant"})
    assert create_calls == []
    kinds = [call.args[2].get("kind") for call in coord._record_observation.await_args_list]
    assert "proposal_materialize_skipped" in kinds


@pytest.mark.asyncio
async def test_materialize_without_filter_keeps_full_grid(tmp_path: Path):
    """``approved_variant_names=None`` leaves the grid untouched."""
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.state = CoordinatorState()
    coord.recipe_kb = _StubRecipeKB()
    coord.bus = _StubBus()
    coord._record_observation = AsyncMock()  # type: ignore[method-assign]
    create_calls: list[dict[str, Any]] = []

    class _StubTaskRegistry:
        async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
            create_calls.append(dict(kwargs))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-explore-full",
                    kind=kwargs["kind"],
                    state="queued",
                    params=kwargs["params"],
                    idempotency_key=kwargs["idempotency_key"],
                ),
                False,
            )

    coord.tasks = _StubTaskRegistry()
    pending = _seed_explore_proposal(
        coord,
        variants=["v_a", "v_b", "v_c"],
    )

    @dataclass
    class _MoreState:
        baseline_config_path: str = ""
        baseline_tput: float = 1000.0
        recipe_kb_session_id: str = "sid-test"
        save_count: int = 0
        synergy_attempted: list[str] = field(default_factory=list)
        backends_search: dict = field(default_factory=dict)
        params_search: dict = field(default_factory=dict)
        current_best: dict = field(default_factory=dict)
        auto_roofline_pending_task_id: str = ""

        def save(self, _session_dir):
            self.save_count += 1

    coord.shared_state = _MoreState()
    await coord._materialize_approved_proposal(pending)
    grid = create_calls[0]["params"]["grid"]
    names = [v["name"] for v in grid]
    assert names == ["v_a", "v_b", "v_c"]
    assert "critic_filtered_count" not in create_calls[0]["params"]


# 5. _handle_delegate — explore grid runs directly (no Critic pre-review)
def _delegate_coord(tmp_path: Path):
    """Coordinator double reaching the direct explore-task creation path."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path

    class _State(_BareSharedState):
        baseline_config_path: str = ""
        tick: int = 0

        def is_pruned(self, _action_name: str) -> bool:
            return False

        def reset_policy_denial_streak(self, _action_name: str) -> None:
            return None

    c.shared_state = _State()
    c.state = CoordinatorState()
    c.recipe_kb = _StubRecipeKB()
    c.bus = _StubBus()
    c._record_observation = AsyncMock()  # type: ignore[method-assign]
    c._record_policy_denied = AsyncMock()  # type: ignore[method-assign]
    c._admission_denial_for_action = lambda *a, **k: None  # type: ignore[method-assign]
    c._registry_lanes_ttl = lambda _name: (set(), 0)  # type: ignore[method-assign]
    c.policy = None
    return c


@pytest.mark.asyncio
async def test_delegate_explore_with_grid_creates_task_directly(tmp_path: Path):
    """A delegate explore with a non-empty grid creates an explore task directly, grid forwarded verbatim."""
    coord = _delegate_coord(tmp_path)
    create_calls: list[dict[str, Any]] = []

    class _TaskRegistry:
        async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
            create_calls.append(dict(kwargs))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-explore-direct",
                    kind=kwargs["kind"],
                    state="queued",
                    params=kwargs["params"],
                    idempotency_key=kwargs["idempotency_key"],
                ),
                False,
            )

    coord.tasks = _TaskRegistry()
    grid = [
        {"name": "v_a", "extra_args": "--flag-a", "provenance": "specialist:serving_specialist"},
        {"name": "v_b", "extra_args": "--flag-b", "provenance": "specialist:serving_specialist"},
    ]
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "explore",
            "params": {"grid": grid},
            "idempotency_key": "explore-round-1",
        },
    )
    await coord._handle_delegate("orchestration", intent)
    assert coord.state.pending_proposals == {}
    assert len(create_calls) == 1
    assert create_calls[0]["kind"] == "explore"
    assert create_calls[0]["params"]["grid"] == grid


@pytest.mark.asyncio
async def test_delegate_explore_seeds_the_stack_with_the_anchor(tmp_path: Path):
    """A delegate explore carries current_best's args/envs, not just its throughput.

    Seeding the anchor alone launched every variant on a bare config and graded it
    against the established recipe, so a whole round read as a ~40% regression.
    """
    coord = _delegate_coord(tmp_path)
    coord.shared_state.baseline_tput = 4663.79
    coord.shared_state.current_best = {
        "tput": 7725.6,
        "extra_server_args": "--kv-cache-dtype fp8_e4m3 --max-num-seqs 64",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
    }
    created: list[dict[str, Any]] = []

    class _TaskRegistry:
        async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
            created.append(dict(kwargs["params"]))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-explore-stack",
                    kind=kwargs["kind"],
                    state="queued",
                    params=kwargs["params"],
                    idempotency_key=kwargs["idempotency_key"],
                ),
                False,
            )

    coord.tasks = _TaskRegistry()
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "explore",
            "params": {"grid": [{"name": "v_a", "extra_args": "--flag-a"}]},
            "idempotency_key": "explore-round-4",
        },
    )
    await coord._handle_delegate("orchestration", intent)
    assert len(created) == 1
    params = created[0]
    assert params["base_tput"] == 7725.6
    assert params["base_extra_args"] == "--kv-cache-dtype fp8_e4m3 --max-num-seqs 64"
    assert params["base_extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}


@pytest.mark.asyncio
async def test_delegate_sweep_seeds_the_stack_too(tmp_path: Path):
    """A delegated sweep scans on top of current_best, as its action contract states.

    ``executors/sweep.py`` assembles each (CONC, ISL, OSL) point from the
    ``base_*`` params, so seeding only ``explore`` left a delegated sweep
    measuring the bare baseline config.
    """
    coord = _delegate_coord(tmp_path)
    coord.shared_state.baseline_tput = 4663.79
    coord.shared_state.current_best = {
        "tput": 7725.6,
        "extra_server_args": "--max-num-seqs 64",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
    }
    created: list[dict[str, Any]] = []

    class _TaskRegistry:
        async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
            created.append(dict(kwargs["params"]))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-sweep-stack",
                    kind=kwargs["kind"],
                    state="queued",
                    params=kwargs["params"],
                    idempotency_key=kwargs["idempotency_key"],
                ),
                False,
            )

    coord.tasks = _TaskRegistry()
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "sweep", "params": {}, "idempotency_key": "sweep-1"},
    )
    await coord._handle_delegate("orchestration", intent)
    assert len(created) == 1
    params = created[0]
    assert params["base_extra_args"] == "--max-num-seqs 64"
    assert params["base_extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    # explore-only runtime knobs must not leak onto a sweep task.
    assert "explore_overtime_kill_ratio" not in params


# 6. Specialist prompt — proposal self-curation contract (Section 1 + 8)
def _build_specialist_prompt_text() -> str:
    from hyperloom.orchestrator.specialists.domains import get_domain
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
        SpecialistPromptInputs,
        build_specialist_prompts,
    )

    inputs = SpecialistPromptInputs(
        task_id="t-test",
        domain=get_domain("serving_specialist"),
        max_turns=12,
        gap_canonical_id="gap-x",
    )
    system_prompt, user_prompt = build_specialist_prompts(inputs)
    return system_prompt + "\n" + user_prompt


def test_specialist_prompt_renders_proposal_target_and_ceiling():
    text = _build_specialist_prompt_text()
    # Section 8 states both numbers; Section 1 repeats them so the target is
    # visible before the specialist starts working, not only at exit time.
    assert "**2 entries is the norm, 4 the hard cap.**" in text
    assert "**2 proposals is the norm, 4 the hard" in text
    # Both the padding and the keep-going pressures need a stated counterweight.
    assert "``empty=true`` is better than one" in text
    assert "stop once" in text and "not the only stop" in text
    assert "a coin-flip proposal is worse than none" in text
    assert "reviews each surviving variant" in text


# critic_robustness breakdown renderer (formerly test_critic_robustness_renderer_units.py)
class TestCriticRobustnessRenderer:
    """Exercises the four observable shapes of the collector input."""

    @staticmethod
    def _render(payload):
        from hyperloom.inference_optimizer.breakdown.reporters._renderers import (
            critic_robustness as cr_mod,
        )

        return cr_mod.render({"critic_robustness": payload})

    def test_empty_returns_skipped(self):
        from hyperloom.inference_optimizer.breakdown.reporters.base import RenderedSection

        out = self._render([])
        assert isinstance(out, RenderedSection)
        assert out.section_id == "critic_robustness"
        assert out.skipped is True
        assert any("no critic robustness" in s.lower() for s in out.key_facts)

    def test_prompt_only_v1_payload_is_skipped(self):
        out = self._render(["raw prompt"])
        assert out.skipped is True
        assert any("prompt-only" in w for w in out.warnings)

    def test_empty_payloads_v2_is_skipped(self):
        out = self._render(
            [
                {"prompt": "x", "response": None, "decision": "", "rationale": ""},
            ]
        )
        assert out.skipped is True
        assert any("non-actionable" in w for w in out.warnings)

    def test_populated_payload_renders_markdown_table(self):
        out = self._render(
            [
                {
                    "ts": "2026-05-13T01:01:01Z",
                    "action": "kernel_opt",
                    "decision": "KEEP",
                    "pass_count": 3,
                    "fail_count": 1,
                    "rationale": "Improved attention kernel reduces decode latency by 4%.",
                },
                {
                    "prompt": "raw fallback",
                },
            ]
        )
        assert out.skipped is False
        assert "decision" in out.markdown_block
        assert "kernel_opt" in out.markdown_block

    def test_excess_rows_truncated_with_banner(self):
        from hyperloom.inference_optimizer.breakdown.reporters._renderers import (
            critic_robustness as cr_mod,
        )

        rows = [
            {
                "decision": "KEEP",
                "pass_count": 1,
                "fail_count": 0,
                "ts": f"t{i}",
            }
            for i in range(cr_mod._MAX_ROWS + 5)
        ]
        out = self._render(rows)
        assert out.skipped is False
        assert "Showing first" in out.markdown_block


# per-action verdict_class metadata (formerly test_n38_action_verdict_class.py)
class TestN38ActionVerdictClass:
    """Per-action ``verdict_class`` metadata so new actions don't reintroduce prior deadlocks."""

    def test_action_metadata_has_verdict_class_field(self):
        from hyperloom.inference_optimizer.protocol.action_surfaces import (
            ActionMetadata,
        )

        fields = {f.name for f in ActionMetadata.__dataclass_fields__.values()}
        assert "verdict_class" in fields, (
            "ActionMetadata must declare verdict_class field so per-action "
            "policy can be looked up in critic review_constraints"
        )

    def test_default_classifier_covers_all_registered_actions(self):
        from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE

        all_actions = list(ACTION_CATALOGUE.values())
        assert all_actions, "expected the catalogue to hold >= 1 action"
        missing = [a.name for a in all_actions if not a.verdict_class]
        assert not missing, f"actions missing verdict_class: {missing} -- set it in ACTION_CATALOGUE"

    def test_default_classifier_matches_expected_buckets(self):
        from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE

        reg = ACTION_CATALOGUE

        def klass(name: str) -> str:
            a = reg.get(name)
            assert a is not None, f"action {name!r} not registered"
            return a.verdict_class

        assert klass("integrate") == "promotion"
        for n in ("report", "session_breakdown", "target_analysis"):
            assert klass(n) == "archival", n
        registered_exploration = (
            "baseline",
            "profile",
            "roofline",
            "explore",
            "sweep",
            "kernel_opt",
            "recover",
        )
        for n in registered_exploration:
            if reg.get(n) is None:
                continue
            assert klass(n) == "exploration", n

    def test_critic_agent_backend_accepts_action_verdict_policy(self, tmp_path):
        from hyperloom.orchestrator.roles.critic_agent import (
            CriticAgentBackend,
        )

        root = tmp_path / "critic-agent"
        (root / "runtime").mkdir(parents=True)
        (root / "runtime" / "cli.py").write_text("# stub")
        sd = tmp_path / "session"
        sd.mkdir()

        def _fake_client_factory():
            class _C:
                pass

            return _C()

        def _fake_runtime_caller_factory():
            def _caller(call):
                return None

            return _caller

        backend = CriticAgentBackend(
            critic_agent_root=root,
            session_dir=sd,
            codex_client_factory=_fake_client_factory,
            runtime_caller_factory=_fake_runtime_caller_factory,
            static_context={"model": "m", "framework": "sglang"},
            action_verdict_policy={"baseline": "exploration", "integrate": "promotion"},
        )
        assert backend.action_verdict_policy == {
            "baseline": "exploration",
            "integrate": "promotion",
        }

    def test_critic_agent_backend_injects_policy_into_judge_bundle(self, tmp_path):
        import asyncio
        import json as _json
        from hyperloom.orchestrator.roles.critic_agent import (
            CriticAgentBackend,
            RuntimeCall,
        )

        root = tmp_path / "critic-agent"
        (root / "runtime").mkdir(parents=True)
        (root / "runtime" / "cli.py").write_text("# stub")
        sd = tmp_path / "session"
        sd.mkdir()

        captured_bundle: dict = {}

        class _FakeAsyncOpenAI:
            def __init__(self):
                self.chat = _FakeChat(captured_bundle)

        class _FakeChat:
            def __init__(self, bucket):
                self.completions = _FakeCompletions(bucket)

        class _FakeCompletions:
            def __init__(self, bucket):
                self._b = bucket

            async def create(self, *, model, messages, max_completion_tokens):
                user_msg = messages[-1]["content"]
                self._b["user_prompt"] = user_msg

                class _Choice:
                    message = type(
                        "M",
                        (),
                        {
                            "content": _json.dumps(
                                {
                                    "review_verdicts": [],
                                }
                            )
                        },
                    )()
                    finish_reason = "stop"

                return type("R", (), {"choices": [_Choice()]})()

        def _fake_runtime_caller_factory():
            def _caller(call: RuntimeCall) -> None:
                if call.phase == "prepare-review":
                    bundle = {
                        "kind": "coordinator_inbox",
                        "session_id": "test",
                        "proposals": [{"msg_id": "abc", "action_name": "params"}],
                        "review_constraints": {
                            "allowed_verdicts": ["approve", "advise"],
                        },
                    }
                    call.out_path.write_text(_json.dumps(bundle), encoding="utf-8")
                else:
                    call.out_path.write_text(
                        _json.dumps(
                            {
                                "intent_envelope": {"intents": []},
                            }
                        ),
                        encoding="utf-8",
                    )

            return _caller

        backend = CriticAgentBackend(
            critic_agent_root=root,
            session_dir=sd,
            codex_client_factory=_FakeAsyncOpenAI,
            runtime_caller_factory=_fake_runtime_caller_factory,
            static_context={"model": "m", "framework": "sglang"},
            action_verdict_policy={
                "params": "exploration",
                "integrate": "promotion",
            },
        )
        asyncio.run(backend.run(prompt="hello"))

        assert "action_verdict_policy" in captured_bundle.get("user_prompt", ""), (
            "action_verdict_policy must appear in the JSON prompt sent to "
            "the LLM-critic so it can look up each proposal's class"
        )
        assert "promotion" in captured_bundle["user_prompt"]

    def test_critic_md_mentions_action_verdict_policy_lookup(self):
        from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir

        p = asset_system_prompts_dir() / "critic.md"
        text = p.read_text(encoding="utf-8")
        assert "action_verdict_policy" in text, (
            "critic.md must mention action_verdict_policy so the LLM-critic "
            "treats it as the primary per-proposal lookup; otherwise newly "
            "added actions will hit the same N33/N35/N37 chicken-and-egg "
            "deadlock"
        )
        for klass in ("archival", "exploration", "promotion"):
            assert klass in text.lower(), klass
