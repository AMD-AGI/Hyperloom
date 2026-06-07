# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 KB_design §3.5 §5 / M5 §5 step 5 / KB_gaps/Gap-11 — Critic
per-variant ``verdict_map`` tests.

KB_gaps/Gap-11 root cause: REVIEW_VERDICT was a v0.6 single-verdict
protocol — a multi-variant ``explore`` proposal could only be approved
or rejected as a whole, so partial-success rounds (3 KEEP + 2 REVERT)
collapsed into "all KEEP" or "all REVERT".

This file exercises the v0.8 upgrade across the four layers it touches:

* :mod:`intent_parser` — the envelope schema accepts either
  ``verdict`` (legacy) or ``verdict_map`` (v0.8 batch); both fields
  are mutually exclusive and the per-variant entries are structurally
  validated.
* :mod:`policy` — ``_validate_review_verdict`` enforces the same
  mutual-exclusion rule and validates every per-variant verdict string
  against ``REVIEW_VERDICTS``.
* :class:`Coordinator._handle_review_verdict` /
  :meth:`_handle_verdict_map` — routes batch verdicts to the new
  per-variant dispatcher, pins the map on
  :class:`PendingProposal.verdict_map`, mirrors it back onto the bus,
  and filters the materialised grid down to the ``approve`` subset.
  (The legacy ``_cortex_t3_critic_rejected`` KB mirror was removed
  alongside the T2/T3 hypothesize/verify protocol.)
* :func:`build_critic_prompt` — the OUTPUT PROTOCOL section advertises
  the new ``verdict_map`` shape and explains the precedence rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    CoordinatorState,
    PendingProposal,
)
from inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
    IntentValidationError,
    validate_envelope,
)
from inference_optimizer.orchestrator.policy import (
    PolicyDenied,
    PolicyGate,
    REVIEW_VERDICTS,
)
from inference_optimizer.orchestrator.system_prompts.critic_prompt_builder import (
    build_critic_prompt,
)


# ===========================================================================
# 1. intent_parser — envelope schema accepts verdict OR verdict_map
# ===========================================================================
def _envelope(**payload: Any) -> dict[str, Any]:
    return {
        "intents": [{
            "intent_type": "review_verdict",
            "payload": payload,
        }],
    }


def test_intent_parser_accepts_legacy_single_verdict():
    intents = validate_envelope(_envelope(
        target_proposal_msg_id="msg-1",
        verdict="approve",
        reasoning="ok",
    ))
    assert len(intents) == 1
    assert intents[0].type is IntentType.REVIEW_VERDICT
    assert intents[0].payload["verdict"] == "approve"


def test_intent_parser_accepts_per_variant_verdict_map():
    intents = validate_envelope(_envelope(
        target_proposal_msg_id="msg-1",
        verdict_map={
            "v_a": {"verdict": "approve", "rationale": "looks promising"},
            "v_b": {"verdict": "reject",  "rationale": "kb says no"},
        },
    ))
    assert intents[0].payload["verdict_map"]["v_a"]["verdict"] == "approve"


def test_intent_parser_rejects_both_verdict_and_verdict_map():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(_envelope(
            target_proposal_msg_id="msg-1",
            verdict="approve",
            verdict_map={"v_a": {"verdict": "reject"}},
        ))
    assert "mutually exclusive" in str(exc.value)


def test_intent_parser_rejects_neither_verdict_nor_verdict_map():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(_envelope(target_proposal_msg_id="msg-1"))
    assert "must include either" in str(exc.value)


def test_intent_parser_rejects_empty_verdict_map():
    with pytest.raises(IntentValidationError):
        validate_envelope(_envelope(
            target_proposal_msg_id="msg-1",
            verdict_map={},
        ))


def test_intent_parser_rejects_verdict_map_entry_missing_verdict():
    with pytest.raises(IntentValidationError) as exc:
        validate_envelope(_envelope(
            target_proposal_msg_id="msg-1",
            verdict_map={"v_a": {"rationale": "no verdict key"}},
        ))
    assert "missing required 'verdict'" in str(exc.value)


def test_intent_parser_rejects_non_dict_verdict_map_entry():
    with pytest.raises(IntentValidationError):
        validate_envelope(_envelope(
            target_proposal_msg_id="msg-1",
            verdict_map={"v_a": "approve"},
        ))


# ===========================================================================
# 2. PolicyGate — verdict_map content validation
# ===========================================================================
@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


def _critic_intent(**payload: Any) -> Intent:
    return Intent(type=IntentType.REVIEW_VERDICT, payload=payload)


def test_policy_gate_accepts_legacy_single_verdict(gate):
    # No exception — single-verdict path stays valid.
    gate.validate_intent("critic", _critic_intent(
        target_proposal_msg_id="msg-1", verdict="approve",
    ))


def test_policy_gate_accepts_per_variant_verdict_map(gate):
    gate.validate_intent("critic", _critic_intent(
        target_proposal_msg_id="msg-1",
        verdict_map={
            "v_a": {"verdict": "approve"},
            "v_b": {"verdict": "reject", "rationale": "no"},
        },
    ))


def test_policy_gate_rejects_when_both_present(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("critic", _critic_intent(
            target_proposal_msg_id="msg-1",
            verdict="approve",
            verdict_map={"v_a": {"verdict": "approve"}},
        ))
    assert exc.value.rule == "payload"
    # PolicyGate phrases the denial as "exactly one of ... must be
    # present"; intent_parser uses "mutually exclusive". Either
    # phrasing satisfies the contract (both gates enforce the same
    # rule, intent_parser fires first in production).
    msg = str(exc.value) + " " + (exc.value.hint or "")
    assert "exactly one" in msg or "mutually exclusive" in msg


def test_policy_gate_rejects_when_neither_present(gate):
    """Defense in depth — intent_parser should have caught this,
    but PolicyGate still rejects to keep both gates' contracts
    independent."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("critic", _critic_intent(
            target_proposal_msg_id="msg-1",
        ))
    assert exc.value.rule == "payload"


def test_policy_gate_rejects_unknown_per_variant_verdict(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("critic", _critic_intent(
            target_proposal_msg_id="msg-1",
            verdict_map={
                "v_a": {"verdict": "approve"},
                "v_b": {"verdict": "obliterate"},  # not in REVIEW_VERDICTS
            },
        ))
    assert "verdict_map" in str(exc.value)
    assert "obliterate" in str(exc.value)


def test_policy_gate_review_verdicts_vocab_contains_canonical_set():
    """Sanity for the verdict_map content gate — REVIEW_VERDICTS
    must still cover the five canonical strings the Critic prompt
    documents in §7."""
    for v in ("approve", "reject", "redirect", "advise", "needs_review"):
        assert v in REVIEW_VERDICTS


# ===========================================================================
# 3. Coordinator helpers — _handle_verdict_map dispatch
# ===========================================================================
@dataclass
class _BareSharedState:
    """SharedState double exposing the few fields the verdict_map
    handler touches (Cortex T3 path + audit hint)."""

    cortex_session_id: str = "sid-test"
    save_count: int = 0
    # ``_materialize_approved_proposal`` reads this field to gate
    # dispatch on a pending auto-roofline task; empty string means
    # "nothing in flight" and the gate is a no-op.
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
        # msg is a Message dataclass; we copy the salient bits.
        self.messages.append(_BusMessage(
            from_agent=getattr(msg, "from_agent", ""),
            to_agent=getattr(msg, "to_agent", ""),
            topic=getattr(msg, "topic", ""),
            payload=dict(getattr(msg, "payload", {}) or {}),
            in_reply_to=getattr(msg, "in_reply_to", "") or "",
            priority=int(getattr(msg, "priority", 1) or 1),
            msg_id=getattr(msg, "msg_id", ""),
        ))
        return None


class _StubCortexKB:
    enabled: bool = True

    def __init__(self) -> None:
        self.verify_calls: list[dict[str, Any]] = []

    def verify(self, **kwargs: Any) -> None:
        self.verify_calls.append(dict(kwargs))


@pytest.fixture
def coord(tmp_path: Path):
    """Coordinator-shaped object with just enough plumbing for
    ``_handle_review_verdict`` / ``_handle_verdict_map`` /
    ``_materialize_approved_proposal`` to run end-to-end."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareSharedState()
    c.state = CoordinatorState()
    c.cortex_kb = _StubCortexKB()
    c.bus = _StubBus()
    # _record_observation just pushes to the bus + handles persistence
    # in production; the stub keeps the test footprint tight.
    c._record_observation = AsyncMock()  # type: ignore[method-assign]
    # Tasks are not actually created in these unit tests — we stub
    # the registry so the materialise path can return a fake task.
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
    grid = [
        {"name": vn, "extra_args": f"--flag-{vn}"} for vn in variants
    ]
    pending = PendingProposal(
        proposal_msg_id=msg_id,
        from_agent="orchestration",
        action_name="explore",
        predicted_gain_pct=1.0,
        payload={"action_name": "explore", "params": {"grid": grid}},
    )
    coord.state.pending_proposals[msg_id] = pending
    return pending


# --- routing ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_legacy_single_verdict_still_materialises_whole_proposal(coord):
    """Non-grid actions (kernel_opt / integrate / ...) keep the v0.6
    single-verdict path: an ``approve`` materialises the whole
    proposal, NO ``approved_variant_names`` filter is passed."""
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
    # Bus mirror — single review_verdict event.
    bus_msgs = [m for m in coord.bus.messages if m.topic == "review_verdict"]
    assert len(bus_msgs) == 1
    assert bus_msgs[0].payload["verdict"] == "approve"
    assert "verdict_map" not in bus_msgs[0].payload
    # Materialise called with no variant filter.
    assert len(coord._materialise_calls) == 1
    assert coord._materialise_calls[0][1] is None
    assert pending.decided is True
    assert pending.verdict == "approve"


@pytest.mark.asyncio
async def test_verdict_map_routes_to_per_variant_handler(coord):
    pending = _seed_explore_proposal(coord)
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict_map": {
                "v_a": {"verdict": "approve"},
                "v_b": {"verdict": "approve"},
                "v_c": {"verdict": "reject", "rationale": "kb says no"},
                "v_d": {"verdict": "reject", "rationale": "duplicate"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert pending.decided is True
    assert pending.verdict == "approve"  # summary — any approved → approve
    assert set(pending.verdict_map) == {"v_a", "v_b", "v_c", "v_d"}


# --- materialise filter ----------------------------------------------------
@pytest.mark.asyncio
async def test_verdict_map_materialises_only_approved_subset(coord):
    pending = _seed_explore_proposal(coord)
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict_map": {
                "v_a": {"verdict": "approve"},
                "v_b": {"verdict": "reject"},
                "v_c": {"verdict": "approve"},
                "v_d": {"verdict": "needs_review"},
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert len(coord._materialise_calls) == 1
    _pending, approved = coord._materialise_calls[0]
    assert approved == {"v_a", "v_c"}


@pytest.mark.asyncio
async def test_verdict_map_all_rejected_does_not_materialise(coord):
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
    assert coord._materialise_calls == []
    assert pending.verdict == "reject"
    # Observation surfaced for the operator audit log.
    coord._record_observation.assert_awaited()
    kind = coord._record_observation.await_args_list[-1][0][2]["kind"]
    assert kind == "verdict_map_all_rejected"


@pytest.mark.asyncio
async def test_verdict_map_unknown_variant_is_dropped_and_logged(coord):
    pending = _seed_explore_proposal(coord, variants=["v_a", "v_b"])
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict_map": {
                "v_a":     {"verdict": "approve"},
                "v_ghost": {"verdict": "approve"},  # not in original grid
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)
    # Only v_a passes the filter.
    _pending, approved = coord._materialise_calls[0]
    assert approved == {"v_a"}
    # The bus mirror surfaces the unknown variants list.
    bus_msgs = [m for m in coord.bus.messages if m.topic == "review_verdict"]
    assert bus_msgs[0].payload["unknown_variants"] == ["v_ghost"]


# --- KB refute -------------------------------------------------------------
# The legacy ``_cortex_t3_critic_rejected`` path that fired a KB ``refuted``
# verify for every critic-rejected variant was removed alongside the
# T2/T3 hypothesize/verify protocol. Rejection is still captured in the
# verdict event + breakdown collector; the previously-tested KB-mirror
# behaviour no longer exists.


# --- bus mirror ------------------------------------------------------------
@pytest.mark.asyncio
async def test_verdict_map_mirror_carries_summary_and_partitions(coord):
    pending = _seed_explore_proposal(coord)
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict_map": {
                "v_a": {"verdict": "approve"},
                "v_b": {"verdict": "reject"},
                "v_c": {"verdict": "approve"},
                "v_d": {"verdict": "reject"},
            },
            "reasoning": "round summary",
        },
    )
    await coord._handle_review_verdict("critic", intent)
    bus_msgs = [m for m in coord.bus.messages if m.topic == "review_verdict"]
    assert len(bus_msgs) == 1
    payload = bus_msgs[0].payload
    assert payload["verdict"] == "approve"  # summary
    assert payload["approved_variants"] == ["v_a", "v_c"]
    assert payload["rejected_variants"] == ["v_b", "v_d"]
    assert payload["reasoning"] == "round summary"
    assert payload["target_proposal_msg_id"] == pending.proposal_msg_id


@pytest.mark.asyncio
async def test_verdict_map_unknown_proposal_logs_observation(coord):
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "ghost-msg",
            "verdict_map": {"v_a": {"verdict": "approve"}},
        },
    )
    await coord._handle_review_verdict("critic", intent)
    coord._record_observation.assert_awaited()
    assert coord._materialise_calls == []


# --- guard: verdict_map for non-explore proposal ---------------------------
@pytest.mark.asyncio
async def test_verdict_map_on_kernel_opt_proposal_logs_and_no_materialise(coord):
    pending = PendingProposal(
        proposal_msg_id="msg-knl",
        from_agent="orchestration",
        action_name="kernel_opt",
        predicted_gain_pct=2.0,
        payload={"action_name": "kernel_opt", "params": {}},
    )
    coord.state.pending_proposals["msg-knl"] = pending
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "msg-knl",
            "verdict_map": {"v_a": {"verdict": "approve"}},
        },
    )
    await coord._handle_review_verdict("critic", intent)
    assert coord._materialise_calls == []
    coord._record_observation.assert_awaited()
    kind = coord._record_observation.await_args_list[-1][0][2]["kind"]
    assert kind == "verdict_map_for_non_explore"


# ===========================================================================
# 4. Critic prompt — OUTPUT PROTOCOL advertises verdict_map shape
# ===========================================================================
def _critic_prompt_text() -> str:
    from inference_optimizer.orchestrator.action_registry import ActionRegistry
    registry = ActionRegistry()
    registry.load()
    return build_critic_prompt(
        action_registry=registry,
        enabled_actions=("baseline", "explore", "report", "recover"),
        framework="sglang",
        kernel_enabled=False,
        max_minutes=60,
    )


def test_critic_prompt_advertises_verdict_map_section():
    text = _critic_prompt_text()
    assert "verdict_map" in text
    assert "Mutually exclusive" in text or "MUTUALLY EXCLUSIVE" in text or \
        "mutually exclusive" in text


def test_critic_prompt_still_documents_legacy_single_verdict():
    text = _critic_prompt_text()
    # Single-proposal path remains documented for kernel_opt / integrate.
    assert "single-proposal" in text.lower()
    assert "verdict:" in text.lower() or "'verdict'" in text.lower()


def test_critic_prompt_phase_review_section_points_at_verdict_map():
    text = _critic_prompt_text()
    # The phase-review block must steer specialist multi-variant
    # packets toward the v0.8 batch shape.
    assert "verdict_map" in text.lower()
    assert "needs_review" in text.lower()


# ===========================================================================
# 5. _materialize_approved_proposal — filter semantics (unit)
# ===========================================================================
@pytest.mark.asyncio
async def test_materialize_filter_drops_rejected_variants(tmp_path: Path):
    """Direct unit test for the new ``approved_variant_names`` filter.

    Bypasses the verdict_map handler so we can pin the
    ``_materialize_approved_proposal`` filter contract independently —
    important because Gap-11's correctness rides on the executor
    only ever seeing the approved subset, even if the handler logic
    above changes shape later.
    """
    # Restore the real method on the stub coord (the verdict_map
    # fixture overrode it with a capture). We use a fresh Coordinator
    # stub here with a stub TaskRegistry + bus instead.
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = _BareSharedState()
    coord.state = CoordinatorState()
    coord.cortex_kb = _StubCortexKB()
    coord.bus = _StubBus()
    coord._record_observation = AsyncMock()  # type: ignore[method-assign]

    create_calls: list[dict[str, Any]] = []

    class _StubTaskRegistry:
        async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
            create_calls.append(dict(kwargs))
            from inference_optimizer.orchestrator.task_registry import Task
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
    # Make sure shared_state has the minimum surface
    # ``_materialize_approved_proposal`` needs.
    class _MoreState(_BareSharedState):
        baseline_config_path: str = ""
        baseline_tput: float = 1000.0
        synergy_attempted: list[str] = field(default_factory=list)
        backends_search: dict = field(default_factory=dict)
        params_search: dict = field(default_factory=dict)
        current_best: dict = field(default_factory=dict)

    coord.shared_state = _MoreState()
    await coord._materialize_approved_proposal(
        pending, approved_variant_names={"v_a", "v_c"},
    )
    assert create_calls, "materialize must enqueue a task"
    grid = create_calls[0]["params"]["grid"]
    names = [v["name"] for v in grid]
    assert names == ["v_a", "v_c"]
    # critic_filtered_count records 4 - 2 = 2 dropped.
    assert create_calls[0]["params"]["critic_filtered_count"] == 2


@pytest.mark.asyncio
async def test_materialize_without_filter_keeps_full_grid(tmp_path: Path):
    """Legacy v0.6 single-verdict path: ``approved_variant_names=None``
    leaves the grid untouched."""
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.state = CoordinatorState()
    coord.cortex_kb = _StubCortexKB()
    coord.bus = _StubBus()
    coord._record_observation = AsyncMock()  # type: ignore[method-assign]
    create_calls: list[dict[str, Any]] = []

    class _StubTaskRegistry:
        async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
            create_calls.append(dict(kwargs))
            from inference_optimizer.orchestrator.task_registry import Task
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
        coord, variants=["v_a", "v_b", "v_c"],
    )

    @dataclass
    class _MoreState:
        baseline_config_path: str = ""
        baseline_tput: float = 1000.0
        cortex_session_id: str = "sid-test"
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


# ===========================================================================
# 6. _handle_delegate — explore grid re-routes through Critic verdict_map path
# ===========================================================================
#
# Until this guard landed, ``delegate{action_name='explore', params={grid}}``
# created an explore task directly via ``tasks.create_or_return_existing``,
# bypassing the per-variant Critic + KB priors lookup entirely. The new
# branch in ``_handle_delegate`` hands the intent off to
# ``_handle_propose_action`` so the proposal lands in ``pending_proposals``
# and the Critic emits a ``verdict_map``. Other delegate actions
# (specialist / integrate_patch / sweep / recover) and empty-grid
# explore delegates keep the legacy direct path.
def _delegate_coord(tmp_path: Path):
    """Coordinator double exposing just the surface ``_handle_delegate``
    needs to reach (and stop at) the explore re-route branch.

    Stubs out ``is_pruned`` / ``_sequence_denial_for_action`` so we
    never reach the per-test-irrelevant pruned + sequence
    sub-systems. (The legacy ``_cortex_t2_hook`` stub is gone — the
    method itself was removed alongside the T2/T3 protocol.)
    """
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path

    class _State(_BareSharedState):
        baseline_config_path: str = ""

        def is_pruned(self, _action_name: str) -> bool:
            return False

        def reset_policy_denial_streak(self, _action_name: str) -> None:
            return None

    c.shared_state = _State()
    c.state = CoordinatorState()
    c.cortex_kb = _StubCortexKB()
    c.bus = _StubBus()
    c._record_observation = AsyncMock()  # type: ignore[method-assign]
    c._record_policy_denied = AsyncMock()  # type: ignore[method-assign]
    c._sequence_denial_for_action = lambda *a, **k: None  # type: ignore[method-assign]
    # _handle_delegate's deprecation guard reads from policy directly.
    # We bypass it by short-circuiting the guard at the source.
    c.policy = None
    return c


@pytest.mark.asyncio
async def test_delegate_explore_with_grid_routes_to_pending_proposals(tmp_path: Path):
    """The Critic gate re-route: a delegate explore with a non-empty
    grid lands in ``pending_proposals`` (so Critic can review per
    variant) and never reaches ``tasks.create_or_return_existing``."""
    coord = _delegate_coord(tmp_path)
    create_calls: list[dict[str, Any]] = []

    class _TaskRegistry:
        async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
            create_calls.append(dict(kwargs))
            raise AssertionError(
                "delegate explore with grid must NOT create a task directly"
            )

    coord.tasks = _TaskRegistry()
    grid = [
        {"name": "v_a", "extra_args": "--flag-a",
         "provenance": "specialist:serving_specialist"},
        {"name": "v_b", "extra_args": "--flag-b",
         "provenance": "specialist:serving_specialist"},
    ]
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "explore",
            "params": {"grid": grid},
        },
    )
    await coord._handle_delegate("orchestration", intent)
    assert create_calls == []
    assert len(coord.state.pending_proposals) == 1
    pending = next(iter(coord.state.pending_proposals.values()))
    assert pending.action_name == "explore"
    assert pending.from_agent == "orchestration"
    assert pending.payload["params"]["grid"] == grid


@pytest.mark.asyncio
async def test_delegate_explore_with_empty_grid_does_not_route_to_critic(tmp_path: Path):
    """Empty / missing grid has nothing for the Critic to filter, so the
    re-route guard must NOT fire — the intent should fall through to
    the legacy direct-task path. We mock ``_handle_propose_action`` so
    the test stays scoped to the guard contract; the post-guard
    legacy path is exercised by separate integration tests."""
    coord = _delegate_coord(tmp_path)
    coord._handle_propose_action = AsyncMock()  # type: ignore[method-assign]
    # Stop execution at the first downstream call we don't want to
    # plumb. The guard runs BEFORE this point, so reaching a raised
    # exception proves the re-route did not intercept.
    coord.tasks = None
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "explore", "params": {"grid": []}},
    )
    with pytest.raises(AttributeError):
        await coord._handle_delegate("orchestration", intent)
    coord._handle_propose_action.assert_not_awaited()
    assert coord.state.pending_proposals == {}


@pytest.mark.asyncio
async def test_delegate_non_explore_action_does_not_route_to_critic(tmp_path: Path):
    """``delegate{action_name='specialist', ...}`` (and every other
    non-explore action) must NOT be intercepted by the explore guard,
    even if the caller smuggles in a ``grid`` param."""
    coord = _delegate_coord(tmp_path)
    coord._handle_propose_action = AsyncMock()  # type: ignore[method-assign]
    coord.tasks = None
    # Match the new async signature; the no-op coroutine has to
    # return ``None`` so ``await self._warm_specialist_params(params)``
    # in _handle_delegate doesn't blow up.
    async def _noop_warm(_params: dict) -> None:
        return None
    coord._warm_specialist_params = _noop_warm  # type: ignore[method-assign]
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "specialist",
            "params": {
                "domain": "serving_specialist",
                "gap_canonical_id": "gap-x",
                "grid": [{"name": "v_a"}],
            },
        },
    )
    with pytest.raises(AttributeError):
        await coord._handle_delegate("orchestration", intent)
    coord._handle_propose_action.assert_not_awaited()
    assert coord.state.pending_proposals == {}


# ===========================================================================
# 7. Specialist prompt — max_proposals self-curation contract (Section 1 + 8)
# ===========================================================================
def _build_specialist_prompt_text(max_proposals: int) -> str:
    from inference_optimizer.orchestrator.specialist_domains import get_domain
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        SpecialistPromptInputs,
        build_specialist_prompts,
    )

    inputs = SpecialistPromptInputs(
        task_id="t-test",
        domain=get_domain("serving_specialist"),
        max_turns=12,
        max_proposals=max_proposals,
        gap_canonical_id="gap-x",
    )
    system_prompt, user_prompt = build_specialist_prompts(inputs)
    return system_prompt + "\n" + user_prompt


def test_default_specialist_max_proposals_is_three():
    """Single-source-of-truth check: policy.py owns the cap (=3) and
    specialist_prompt_builder re-exports it. Both must agree."""
    from inference_optimizer.orchestrator.policy import (
        DEFAULT_SPECIALIST_MAX_PROPOSALS,
    )
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        DEFAULT_SPECIALIST_MAX_PROPOSALS as PROMPT_DEFAULT,
    )
    assert DEFAULT_SPECIALIST_MAX_PROPOSALS == 3
    assert PROMPT_DEFAULT == 3


def test_specialist_prompt_renders_max_proposals_5():
    """Caller can still override to a larger value at the prompt layer
    (the SpecialistRunner separately clamps to the policy cap)."""
    text = _build_specialist_prompt_text(max_proposals=5)
    # Section 8 hard cap line.
    assert "AT MOST **5** entries" in text
    # Section 1 autonomy paragraph mentions the cap once.
    assert "top-5" in text
    # The Critic-feedback warning is present so the specialist knows
    # marginal candidates have a cost.
    assert "reviews each surviving variant" in text


def test_specialist_prompt_renders_default_top_3_cap():
    text = _build_specialist_prompt_text(max_proposals=3)
    assert "AT MOST **3** entries" in text
    assert "top-3" in text
    # The legacy default 5 must not appear when the cap is 3.
    assert "AT MOST **5** entries" not in text


# ==============================================================================
# critic prompt builder (formerly test_critic_prompt_builder.py)
# ==============================================================================


class TestCriticPromptBuilder:
    """Tests for :mod:`critic_prompt_builder`."""

    @pytest.fixture
    def registry(self):
        from inference_optimizer.orchestrator.action_registry import ActionRegistry
        return ActionRegistry().load()

    @staticmethod
    def _rules_path():
        from inference_optimizer.paths import asset_system_prompts_dir
        return asset_system_prompts_dir() / "critic.md"

    def test_section_headers_present(self, registry):
        from inference_optimizer.orchestrator.system_prompts.critic_prompt_builder import (
            build_critic_prompt,
        )
        from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
            default_enabled_actions,
        )
        text = build_critic_prompt(
            action_registry=registry,
            enabled_actions=default_enabled_actions(no_kernel=False),
            framework="sglang",
            kernel_enabled=True,
            max_minutes=120,
            rules_fragment_path=self._rules_path(),
        )
        for header in (
            "## 1. MISSION",
            "## 2. RUN CONTEXT",
            "## 3. KNOWN ACTIONS",
            "## 4. DEFAULT VERDICT",
            "## 5. PHASE REVIEW CONTRACT (v0.8 §3.3)",
            "## 5b. KERNEL-OWNED CARVE-OUT",
            "## 6. RULES",
            "## 7. OUTPUT PROTOCOL",
        ):
            assert header in text, f"missing {header}"

    def test_deterministic(self, registry):
        from inference_optimizer.orchestrator.system_prompts.critic_prompt_builder import (
            build_critic_prompt,
        )
        from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
            default_enabled_actions,
        )
        kwargs = dict(
            action_registry=registry,
            enabled_actions=default_enabled_actions(no_kernel=False),
            framework="vllm",
            kernel_enabled=True,
            max_minutes=60,
            rules_fragment_path=self._rules_path(),
        )
        assert build_critic_prompt(**kwargs) == build_critic_prompt(**kwargs)

    def test_full_prompt_contains_all_registered_actions(self, registry):
        """Regression guard: every action in _meta must appear in §3."""
        from inference_optimizer.orchestrator.system_prompts.critic_prompt_builder import (
            build_critic_prompt,
        )
        text = build_critic_prompt(
            action_registry=registry,
            enabled_actions=registry.names(),
            framework="sglang",
            kernel_enabled=True,
            max_minutes=60,
            rules_fragment_path=self._rules_path(),
        )
        for name in registry.names():
            assert f"**{name}**" in text, f"action {name!r} missing from KNOWN ACTIONS"

    def test_validate_stack_in_both_modes(self, registry):
        from inference_optimizer.orchestrator.system_prompts.critic_prompt_builder import (
            build_critic_prompt,
        )
        from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
            default_enabled_actions,
        )
        for no_kernel in (False, True):
            enabled = default_enabled_actions(no_kernel=no_kernel)
            text = build_critic_prompt(
                action_registry=registry,
                enabled_actions=enabled,
                framework="sglang",
                kernel_enabled=not no_kernel,
                max_minutes=60,
                rules_fragment_path=self._rules_path(),
            )
            assert "validate_stack" in text, (
                f"validate_stack missing (no_kernel={no_kernel})"
            )

    def test_no_kernel_mode_drops_kernel_owned(self, registry):
        from inference_optimizer.orchestrator.system_prompts.critic_prompt_builder import (
            build_critic_prompt,
        )
        from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
            default_enabled_actions,
        )
        text = build_critic_prompt(
            action_registry=registry,
            enabled_actions=default_enabled_actions(no_kernel=True),
            framework="sglang",
            kernel_enabled=False,
            max_minutes=60,
            rules_fragment_path=self._rules_path(),
        )
        assert "## 5. KERNEL-OWNED CARVE-OUT" not in text
        for name in ("kernel_opt", "integrate", "deep_kernel_analysis"):
            assert f"**{name}**" not in text, (
                f"{name} should not appear in no-kernel catalogue"
            )


# ==============================================================================
# critic_robustness breakdown renderer
# (formerly test_critic_robustness_renderer_units.py)
# ==============================================================================


class TestCriticRobustnessRenderer:
    """Exercises the four observable shapes of the collector input: empty,
    prompt-only V1 payloads, V2 dicts with empty fields, and fully-populated
    entries with a truncated rationale.
    """

    @staticmethod
    def _render(payload):
        from inference_optimizer.breakdown.reporters._renderers import (
            critic_robustness as cr_mod,
        )
        return cr_mod.render({"critic_robustness": payload})

    def test_empty_returns_skipped(self):
        from inference_optimizer.breakdown.reporters.base import RenderedSection
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
        out = self._render([
            {"prompt": "x", "response": None, "decision": "", "rationale": ""},
        ])
        assert out.skipped is True
        assert any("non-actionable" in w for w in out.warnings)

    def test_populated_payload_renders_markdown_table(self):
        out = self._render([
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
        ])
        assert out.skipped is False
        assert "decision" in out.markdown_block
        assert "kernel_opt" in out.markdown_block

    def test_excess_rows_truncated_with_banner(self):
        from inference_optimizer.breakdown.reporters._renderers import (
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


# ==============================================================================
# N38 — per-action verdict_class metadata
# (formerly test_n38_action_verdict_class.py)
# ==============================================================================


class TestN38ActionVerdictClass:
    """N38 (May 2026) — structural fix: per-action ``verdict_class``
    metadata so newly added actions don't reintroduce the N33/N35/N37
    deadlocks. Pins the ActionMetadata field, the default classifier
    bucket mapping, the CriticAgentBackend constructor wiring, and the
    critic.md primary lookup.
    """

    def test_action_metadata_has_verdict_class_field(self):
        from inference_optimizer.orchestrator.action_registry import (
            ActionMetadata,
        )
        fields = {f.name for f in ActionMetadata.__dataclass_fields__.values()}
        assert "verdict_class" in fields, (
            "ActionMetadata must declare verdict_class field so per-action "
            "policy can be looked up in critic review_constraints"
        )

    def test_default_classifier_covers_all_registered_actions(self):
        from inference_optimizer.orchestrator.action_registry import (
            ActionRegistry,
        )
        reg = ActionRegistry().load()
        all_actions = reg.all()
        assert all_actions, "expected ActionRegistry to load >= 1 action"
        missing = [a.name for a in all_actions if not a.verdict_class]
        assert not missing, (
            f"actions missing verdict_class default: {missing} -- update the "
            f"default classifier in action_registry.py or add the field to "
            f"the yaml"
        )

    def test_default_classifier_matches_expected_buckets(self):
        from inference_optimizer.orchestrator.action_registry import (
            ActionRegistry,
        )
        reg = ActionRegistry().load()

        def klass(name: str) -> str:
            a = reg.get(name)
            assert a is not None, f"action {name!r} not registered"
            return a.verdict_class

        assert klass("integrate") == "promotion"
        for n in ("report", "session_breakdown", "target_analysis"):
            assert klass(n) == "archival", n
        registered_exploration = (
            "baseline", "profile", "roofline", "explore", "sweep",
            "kernel_opt", "operator_tuning", "vendor_kernel_config",
            "deep_kernel_analysis", "recover",
        )
        for n in registered_exploration:
            if reg.get(n) is None:
                continue
            assert klass(n) == "exploration", n

    def test_critic_agent_backend_accepts_action_verdict_policy(self, tmp_path):
        from inference_optimizer.orchestrator.backends.critic_agent import (
            CriticAgentBackend,
        )
        root = tmp_path / "critic-agent"
        (root / "runtime").mkdir(parents=True)
        (root / "runtime" / "cli.py").write_text("# stub")
        sd = tmp_path / "session"
        sd.mkdir()

        def _fake_client_factory():
            class _C: pass
            return _C()

        def _fake_runtime_caller_factory():
            def _caller(call): return None
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
            "baseline": "exploration", "integrate": "promotion",
        }

    def test_critic_agent_backend_injects_policy_into_judge_bundle(self, tmp_path):
        import asyncio
        import json as _json
        from inference_optimizer.orchestrator.backends.critic_agent import (
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
            def __init__(self): self.chat = _FakeChat(captured_bundle)
        class _FakeChat:
            def __init__(self, bucket): self.completions = _FakeCompletions(bucket)
        class _FakeCompletions:
            def __init__(self, bucket): self._b = bucket
            async def create(self, *, model, messages, max_completion_tokens):
                user_msg = messages[-1]["content"]
                self._b["user_prompt"] = user_msg
                class _Choice:
                    message = type("M", (), {"content": _json.dumps({
                        "review_verdicts": [],
                    })})()
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
                    call.out_path.write_text(_json.dumps({
                        "intent_envelope": {"intents": []},
                    }), encoding="utf-8")
            return _caller

        backend = CriticAgentBackend(
            critic_agent_root=root,
            session_dir=sd,
            codex_client_factory=_FakeAsyncOpenAI,
            runtime_caller_factory=_fake_runtime_caller_factory,
            static_context={"model": "m", "framework": "sglang"},
            action_verdict_policy={
                "params": "exploration", "integrate": "promotion",
            },
        )
        asyncio.run(backend.run(prompt="hello"))

        assert "action_verdict_policy" in captured_bundle.get("user_prompt", ""), (
            "action_verdict_policy must appear in the JSON prompt sent to "
            "the LLM-critic so it can look up each proposal's class"
        )
        assert "promotion" in captured_bundle["user_prompt"]

    def test_critic_md_mentions_action_verdict_policy_lookup(self):
        from pathlib import Path as _Path
        p = (
            _Path(__file__).resolve().parent.parent
            / "orchestrator" / "system_prompts" / "critic.md"
        )
        text = p.read_text(encoding="utf-8")
        assert "action_verdict_policy" in text, (
            "critic.md must mention action_verdict_policy so the LLM-critic "
            "treats it as the primary per-proposal lookup; otherwise newly "
            "added actions will hit the same N33/N35/N37 chicken-and-egg "
            "deadlock"
        )
        for klass in ("archival", "exploration", "promotion"):
            assert klass in text.lower(), klass
