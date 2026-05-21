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
  filters the materialised grid down to the ``approve`` subset, and
  fires :meth:`_cortex_t3_critic_rejected` for every ``reject`` so the
  KB view captures critic-rejected edges before the executor runs.
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
from inference_optimizer.orchestrator.intent_parser import (
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
    kb_edge_ids: dict[str, str] | None = None,
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
        kb_edge_ids=dict(kb_edge_ids or {}),
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
@pytest.mark.asyncio
async def test_verdict_map_rejected_variants_fire_kb_refuted(coord):
    pending = _seed_explore_proposal(coord, kb_edge_ids={
        "v_a": "edge-a", "v_b": "edge-b", "v_c": "edge-c", "v_d": "edge-d",
    })
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict_map": {
                "v_a": {"verdict": "approve"},
                "v_b": {"verdict": "reject", "rationale": "kb refuted"},
                "v_c": {"verdict": "reject", "rationale": "duplicate"},
                "v_d": {"verdict": "needs_review"},  # NOT refuted
            },
        },
    )
    await coord._handle_review_verdict("critic", intent)
    refuted = coord.cortex_kb.verify_calls
    # Two refute calls — one per rejected variant. needs_review is not a refute.
    refuted_edges = sorted(r["edge_id"] for r in refuted)
    assert refuted_edges == ["edge-b", "edge-c"]
    for call in refuted:
        assert call["outcome"] == "refuted"
        # Idempotency key carries the variant name at the tail.
        assert call["idempotency_key"].endswith(":v_b") or \
               call["idempotency_key"].endswith(":v_c")
        assert call["promote_authority"] is None


@pytest.mark.asyncio
async def test_verdict_map_skip_kb_refuted_when_no_edge_id(coord):
    """No T2 edge (e.g. --degraded-kb run) → silently skip the refute
    so the verdict_map path still works end-to-end."""
    pending = _seed_explore_proposal(coord, kb_edge_ids={})
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
    await coord._handle_review_verdict("critic", intent)
    assert coord.cortex_kb.verify_calls == []


@pytest.mark.asyncio
async def test_verdict_map_skip_kb_refuted_when_cortex_disabled(coord):
    coord.cortex_kb = None  # type: ignore[assignment]
    pending = _seed_explore_proposal(coord, kb_edge_ids={"v_b": "edge-b"})
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
    # Must not raise.
    await coord._handle_review_verdict("critic", intent)


@pytest.mark.asyncio
async def test_verdict_map_skip_kb_refuted_when_no_session_id(coord):
    coord.shared_state.cortex_session_id = ""
    pending = _seed_explore_proposal(coord, kb_edge_ids={"v_b": "edge-b"})
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
    await coord._handle_review_verdict("critic", intent)
    assert coord.cortex_kb.verify_calls == []


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
        kb_edge_ids={
            "v_a": "edge-a", "v_b": "edge-b",
            "v_c": "edge-c", "v_d": "edge-d",
        },
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
    # KB edge stamping still works under the filter.
    assert grid[0].get("kb_edge_id") == "edge-a"
    assert grid[1].get("kb_edge_id") == "edge-c"
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
        coord, variants=["v_a", "v_b", "v_c"], kb_edge_ids={},
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

        def save(self, _session_dir):
            self.save_count += 1

    coord.shared_state = _MoreState()
    await coord._materialize_approved_proposal(pending)
    grid = create_calls[0]["params"]["grid"]
    names = [v["name"] for v in grid]
    assert names == ["v_a", "v_b", "v_c"]
    assert "critic_filtered_count" not in create_calls[0]["params"]
