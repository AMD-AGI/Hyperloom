# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Critic ``review_verdict`` handling: the gate, routing, the rebroadcast, the patch-verdict mirror and materialisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.coordinator_state import CoordinatorState
from hyperloom.orchestrator.state.pending_proposal import PendingProposal
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.gate import (
    INTEGRATE_PATCH_PERMISSIVE_VERDICTS,
    PolicyGate,
)


@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


def test_policy_gate_accepts_a_critic_verdict(gate):
    gate.validate_intent(
        "critic",
        Intent(type=IntentType.REVIEW_VERDICT, payload={"target_proposal_msg_id": "msg-1", "verdict": "approve"}),
    )


# Coordinator helpers — _handle_review_verdict dispatch
@dataclass
class _BareSharedState:
    """SharedState double exposing the fields the review-verdict handler touches."""

    recipe_kb_session_id: str = "sid-test"
    save_count: int = 0
    # Empty string means "nothing in flight"; the auto-roofline dispatch gate is a no-op.
    auto_roofline_pending_task_id: str = ""
    # Fields read by inject_explore_runtime_params and related helpers.
    baseline_runtime_sec: float = 0.0
    baseline_accuracy: float = 0.0
    baseline_warm_runtime_sec: float = 0.0
    explore_search: dict = field(default_factory=dict)
    macro_cycle: int = 0
    specialist_rounds: list = field(default_factory=list)

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1

    def record_specialist_patch_verdict(self, specialist_task_id: str, verdict: str) -> None:
        """No-op stub; tests that care about this use _PatchVerdictSharedState."""


@dataclass
class _BusMessage:
    from_agent: str
    to_agent: str
    topic: str
    payload: dict[str, Any]
    in_reply_to: str = ""
    msg_id: str = ""


class _StubBus:
    """MessageBus double that captures every appended message."""

    def __init__(self) -> None:
        self.messages: list[_BusMessage] = []

    async def append_and_seq(self, msg: Any) -> Any:
        bm = _BusMessage(
            from_agent=getattr(msg, "from_agent", ""),
            to_agent=getattr(msg, "to_agent", ""),
            topic=getattr(msg, "topic", ""),
            payload=dict(getattr(msg, "payload", {}) or {}),
            in_reply_to=getattr(msg, "in_reply_to", "") or "",
            msg_id=getattr(msg, "msg_id", ""),
        )
        self.messages.append(bm)
        return None


class _StubRecipeKB:
    enabled: bool = True

    def __init__(self) -> None:
        self.verify_calls: list[dict[str, Any]] = []

    def verify(self, **kwargs: Any) -> None:
        self.verify_calls.append(dict(kwargs))


def _posted_verdict(coord: Coordinator) -> str:
    return next(m.payload["verdict"] for m in reversed(coord.bus.messages) if m.topic == "review_verdict")


@pytest.fixture
def coord(tmp_path: Path):
    """Coordinator-shaped object with just enough plumbing for the review-verdict path."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareSharedState()
    c.state = CoordinatorState()
    c.knowledge_plane = KnowledgePlane(recipe_kb=_StubRecipeKB())
    c.bus = _StubBus()
    c.bus.record_observation = AsyncMock()  # type: ignore[method-assign]
    materialise_calls: list[PendingProposal] = []
    c._materialise_calls = materialise_calls  # type: ignore[attr-defined]

    async def _mat(pending: PendingProposal) -> None:
        materialise_calls.append(pending)

    c.proposals.materialize_approved_proposal = _mat  # type: ignore[method-assign]
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
async def test_an_approve_materialises_the_proposal(coord):
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
    await coord.router._handle_review_verdict("critic", intent)
    bus_msgs = [m for m in coord.bus.messages if m.topic == "review_verdict"]
    assert len(bus_msgs) == 1
    assert bus_msgs[0].payload["verdict"] == "approve"
    assert coord._materialise_calls == [pending]
    assert pending.proposal_msg_id not in coord.state.pending_proposals
    assert _posted_verdict(coord) == "approve"


@pytest.mark.asyncio
async def test_verdict_for_unknown_proposal_logs_observation(coord):
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "ghost-msg",
            "verdict": "approve",
        },
    )
    await coord.router._handle_review_verdict("critic", intent)
    coord.bus.record_observation.assert_awaited()
    assert coord._materialise_calls == []


@pytest.mark.asyncio
async def test_an_approve_after_the_timeout_deny_creates_nothing(coord):
    """The timeout deny is final: it removes the pending proposal, so a late approve has no target."""
    pending = _seed_explore_proposal(coord, msg_id="msg-timed-out", variants=["v_a"])
    coord.state.pending_proposals.pop(pending.proposal_msg_id)
    intent = Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={"target_proposal_msg_id": pending.proposal_msg_id, "verdict": "approve"},
    )

    await coord.router._handle_review_verdict("critic", intent)

    (call,) = coord.bus.record_observation.await_args_list
    assert call.args[2]["kind"] == "verdict_for_unknown_proposal"
    assert coord._materialise_calls == []
    assert [m for m in coord.bus.messages if m.topic == "review_verdict"] == []


@pytest.mark.asyncio
async def test_single_verdict_rebroadcast_carries_full_advisory_fieldset(coord):
    """The rebroadcast payload and the compact inbox line both flow through the one serializer, carrying the full advisory field set."""
    from hyperloom.orchestrator.loop.conversation import _format_inbox_event
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
    await coord.router._handle_review_verdict("critic", intent)

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
    from hyperloom.orchestrator.loop.conversation import _format_inbox_event
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
    await coord.router._handle_review_verdict("critic", intent)
    payload = [m for m in coord.bus.messages if m.topic == "review_verdict"][0].payload
    for key in ("required_evidence", "risks", "advice_text", "notes"):
        assert key not in payload

    msg = Message.new("critic", "orchestration", "review_verdict", payload)
    msg.seq = 6
    line = _format_inbox_event(msg)
    assert "required_evidence" not in line
    assert "risks=" not in line
    assert "advice=" not in line


# A reject ends the proposal, and the verdict is mirrored for the integrate_patch gate
@dataclass
class _PatchVerdictSharedState(_BareSharedState):
    """Adds the patch-verdict mirror the integrate_patch gate reads."""

    patch_verdicts: dict[str, str] = field(default_factory=dict)

    def record_specialist_patch_verdict(self, specialist_task_id: str, verdict: str) -> None:
        self.patch_verdicts[specialist_task_id] = verdict.strip().lower()


def _seed_pending(coord, msg_id: str, action_name: str, params: dict[str, Any]) -> PendingProposal:
    pending = PendingProposal(
        proposal_msg_id=msg_id,
        from_agent="orchestration",
        action_name=action_name,
        predicted_gain_pct=0.0,
        payload={"action_name": action_name, "params": params},
    )
    coord.state.pending_proposals[msg_id] = pending
    return pending


async def _rule(coord, msg_id: str, **payload: Any) -> None:
    intent = Intent(type=IntentType.REVIEW_VERDICT, payload={"target_proposal_msg_id": msg_id, **payload})
    await coord.router._handle_review_verdict("critic", intent)


@pytest.mark.asyncio
async def test_a_reject_ends_the_proposal(coord):
    _seed_pending(coord, "msg-real", "specialist", {"task_id": "t-3"})
    await _rule(coord, "msg-real", verdict="reject")

    assert _posted_verdict(coord) == "reject"
    assert coord._materialise_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_name", "params"),
    [
        pytest.param("specialist", {"task_id": "t-permit"}, id="specialist"),
        pytest.param("integrate_patch", {"specialist_task_id": "t-permit"}, id="integrate_patch"),
    ],
)
async def test_a_rejected_proposal_is_not_a_landing_permit(coord, action_name, params):
    """The mirrored verdict is what the integrate_patch gate consults."""
    coord.shared_state = _PatchVerdictSharedState()
    _seed_pending(coord, "msg-permit", action_name, params)
    await _rule(coord, "msg-permit", verdict="reject")

    assert coord.shared_state.patch_verdicts["t-permit"] == "reject"
    assert coord.shared_state.patch_verdicts["t-permit"] not in INTEGRATE_PATCH_PERMISSIVE_VERDICTS


@pytest.mark.asyncio
async def test_an_approved_explore_materialises_its_whole_grid(tmp_path: Path):
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.state = CoordinatorState()
    coord.knowledge_plane = KnowledgePlane(recipe_kb=_StubRecipeKB())
    coord.bus = _StubBus()
    coord.bus.record_observation = AsyncMock()  # type: ignore[method-assign]
    create_calls: list[dict[str, Any]] = []

    class _StubTaskRegistry:
        async def create_or_return_existing(self, spec):
            create_calls.append(dict(vars(spec)))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-explore-full",
                    kind=spec.kind,
                    state="queued",
                    params=spec.params,
                    idempotency_key=spec.idempotency_key,
                ),
                False,
            )

    coord.tasks = _StubTaskRegistry()
    pending = _seed_explore_proposal(
        coord,
        variants=["v_a", "v_b", "v_c"],
    )

    @dataclass
    class _MoreState(_BareSharedState):
        baseline_config_path: str = ""
        baseline_tput: float = 1000.0
        backends_search: dict = field(default_factory=dict)
        params_search: dict = field(default_factory=dict)
        current_best: dict = field(default_factory=dict)

    coord.shared_state = _MoreState()
    await coord.proposals.materialize_approved_proposal(pending)
    grid = create_calls[0]["params"]["grid"]
    names = [v["name"] for v in grid]
    assert names == ["v_a", "v_b", "v_c"]


# handle_delegate — explore grid runs directly (no Critic pre-review)
def _delegate_coord(tmp_path: Path):
    """Coordinator double reaching the direct explore-task creation path."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path

    @dataclass
    class _State(_BareSharedState):
        baseline_config_path: str = ""
        baseline_tput: float = 0.0
        current_best: dict = field(default_factory=dict)
        tick: int = 0

        def is_pruned(self, _action_name: str) -> bool:
            return False

    c.shared_state = _State()
    c.state = CoordinatorState()
    c.knowledge_plane = KnowledgePlane(recipe_kb=_StubRecipeKB())
    c.bus = _StubBus()
    c.bus.record_observation = AsyncMock()  # type: ignore[method-assign]
    c.router.record_policy_denied = AsyncMock()  # type: ignore[method-assign]
    c.dispatcher.admission_denial_for_action = lambda *a, **k: None  # type: ignore[method-assign]
    c.dispatcher.registry_lanes_ttl = lambda _name: (set(), 0)  # type: ignore[method-assign]
    c.policy = None
    return c


@pytest.mark.asyncio
async def test_delegate_explore_with_grid_creates_task_directly(tmp_path: Path):
    """A delegate explore with a non-empty grid creates an explore task directly, grid forwarded verbatim."""
    coord = _delegate_coord(tmp_path)
    create_calls: list[dict[str, Any]] = []

    class _TaskRegistry:
        async def queued(self) -> list:
            return []

        async def running(self) -> list:
            return []

        async def create_or_return_existing(self, spec):
            create_calls.append(dict(vars(spec)))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-explore-direct",
                    kind=spec.kind,
                    state="queued",
                    params=spec.params,
                    idempotency_key=spec.idempotency_key,
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
    await coord.router.handle_delegate("orchestration", intent)
    assert coord.state.pending_proposals == {}
    assert len(create_calls) == 1
    assert create_calls[0]["kind"] == "explore"
    assert create_calls[0]["params"]["grid"] == grid


@pytest.mark.asyncio
async def test_delegate_explore_seeds_the_stack_with_the_anchor(tmp_path: Path):
    """A delegate explore carries current_best's args/envs, not just its throughput."""
    coord = _delegate_coord(tmp_path)
    coord.shared_state.baseline_tput = 4663.79
    coord.shared_state.current_best = {
        "tput": 7725.6,
        "extra_server_args": "--kv-cache-dtype fp8_e4m3 --max-num-seqs 64",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
    }
    created: list[dict[str, Any]] = []

    class _TaskRegistry:
        async def queued(self) -> list:
            return []

        async def running(self) -> list:
            return []

        async def create_or_return_existing(self, spec):
            created.append(dict(spec.params))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-explore-stack",
                    kind=spec.kind,
                    state="queued",
                    params=spec.params,
                    idempotency_key=spec.idempotency_key,
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
    await coord.router.handle_delegate("orchestration", intent)
    assert len(created) == 1
    params = created[0]
    assert params["base_tput"] == 7725.6
    assert params["base_extra_args"] == "--kv-cache-dtype fp8_e4m3 --max-num-seqs 64"
    assert params["base_extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}


class TestN38ActionVerdictClass:
    """Per-action ``verdict_class`` metadata so new actions don't reintroduce prior deadlocks."""

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
