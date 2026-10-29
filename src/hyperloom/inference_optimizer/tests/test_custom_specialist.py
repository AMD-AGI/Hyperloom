# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Operator-defined ``custom_specialist`` type: prompts, gate, CLI, and its Coordinator-owned dispatch and bench."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
from hashlib import sha1
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hyperloom.inference_optimizer.cli import (
    _apply_custom_specialist_resume,
    OrchestrationPromptContext,
    _build_orchestration_prompt,
    _custom_specialist_conflict,
    _load_custom_specialist_args,
    main,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.phases.dispatch_order import in_dispatch_order
from hyperloom.orchestrator.policy.gate import PolicyDenied, PolicyGate
from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_execution_budget,
    _section_gap,
    _section_hardware,
    _section_identity,
    _section_iron_rules,
    _section_mandate,
    _section_output_protocol,
    _section_pd_disaggregation,
    _section_source_hint,
    build_specialist_prompts,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.specialists.dispatch import SpecialistDispatchCollaborator
from hyperloom.orchestrator.specialists.domains import KNOWLEDGE_DOMAIN_TAGS, get_domain
from hyperloom.orchestrator.specialists.runner import SpecialistRunner, SpecialistSubprocessConfig
from hyperloom.orchestrator.state.objective import build_objective
from hyperloom.orchestrator.state.shared_state import SharedState

_PROMPT = "Focus on RCCL all-reduce.\n<keep>```verbatim```</keep>"
_DESCRIPTION = "Tunes RCCL all-reduce for TP communication."


def _custom_domain():
    return replace(get_domain("custom_specialist"), description=_DESCRIPTION)


def _inputs(domain, **kw: Any) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        domain=domain,
        gap_canonical_id="gap.x",
        gap_symptom="s",
        framework="sglang",
        workspace_path="/tmp/w",
        **kw,
    )


def _headers(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("## ")]


def _joined(*sections: list[str]) -> str:
    out: list[str] = []
    for sec in sections:
        if sec:
            if out:
                out.append("")
            out.extend(sec)
    return "\n".join(out) + "\n"


# Specialist prompt
def test_custom_system_prompt_carries_operator_identity_and_verbatim_focus():
    inp = _inputs(_custom_domain(), custom_specialist_prompt=_PROMPT)
    system, _ = build_specialist_prompts(inp)

    assert system == _joined(_section_identity(inp), _section_output_protocol(inp), _section_iron_rules(inp))
    assert system.splitlines()[:7] == [
        "## 1. IDENTITY & AUTONOMY",
        "",
        "You are a fully autonomous **custom_specialist** dispatched by the",
        "Hyperloom Coordinator. Layer: operator-defined.",
        "KB anchor: custom_specialist.",
        "",
        f"Description: {_DESCRIPTION}",
    ]
    focus = system.split("### Domain focus — custom_specialist\n\n", 1)[1].split("\n\n## 8. OUTPUT PROTOCOL", 1)[0]
    assert focus == _PROMPT
    assert _headers(system) == [
        "## 1. IDENTITY & AUTONOMY",
        "## 8. OUTPUT PROTOCOL",
        "## 9. IRON RULES (Inv-5.1 / Inv-5.3)",
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "research",
            ["## 0. MANDATE", "## 2. HARDWARE CONTEXT", "## 3. GAP STATEMENT", "## 10. NOTES FROM ORCHESTRATION"],
        ),
        (
            "patch",
            [
                "## 0. MANDATE",
                "## 2. HARDWARE CONTEXT",
                "## 3. GAP STATEMENT",
                "## 7. LOCAL SOURCE NAVIGATION HINT",
                "## 10. NOTES FROM ORCHESTRATION",
            ],
        ),
    ],
)
def test_custom_user_prompt_is_minimal(mode, expected):
    inp = _inputs(_custom_domain(), custom_specialist_prompt=_PROMPT, mode=mode, notes="n")
    _, user = build_specialist_prompts(inp)

    source_hint = _section_source_hint(inp) if mode == "patch" else []
    assert user == _joined(
        _section_mandate(inp),
        _section_hardware(inp),
        _section_pd_disaggregation(inp),
        _section_execution_budget(inp),
        _section_gap(inp),
        source_hint,
        ["## 10. NOTES FROM ORCHESTRATION", "", "n"],
    )
    assert _headers(user) == expected


def test_builtin_tagged_custom_specialist_renders_custom_focus_only_when_configured():
    serving = get_domain("serving_specialist")
    configured, _ = build_specialist_prompts(
        _inputs(serving, extra_focus_tags=("custom_specialist",), custom_specialist_prompt=_PROMPT)
    )
    unconfigured, _ = build_specialist_prompts(_inputs(serving, extra_focus_tags=("custom_specialist",)))
    generic_tag, _ = build_specialist_prompts(
        _inputs(serving, extra_focus_tags=("custom",), custom_specialist_prompt=_PROMPT)
    )
    untagged, _ = build_specialist_prompts(_inputs(serving))
    untagged_configured, _ = build_specialist_prompts(_inputs(serving, custom_specialist_prompt=_PROMPT))

    assert f"### Domain focus — custom_specialist\n\n{_PROMPT}\n" in configured
    assert unconfigured == untagged
    assert generic_tag == untagged_configured


@dataclass
class _StubTask:
    task_id: str
    params: dict[str, Any]
    kind: str = "specialist"


async def test_runner_prepare_reads_the_custom_definition_from_shared_state(tmp_path):
    runner = SpecialistRunner(subprocess_config=SpecialistSubprocessConfig(), session_dir=tmp_path)
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    task = _StubTask(
        task_id="task-custom",
        params={"domain": "custom_specialist", "gap_canonical_id": "gap.x", "mode": "research"},
    )

    prep = await runner._prepare(RunnerContext(task=task, lease=None, extra={"shared_state": state}))

    assert prep.domain.description == _DESCRIPTION
    assert f"Description: {_DESCRIPTION}\n" in prep.system_prompt
    assert f"### Domain focus — custom_specialist\n\n{_PROMPT}\n" in prep.system_prompt


def _coordinator(session_dir: Path, state: SharedState):
    from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan

    state.save(session_dir)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    coord = Coordinator(
        session_dir=session_dir,
        backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
        role_registry=default_role_registry(),
        advisory=AdvisoryServices(knowledge_plane=None),
    )
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    return coord


async def test_custom_specialist_prompt_shows_the_experience_kb_block_its_warmup_records(tmp_path):
    block = "Experience exp-1: raise chunked prefill"
    evidence = SimpleNamespace(
        read_id="read-1",
        rendered_refs=({"id": "exp-1"},),
        status="completed",
        prompt_block=block,
        experiences=({"id": "exp-1"},),
    )
    coord = _coordinator(
        tmp_path, SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    )
    coord.__dict__["_experience_kb"] = SimpleNamespace(read_for_specialist=lambda state, params: evidence)
    params = {"domain": "custom_specialist", "gap_canonical_id": "gap.x", "mode": "research"}

    await coord.specialist_dispatch.warm_specialist_params(params)
    runner = SpecialistRunner(subprocess_config=SpecialistSubprocessConfig(), session_dir=tmp_path)
    prep = await runner._prepare(
        RunnerContext(task=_StubTask("task-custom", params), lease=None, extra={"shared_state": coord.shared_state}),
    )

    assert [r["read_id"] for r in coord.shared_state.experience_kb_injections] == ["read-1"]
    assert block in prep.user_prompt


# Orchestration prompt
@pytest.mark.parametrize("phase", ["ENABLEMENT", "FRAMEWORK_AGENT"])
def test_orchestration_prompt_never_offers_custom_specialist(phase):
    prompt = _build_orchestration_prompt(
        OrchestrationPromptContext(
            no_kernel=False, framework="sglang", objective=build_objective({"MAX_HOURS": 1}), max_minutes=60
        ),
        phase=phase,
    )
    emit = next(line for line in prompt.splitlines() if "delegate{action_name='specialist'" in line)
    domains = emit.split("domain=<one of ", 1)[1].split(">", 1)[0]

    assert domains == (
        "serving_specialist|kernel_switch_specialist|comm_specialist|compiler_specialist|system_specialist"
        "|speculative_decoding_specialist"
        "|candidate_discovery_specialist|research_scout_specialist|static_recon_specialist"
        "|framework_rewrite_specialist"
    )
    assert prompt.count("custom_specialist") == 0


# PolicyGate
def _delegate(params: dict[str, Any]) -> Intent:
    return Intent(type=IntentType.DELEGATE, payload={"action_name": "specialist", "params": params})


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize(
    "params",
    [
        {"domain": "custom_specialist", "gap_canonical_id": "gap.x"},
        {"domain": "serving_specialist", "tags": ["custom_specialist"], "gap_canonical_id": "gap.x"},
        {"tags": ["custom_specialist"], "gap_canonical_id": "gap.x"},
        {"scope": "freeform", "domain": "custom_specialist", "task_description": "do it"},
    ],
)
def test_gate_denies_every_custom_specialist_delegate(params, configured):
    gate = PolicyGate(role_registry=default_role_registry())
    gate.shared_state = SharedState(custom_specialist_prompt=_PROMPT if configured else "")
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate(params))
    assert exc.value.rule == "specialist_custom_coordinator_owned"


def test_gate_allows_a_builtin_dispatch_with_the_generic_custom_tag():
    gate = PolicyGate(role_registry=default_role_registry())
    gate.shared_state = SharedState()
    gate.validate_intent(
        "orchestration",
        _delegate({"domain": "serving_specialist", "tags": ["framework", "custom"], "gap_canonical_id": "gap.x"}),
    )


def test_knowledge_domain_tags_do_not_include_the_custom_specialist():
    assert KNOWLEDGE_DOMAIN_TAGS == (
        "framework",
        "kernel_agent",
        "communication",
        "compiler",
        "systems",
        "speculative_decoding",
        "pr_intelligence",
        "research_scout",
        "static_recon",
    )


# CLI
def _ns(**kw: Any) -> argparse.Namespace:
    base = {
        "custom_specialist_prompt_file": None,
        "custom_specialist_description": None,
        "custom_specialist_only": False,
        "orch_prompt": None,
        "no_framework_agent": False,
        "research_lane_capacity": 4,
        "reset_state": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_load_custom_specialist_args_strips_and_sets_values(tmp_path):
    path = tmp_path / "p.md"
    path.write_text(f"\n{_PROMPT}\n\n", encoding="utf-8")
    args = _ns(custom_specialist_prompt_file=str(path), custom_specialist_description=f"  {_DESCRIPTION} ")

    _load_custom_specialist_args(args)

    assert (args.custom_specialist_prompt, args.custom_specialist_description) == (_PROMPT, _DESCRIPTION)


def test_load_custom_specialist_args_absent_flags_leave_empty_prompt():
    args = _ns()
    _load_custom_specialist_args(args)
    assert args.custom_specialist_prompt == ""


@pytest.mark.parametrize(
    ("content", "description", "message"),
    [
        (None, _DESCRIPTION, "must be passed together"),
        ("x", None, "must be passed together"),
        ("missing", _DESCRIPTION, "cannot read"),
        ("  \n", _DESCRIPTION, "is empty"),
        ("a" * (16 * 1024 + 1), _DESCRIPTION, "exceeds 16384 bytes"),
        ("x", "   ", "single non-empty line"),
        ("x", "two\nlines", "single non-empty line"),
        ("x", "d" * 201, "exceeds 200 characters"),
    ],
)
def test_load_custom_specialist_args_rejects_bad_input(tmp_path, content, description, message):
    path = None
    if content == "missing":
        path = str(tmp_path / "absent.md")
    elif content is not None:
        (tmp_path / "p.md").write_text(content, encoding="utf-8")
        path = str(tmp_path / "p.md")
    with pytest.raises(ValueError, match=message):
        _load_custom_specialist_args(_ns(custom_specialist_prompt_file=path, custom_specialist_description=description))


def test_resume_without_flags_keeps_the_stored_definition():
    state = SharedState(custom_specialist_prompt="old", custom_specialist_description="old desc")
    _apply_custom_specialist_resume(_ns(custom_specialist_prompt=""), state)
    assert (state.custom_specialist_prompt, state.custom_specialist_description) == ("old", "old desc")


def test_resume_with_a_new_definition_replaces_the_stored_one():
    state = SharedState(custom_specialist_prompt="old", custom_specialist_description="old desc")
    _apply_custom_specialist_resume(
        _ns(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION), state
    )
    assert (state.custom_specialist_prompt, state.custom_specialist_description) == (_PROMPT, _DESCRIPTION)


@pytest.mark.parametrize(
    ("overrides", "state_kw", "expected"),
    [
        ({}, {}, ""),
        ({"orch_prompt": "x"}, {}, ""),
        (
            {"no_framework_agent": True},
            {},
            "the FRAMEWORK_AGENT phase is disabled, so custom_specialist cannot be guaranteed to run",
        ),
        (
            {},
            {"framework_agent_phase_enabled": False},
            "the FRAMEWORK_AGENT phase is disabled, so custom_specialist cannot be guaranteed to run",
        ),
        ({"research_lane_capacity": 0}, {}, "--research-lane-capacity 0 builds no specialist executor"),
        ({"reset_state": True}, {}, ""),
    ],
)
def test_custom_specialist_conflicts(overrides, state_kw, expected):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION, **state_kw)
    assert _custom_specialist_conflict(_ns(**overrides), state) == expected


def test_conflicts_apply_to_a_definition_passed_on_this_launch():
    args = _ns(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    state = SharedState(framework_agent_phase_enabled=False)
    assert _custom_specialist_conflict(args, state) == (
        "the FRAMEWORK_AGENT phase is disabled, so custom_specialist cannot be guaranteed to run"
    )


@pytest.mark.parametrize("stored", [True, False])
def test_reset_state_conflicts_with_a_definition_passed_on_this_launch(stored):
    args = _ns(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION, reset_state=True)
    state = SharedState(custom_specialist_prompt=_PROMPT if stored else "")
    assert _custom_specialist_conflict(args, state) == (
        "--reset-state wipes the custom specialist definition; reset first, then launch with the flags"
    )


def test_conflicts_are_ignored_when_no_custom_specialist_is_configured():
    args = _ns(orch_prompt="x", no_framework_agent=True, research_lane_capacity=0, reset_state=True)
    assert _custom_specialist_conflict(args, SharedState()) == ""


def test_custom_specialist_fields_roundtrip_and_default_on_old_state():
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    restored = SharedState.from_dict(state.to_dict())
    old = SharedState.from_dict({"session_id": "old", "custom_specialist_dispatched": True})

    assert (restored.custom_specialist_prompt, restored.custom_specialist_description) == (_PROMPT, _DESCRIPTION)
    assert (old.custom_specialist_prompt, old.custom_specialist_description) == ("", "")


# Coordinator-owned dispatch
def _custom_key(cycle: int = 0, prompt: str = _PROMPT) -> str:
    return f"custom-c{cycle}-{sha1(prompt.encode(), usedforsecurity=False).hexdigest()[:8]}"


@pytest.fixture
def custom_coord(tmp_path: Path, monkeypatch):
    """Coordinator stand-in with a configured custom specialist and a mocked router."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    c.shared_state.phase = "FRAMEWORK_AGENT"
    source_root = tmp_path / "framework"
    (source_root / ".git").mkdir(parents=True)
    c.shared_state.framework_repo_path = str(source_root)
    c.tasks = SimpleNamespace(
        queued=AsyncMock(return_value=[]),
        running=AsyncMock(return_value=[]),
        find_by_idempotency_key=AsyncMock(return_value=None),
    )
    c.bus = SimpleNamespace(record_observation=AsyncMock())
    monkeypatch.setattr(
        SpecialistDispatchCollaborator,
        "warm_specialist_params",
        AsyncMock(side_effect=lambda params: params.setdefault("framework", "sglang")),
    )
    c.router.handle_delegate = AsyncMock()  # type: ignore[assignment]
    c.router.handle_intent = AsyncMock()  # type: ignore[assignment]
    return c


def _dispatched(coord) -> tuple[dict[str, Any], str]:
    coord.router.handle_delegate.assert_awaited_once()
    coord.router.handle_intent.assert_not_awaited()
    src, intent = coord.router.handle_delegate.call_args.args
    assert src == "orchestration"
    return intent.payload["params"], intent.payload["idempotency_key"]


_BASE_PARAMS = {
    "domain": "custom_specialist",
    "scope": "domain",
    "source": "coordinator_internal",
    "reason": "custom_specialist",
    "framework": "sglang",
}


async def test_dispatch_targets_the_most_actionable_gap(custom_coord):
    state = custom_coord.shared_state
    state.upsert_gap({"canonical_id": "gap.low", "severity": "low"})
    state.upsert_gap({"canonical_id": "gap.high", "severity": "high"})

    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    params, key = _dispatched(custom_coord)
    assert params == {**_BASE_PARAMS, "gap_canonical_id": "gap.high"}
    assert key == _custom_key()
    custom_coord.bus.record_observation.assert_not_awaited()


async def test_dispatch_without_gaps_uses_the_session_gap(custom_coord):
    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    params, _ = _dispatched(custom_coord)
    assert params == {
        **_BASE_PARAMS,
        "gap_canonical_id": "gap.custom.session",
        "gap_symptom": "No profiled gap; follow the operator-defined focus.",
        "gap_layer": "operator-defined",
    }


async def test_dispatch_key_names_the_cycle_and_the_definition(custom_coord):
    custom_coord.shared_state.macro_cycle = 2
    custom_coord.shared_state.custom_specialist_prompt = "other focus"

    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    _, key = _dispatched(custom_coord)
    assert key == _custom_key(2, "other focus")


@pytest.mark.parametrize("state", ["queued", "running", "succeeded", "failed", "cancelled"])
async def test_one_dispatch_per_cycle_whatever_became_of_it(custom_coord, state):
    custom_coord.tasks.find_by_idempotency_key.return_value = SimpleNamespace(state=state)

    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    custom_coord.tasks.find_by_idempotency_key.assert_awaited_once_with(_custom_key())
    custom_coord.router.handle_delegate.assert_not_awaited()


async def test_a_task_that_only_borrows_the_custom_tag_does_not_hold_the_dispatch_back(custom_coord):
    custom_coord.tasks.queued.return_value = [
        SimpleNamespace(kind="specialist", params={"domain": "serving_specialist", "tags": ["custom_specialist"]})
    ]

    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    _, key = _dispatched(custom_coord)
    assert key == _custom_key()


@pytest.mark.parametrize(("prompt", "phase"), [("", "FRAMEWORK_AGENT"), (_PROMPT, "ENABLEMENT")])
async def test_dispatch_is_a_no_op(custom_coord, prompt, phase):
    custom_coord.shared_state.custom_specialist_prompt = prompt
    custom_coord.shared_state.phase = phase

    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    custom_coord.router.handle_delegate.assert_not_awaited()


async def test_dispatch_falls_back_to_research_when_patches_are_impossible(custom_coord, monkeypatch):
    import hyperloom.orchestrator.specialists.runner as runner_mod
    from hyperloom.orchestrator.specialists.runner import NO_GIT_FRAMEWORK_SOURCE_ROOT

    # An installed or env-pointed framework tree on the host must not satisfy the preflight.
    monkeypatch.setattr(runner_mod, "resolve_framework_tree", lambda _framework: "")
    custom_coord.shared_state.framework_repo_path = ""

    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    params, _ = _dispatched(custom_coord)
    assert params["mode"] == "research"
    custom_coord.bus.record_observation.assert_awaited_once_with(
        "coordinator",
        "observation",
        {"kind": "custom_specialist_research_only", "reason": NO_GIT_FRAMEWORK_SOURCE_ROOT},
    )


async def test_dispatch_enqueues_one_task_per_cycle_through_the_real_router(tmp_path):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    state.max_minutes = 30
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)
    dispatch = coord.specialist_dispatch

    await dispatch.ensure_custom_specialist()
    await dispatch.ensure_custom_specialist()
    first = [t.idempotency_key for t in await coord.tasks.queued()]
    coord.shared_state.macro_cycle = 1
    await dispatch.ensure_custom_specialist()

    queued = await coord.tasks.queued()
    assert first == [_custom_key()]
    assert [(t.idempotency_key, t.params["domain"], t.params["mode"]) for t in queued] == [
        (_custom_key(0), "custom_specialist", "research"),
        (_custom_key(1), "custom_specialist", "research"),
    ]


@pytest.mark.parametrize("only", [False, True])
async def test_the_coordinator_custom_task_passes_the_dispatch_time_gate_and_runs(tmp_path, only):
    state = SharedState(
        custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION, custom_specialist_only=only
    )
    state.max_minutes = 30
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)
    ran: list[str] = []

    async def _specialist(ctx) -> dict:
        ran.append(str(ctx.task.params.get("domain")))
        return {"runner_status": "succeeded"}

    coord.sub.register_executor("specialist", _specialist)
    await coord.specialist_dispatch.ensure_custom_specialist()
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await asyncio.wait_for(coord.dispatcher.wait_for_running_work(timeout=5), timeout=5)

    assert ran == ["custom_specialist"]


async def test_a_refused_custom_dispatch_is_requested_once_per_cycle(tmp_path, monkeypatch):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    # Below the research-mode base, so admission refuses it.
    state.max_minutes = 5
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)
    denials: list[tuple[int, str]] = []
    record = coord.router.record_policy_denied

    async def _record(source, intent, denied, **kwargs):
        denials.append((coord.shared_state.macro_cycle, denied.rule))
        await record(source, intent, denied, **kwargs)

    monkeypatch.setattr(coord.router, "record_policy_denied", _record)
    for cycle in (0, 0, 0, 1, 1):
        coord.shared_state.macro_cycle = cycle
        await coord.specialist_dispatch.ensure_custom_specialist()

    assert [cycle for cycle, _ in denials] == [0, 1]
    assert await coord.tasks.queued() == []
    assert coord.specialist_dispatch.refused_custom_key == _custom_key(1)


async def test_stalled_domain_force_ignores_a_custom_hinted_gap_when_not_configured(custom_coord):
    state = custom_coord.shared_state
    state.custom_specialist_prompt = ""
    state.upsert_gap({"canonical_id": "gap.hint", "domain_hint": "custom", "severity": "high"})
    state.upsert_gap({"canonical_id": "gap.serving", "domain_hint": "serving_specialist", "severity": "low"})
    for _ in range(50):
        state.bump_domain_round_counters()

    await custom_coord.specialist_dispatch.maybe_force_stalled_domain_specialist()

    custom_coord.router.handle_intent.assert_awaited_once()
    params = custom_coord.router.handle_intent.call_args.args[1].payload["params"]
    assert (params["domain"], params["tags"], params["gap_canonical_id"]) == (
        "serving_specialist",
        ["framework"],
        "gap.serving",
    )


# Dispatch order
def _task(task_id: str, kind: str, **params: Any) -> SimpleNamespace:
    return SimpleNamespace(task_id=task_id, kind=kind, params=params)


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("FRAMEWORK_AGENT", ["custom", "operator-explore", "patch", "serving", "tag-only", "explore"]),
        ("SWEEP", ["custom", "operator-explore", "serving", "tag-only", "explore", "patch"]),
    ],
)
def test_operator_work_dispatches_first_in_every_phase(phase, expected):
    queued = [
        _task("serving", "specialist", domain="serving_specialist"),
        _task("tag-only", "specialist", domain="serving_specialist", tags=["custom_specialist"]),
        _task("explore", "explore", grid=[]),
        _task("patch", "integrate_patch"),
        _task("custom", "specialist", domain="custom_specialist"),
        _task("operator-explore", "explore", grid=[], operator_priority=True),
    ]

    assert [t.task_id for t in in_dispatch_order(queued, phase)] == expected


# Bench of custom proposals
class _BenchTasks:
    def __init__(self, queued=(), running=()) -> None:
        self._queued, self._running = list(queued), list(running)
        self.created: list[dict[str, Any]] = []
        self.known: dict[str, Any] = {}

    async def queued(self):
        return self._queued

    async def running(self):
        return self._running

    async def create_or_return_existing(self, spec):
        existing = any(c["idempotency_key"] == spec.idempotency_key for c in self.created)
        self.created.append(vars(spec))
        return SimpleNamespace(task_id=f"t-{len(self.created)}"), existing

    async def find_by_idempotency_key(self, key):
        if key in self.known:
            return self.known[key]
        return next((SimpleNamespace(state="succeeded") for c in self.created if c["idempotency_key"] == key), None)


def _bench_round(task_id: str, domain: str, *names: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "domain": domain,
        "gap_canonical_id": "gap.s1",
        "cycle": 0,
        "proposal_set": [{"name": name, "extra_args": f"--{name}", "reason": "why"} for name in names],
    }


def _framework(*, rounds, queued=(), running=(), phase_done=False):
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    state.phase = "FRAMEWORK_AGENT"
    state.baseline_tput = 100.0
    state.baseline_config_path = "/cfg.yaml"
    state.specialist_rounds = list(rounds)
    state.framework_agent_phase_done = phase_done
    dispatcher = SimpleNamespace(
        admission_frozen=False,
        dispatch_paused_for_phase_budget=lambda: False,
        registry_lanes_ttl=lambda kind: (["benchmark_lane"], 1800),
    )
    coord = SimpleNamespace(
        shared_state=state,
        tasks=_BenchTasks(queued, running),
        dispatcher=dispatcher,
        state=SimpleNamespace(pending_proposals={}),
        specialist_dispatch=SimpleNamespace(refused_custom_key=""),
    )
    coord.phase_framework = FrameworkPhase(coord)
    return coord.phase_framework


_ROUNDS = [
    _bench_round("spec-serving", "serving_specialist", "serving-a"),
    _bench_round("spec-custom", "custom_specialist", "idx-4", "idx-2"),
]


@pytest.mark.parametrize(
    ("queued", "phase_done"),
    [([], False), ([_task("orch", "explore", grid=[])], False), ([], True)],
)
async def test_custom_proposals_are_benched_alone_ahead_of_every_other_gate(queued, phase_done):
    fp = _framework(rounds=_ROUNDS, queued=queued, phase_done=phase_done)

    await fp._maybe_bench_untested_proposals()

    (call,) = fp._coord.tasks.created
    assert [v["name"] for v in call["params"]["grid"]] == ["idx-4", "idx-2"]
    assert call["params"]["operator_priority"] is True
    assert call["idempotency_key"].startswith("auto-explore-custom-c0-")


async def test_a_queued_custom_bench_holds_back_another_one_and_the_general_bench():
    fp = _framework(rounds=_ROUNDS, queued=[_task("bench", "explore", grid=[], operator_priority=True)])

    await fp._maybe_bench_untested_proposals()

    assert fp._coord.tasks.created == []


async def test_the_general_bench_leaves_custom_proposals_out():
    rounds = [_bench_round("spec-serving", "serving_specialist", "serving-a")]
    fp = _framework(rounds=rounds)

    await fp._maybe_bench_untested_proposals()

    (call,) = fp._coord.tasks.created
    assert [v["name"] for v in call["params"]["grid"]] == ["serving-a"]
    assert "operator_priority" not in call["params"]


@pytest.mark.parametrize(
    ("rounds", "queued", "expected"),
    [
        (_ROUNDS, [], True),
        ([], [_task("custom", "specialist", domain="custom_specialist")], True),
        ([], [_task("bench", "explore", grid=[], operator_priority=True)], True),
        ([_ROUNDS[0]], [_task("tag-only", "specialist", tags=["custom_specialist"])], False),
    ],
)
async def test_operator_work_pending(rounds, queued, expected):
    fp = _framework(rounds=rounds, queued=queued)

    assert await fp.operator_work_pending() is expected


# FRAMEWORK exit waits for operator work
@pytest.mark.parametrize(
    ("next_phase", "held"),
    [
        (("SWEEP", "optimize_no_more_leverage", {"kernel_skipped": True}), True),
        (("KERNEL_AGENT", "optimize_no_more_leverage", {}), True),
        (("SWEEP", "optimize_phase_budget_exhausted", {"kernel_skipped": True}), False),
        (("CLOSE", "time_exhausted", {"terminal": True}), False),
    ],
)
def test_only_a_plateau_exit_from_framework_waits_for_operator_work(next_phase, held):
    from hyperloom.orchestrator.phases.machine import is_leverage_exit

    assert is_leverage_exit(next_phase) is held


@pytest.mark.parametrize(
    ("hint", "elapsed_min", "expected"),
    [("skip_to_sweep", 0, ("FRAMEWORK_AGENT", ["custom_specialist"])), ("", 600, ("KERNEL_AGENT", ["kernel_agent"]))],
)
async def test_framework_holds_a_leverage_exit_for_queued_custom_work_but_not_a_budget_exit(
    tmp_path, hint, elapsed_min, expected
):
    import time

    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    state.max_minutes = 120
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)
    state = coord.shared_state
    await coord.specialist_dispatch.ensure_custom_specialist()
    state.pending_escalate_hint = hint
    state.phase_started_unix = time.time() - elapsed_min * 60

    await coord.phase_machine.advance_phase_if_needed()

    assert (state.phase, [t.params.get("domain", t.kind) for t in await coord.tasks.queued()]) == expected


# HYPERLOOM_CUSTOM_SPECIALIST_ONLY
@pytest.fixture
def launch(tmp_path, monkeypatch):
    """Run ``main(["optimize", ...])`` up to ``_preflight``; return its exit code and the only mode it settled."""
    import os

    reached: list[bool] = []

    def _preflight(args):
        reached.append(args.custom_specialist_only)
        raise SystemExit(99)

    monkeypatch.setattr("hyperloom.inference_optimizer.cli._preflight", _preflight)
    prompt_file = tmp_path / "p.md"
    prompt_file.write_text(_PROMPT, encoding="utf-8")
    pair = ["--custom-specialist-prompt-file", str(prompt_file), "--custom-specialist-description", _DESCRIPTION]
    saved = dict(os.environ)
    os.environ.pop("HYPERLOOM_CUSTOM_SPECIALIST_ONLY", None)

    def _run(*argv: str, with_pair: bool = True) -> tuple[int, list[bool]]:
        with pytest.raises(SystemExit) as exc:
            main(["optimize", *(pair if with_pair else []), *argv])
        return exc.value.code, reached

    yield _run
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize(
    ("argv", "exported", "expected"),
    [
        (["--extra-env", "HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1"], None, (99, [True])),
        (["--extra-env", "HYPERLOOM_CUSTOM_SPECIALIST_ONLY=0"], "1", (99, [False])),
        ([], "1", (99, [True])),
        ([], None, (99, [False])),
    ],
)
def test_only_mode_reaches_the_launch_by_export_or_by_extra_env_pin(launch, argv, exported, expected):
    import os

    if exported is not None:
        os.environ["HYPERLOOM_CUSTOM_SPECIALIST_ONLY"] = exported

    assert launch(*argv) == expected


@pytest.mark.parametrize(
    ("argv", "with_pair", "message"),
    [
        (["--custom-specialist-only"], True, "unrecognized arguments: --custom-specialist-only"),
        (
            ["--extra-env", "HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1"],
            False,
            "HYPERLOOM_CUSTOM_SPECIALIST_ONLY requires --custom-specialist-prompt-file and "
            "--custom-specialist-description",
        ),
        (
            ["--extra-env", "HYPERLOOM_CUSTOM_SPECIALIST_ONLY=maybe"],
            True,
            "HYPERLOOM_CUSTOM_SPECIALIST_ONLY='maybe' is not a boolean",
        ),
        (["--research-lane-capacity", "0"], True, "--research-lane-capacity 0 builds no specialist executor"),
    ],
)
def test_a_bad_custom_specialist_launch_exits_before_preflight(launch, capsys, argv, with_pair, message):
    assert launch(*argv, with_pair=with_pair) == (2, [])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args_kw", "state_kw", "expected"),
    [
        (
            {"custom_specialist_only": True},
            {},
            "HYPERLOOM_CUSTOM_SPECIALIST_ONLY requires --custom-specialist-prompt-file and "
            "--custom-specialist-description",
        ),
        ({"custom_specialist_only": True}, {"custom_specialist_prompt": _PROMPT}, ""),
        (
            {"custom_specialist_only": True, "custom_specialist_prompt": _PROMPT},
            {},
            "",
        ),
    ],
)
def test_only_mode_needs_a_custom_specialist(args_kw, state_kw, expected):
    assert _custom_specialist_conflict(_ns(**args_kw), SharedState(**state_kw)) == expected


@pytest.mark.parametrize(
    ("passed", "stored", "expected"), [(True, False, True), (False, True, True), (False, False, False)]
)
def test_resume_turns_only_mode_on_and_never_off(passed, stored, expected):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=stored)
    _apply_custom_specialist_resume(_ns(custom_specialist_prompt="", custom_specialist_only=passed), state)
    assert state.custom_specialist_only is expected


def _only_gate(*, phase: str = "FRAMEWORK_AGENT", only: bool = True) -> PolicyGate:
    gate = PolicyGate(role_registry=default_role_registry())
    gate.shared_state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=only)
    gate.shared_state.phase = phase
    return gate


_FRAMEWORK_SEARCH_INTENTS = [
    _delegate({"domain": "serving_specialist", "gap_canonical_id": "gap.x"}),
    Intent(type=IntentType.DELEGATE, payload={"action_name": "explore", "params": {"grid": []}}),
    Intent(type=IntentType.DELEGATE, payload={"action_name": "integrate_patch", "params": {"specialist_task_id": "s"}}),
    Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "explore", "params": {"grid": []}, "predicted_gain_pct": 1.0},
    ),
]


@pytest.mark.parametrize("intent", _FRAMEWORK_SEARCH_INTENTS)
def test_only_mode_gate_denies_every_framework_search_orchestration_emits(intent):
    with pytest.raises(PolicyDenied) as exc:
        _only_gate().validate_intent("orchestration", intent)
    assert exc.value.rule == "custom_specialist_only"


@pytest.mark.parametrize(("phase", "only"), [("FRAMEWORK_AGENT", False), ("SWEEP", True)])
@pytest.mark.parametrize("intent", _FRAMEWORK_SEARCH_INTENTS)
def test_the_framework_search_denial_is_scoped_to_only_mode_in_framework(intent, phase, only):
    try:
        _only_gate(phase=phase, only=only).validate_intent("orchestration", intent)
    except PolicyDenied as denied:
        assert denied.rule != "custom_specialist_only"


def test_only_mode_lets_the_custom_bench_row_through_the_dispatch_time_gate():
    _only_gate().validate_dispatched_task("explore", {"grid": [], "operator_priority": True})


def _recorder(calls: list[str]):
    def _rec(name: str):
        async def _call(*args, **kwargs):
            calls.append(name)

        return _call

    return _rec


@pytest.mark.parametrize(
    ("only", "expected"),
    [
        (True, ["custom", "bench", "finish"]),
        (False, ["framework", "bench", "scout", "recon", "stalled", "custom", "predictor"]),
    ],
)
async def test_only_mode_tick_runs_the_custom_work_alone(monkeypatch, only, expected):
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    calls: list[str] = []
    _rec = _recorder(calls)
    monkeypatch.setattr(FrameworkPhase, "_pump_framework_agent_phase", _rec("framework"))
    monkeypatch.setattr(FrameworkPhase, "_maybe_bench_untested_proposals", _rec("bench"))
    monkeypatch.setattr(FrameworkPhase, "_maybe_finish_custom_only_cycle", _rec("finish"))
    monkeypatch.setattr(FrameworkPhase, "_predictor_pump", lambda self: SimpleNamespace(step=_rec("predictor")))
    monkeypatch.setattr(FrameworkPhase, "_record_advisory_plateau", lambda self: None)
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=only)
    coord = SimpleNamespace(
        shared_state=state,
        phase_internal=SimpleNamespace(
            maybe_enqueue_explore_research_scout=_rec("scout"),
            maybe_enqueue_config_plateau_static_recon=_rec("recon"),
        ),
        specialist_dispatch=SimpleNamespace(
            maybe_force_stalled_domain_specialist=_rec("stalled"), ensure_custom_specialist=_rec("custom")
        ),
        record_exception=lambda **kwargs: pytest.fail(f"the tick raised: {kwargs}"),
    )

    await FrameworkPhase(coord).pump()

    assert calls == expected


@pytest.mark.parametrize(
    ("only", "expected"),
    [(True, ["reprofile", "custom"]), (False, ["reprofile", "custom", "framework", "predictor"])],
)
async def test_framework_entry_enqueues_the_custom_specialist_before_any_other_work(monkeypatch, only, expected):
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    calls: list[str] = []
    _rec = _recorder(calls)
    monkeypatch.setattr(FrameworkPhase, "_open_framework_timeline", lambda self: None)
    monkeypatch.setattr(FrameworkPhase, "_pump_framework_agent_phase", _rec("framework"))
    monkeypatch.setattr(FrameworkPhase, "_predictor_pump", lambda self: SimpleNamespace(settle=_rec("predictor")))
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=only)
    coord = SimpleNamespace(
        shared_state=state,
        phase_macro_cycle=SimpleNamespace(on_cycle_start_reprofile=_rec("reprofile")),
        specialist_dispatch=SimpleNamespace(ensure_custom_specialist=_rec("custom")),
    )

    await FrameworkPhase(coord).on_enter_framework(SimpleNamespace(from_phase="PRELUDE"))

    assert calls == expected


async def test_only_mode_benches_no_proposal_but_the_custom_ones():
    fp = _framework(rounds=[_bench_round("spec-serving", "serving_specialist", "serving-a")])
    fp.shared_state.custom_specialist_only = True

    await fp._maybe_bench_untested_proposals()

    assert fp._coord.tasks.created == []


def _only_cycle(*, custom_state, rounds=(), queued=(), pending=False, refused=""):
    from hyperloom.orchestrator.specialists.domains import custom_specialist_key

    fp = _framework(rounds=rounds, queued=queued)
    fp.shared_state.custom_specialist_only = True
    fp._coord.specialist_dispatch.refused_custom_key = refused
    if custom_state:
        fp._coord.tasks.known[custom_specialist_key(0, _PROMPT)] = SimpleNamespace(
            task_id="custom-task", state=custom_state
        )
    if pending:
        fp._coord.state.pending_proposals["p1"] = SimpleNamespace(
            action_name="integrate_patch", payload={"params": {"specialist_task_id": "custom-task"}}
        )
    return fp


_SILENT_CUSTOM_ROUND = {**_bench_round("custom-task", "custom_specialist"), "proposal_set": []}


@pytest.mark.parametrize(
    ("kw", "hint"),
    [
        ({"custom_state": "succeeded", "rounds": [_SILENT_CUSTOM_ROUND]}, "skip_to_sweep"),
        ({"custom_state": "failed"}, "skip_to_sweep"),
        ({"custom_state": None}, ""),
        ({"custom_state": None, "refused": _custom_key(0)}, "skip_to_sweep"),
        ({"custom_state": None, "refused": _custom_key(1)}, ""),
        ({"custom_state": "running"}, ""),
        ({"custom_state": "succeeded"}, ""),
        ({"custom_state": "succeeded", "rounds": [_bench_round("custom-task", "custom_specialist", "idx-4")]}, ""),
        (
            {
                "custom_state": "succeeded",
                "rounds": [_SILENT_CUSTOM_ROUND],
                "queued": [_task("patch", "integrate_patch")],
            },
            "",
        ),
        ({"custom_state": "succeeded", "rounds": [_SILENT_CUSTOM_ROUND], "pending": True}, ""),
    ],
)
async def test_only_mode_leaves_framework_once_the_custom_cycle_is_done(kw, hint):
    fp = _only_cycle(**kw)

    await fp._maybe_finish_custom_only_cycle()

    assert fp.shared_state.pending_escalate_hint == hint


@pytest.mark.parametrize(
    ("phase", "only", "expected"),
    [
        ("FRAMEWORK_AGENT", True, ["critic"]),
        ("FRAMEWORK_AGENT", False, ["orchestration", "critic"]),
        ("SWEEP", True, ["orchestration", "critic"]),
    ],
)
async def test_only_mode_framework_ticks_without_an_orchestration_turn(tmp_path, monkeypatch, phase, only, expected):
    import time

    coord = _coordinator(tmp_path, SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=only))
    coord.shared_state.phase = phase
    turns: list[str] = []

    async def _noop(*args, **kwargs):
        return None

    async def _turn(name, **kwargs):
        turns.append(name)

    monkeypatch.setattr(coord, "reactor_pass", _turn)
    monkeypatch.setattr(coord.reconciler, "run", _noop)
    monkeypatch.setattr(coord.phase_machine, "advance_phase_if_needed", _noop)
    monkeypatch.setattr(coord.dispatcher, "pump_dispatcher_once", _noop)
    monkeypatch.setattr(coord.enablement_lane, "pump_enablement_safely", _noop)
    coord.phase_machine.pump_table = {"CLOSE": _noop}
    coord._last_maintenance_ts = time.monotonic()

    await coord._tick_once()

    assert turns == expected


@pytest.mark.parametrize(("only", "expected"), [(True, 240 * 60), (False, 10 * 60)])
async def test_only_mode_custom_specialist_runs_to_the_phase_end_without_an_extension(tmp_path, only, expected):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=only)
    state.max_minutes = 30
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)

    await coord.specialist_dispatch.ensure_custom_specialist()

    (task,) = await coord.tasks.queued()
    assert coord.dispatcher._specialist_wall_budget_sec(params=task.params) == expected
