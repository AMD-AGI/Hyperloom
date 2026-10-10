# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Operator-defined ``custom_specialist`` type: prompts, gate, CLI, and the at-least-once dispatch."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hyperloom.inference_optimizer.cli import (
    _apply_custom_specialist_resume,
    _build_orchestration_prompt,
    _custom_specialist_conflict,
    _load_custom_specialist_args,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
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
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.specialists.runner import SpecialistRunner, SpecialistSubprocessConfig
from hyperloom.orchestrator.state.objective import build_objective
from hyperloom.orchestrator.state.shared_state import SharedState

_PROMPT = "Focus on RCCL all-reduce.\n<keep>```verbatim```</keep>"
_DESCRIPTION = "Tunes RCCL all-reduce for TP communication."


def _custom_domain():
    return replace(get_domain("custom_specialist"), description=_DESCRIPTION)


def _inputs(domain, **kw: Any) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        task_id="t",
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
        "KB anchor: custom.",
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


def test_builtin_with_custom_tag_renders_custom_focus_only_when_configured():
    serving = get_domain("serving_specialist")
    configured, _ = build_specialist_prompts(
        _inputs(serving, extra_focus_tags=("custom",), custom_specialist_prompt=_PROMPT)
    )
    unconfigured, _ = build_specialist_prompts(_inputs(serving, extra_focus_tags=("custom",)))
    untagged, _ = build_specialist_prompts(_inputs(serving))

    assert f"### Domain focus — custom_specialist\n\n{_PROMPT}\n" in configured
    assert unconfigured == untagged


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

    prep = await runner._prepare(
        RunnerContext(task=task, lease=None, extra={"shared_state": state}), prompt_inputs=None
    )

    assert prep.domain.description == _DESCRIPTION
    assert f"Description: {_DESCRIPTION}\n" in prep.system_prompt
    assert f"### Domain focus — custom_specialist\n\n{_PROMPT}\n" in prep.system_prompt


# Orchestration prompt
def _orch(description: str, phase: str = "FRAMEWORK_AGENT") -> str:
    return _build_orchestration_prompt(
        no_kernel=False,
        framework="sglang",
        objective=build_objective({"MAX_HOURS": 1}),
        max_minutes=60,
        phase=phase,
        custom_specialist_description=description,
    )


@pytest.mark.parametrize("phase", ["ENABLEMENT", "FRAMEWORK_AGENT"])
def test_orchestration_prompt_offers_custom_specialist_when_configured(phase):
    prompt = _orch(_DESCRIPTION, phase)
    emit = next(line for line in prompt.splitlines() if "delegate{action_name='specialist'" in line)
    guide = [line for line in prompt.splitlines() if "OPERATOR-DEFINED DOMAIN" in line]

    assert "|framework_rewrite_specialist|custom_specialist>" in emit
    assert guide == [f"    OPERATOR-DEFINED DOMAIN: custom_specialist — {_DESCRIPTION}"]


def test_orchestration_prompt_without_custom_specialist_keeps_the_builtin_domain_list():
    prompt = _orch("")
    emit = next(line for line in prompt.splitlines() if "delegate{action_name='specialist'" in line)
    domains = emit.split("domain=<one of ", 1)[1].split(">", 1)[0]

    assert domains == (
        "serving_specialist|kernel_switch_specialist|comm_specialist|compiler_specialist|system_specialist"
        "|candidate_discovery_specialist|research_scout_specialist|static_recon_specialist"
        "|framework_rewrite_specialist"
    )
    assert prompt.count("custom_specialist") == 0


# PolicyGate
def _delegate(params: dict[str, Any]) -> Intent:
    return Intent(type=IntentType.DELEGATE, payload={"action_name": "specialist", "params": params})


@pytest.mark.parametrize(
    "params",
    [
        {"domain": "custom_specialist", "gap_canonical_id": "gap.x"},
        {"tags": ["custom"], "gap_canonical_id": "gap.x"},
        {"scope": "freeform", "domain": "custom_specialist", "task_description": "do it"},
    ],
)
def test_gate_denies_custom_specialist_when_not_configured(params):
    gate = PolicyGate(role_registry=default_role_registry())
    gate.shared_state = SharedState()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate(params))
    assert exc.value.rule == "specialist_custom_not_configured"


def test_gate_allows_custom_specialist_when_configured():
    gate = PolicyGate(role_registry=default_role_registry())
    gate.shared_state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    gate.validate_intent("orchestration", _delegate({"domain": "custom_specialist", "gap_canonical_id": "gap.x"}))


# CLI
def _ns(**kw: Any) -> argparse.Namespace:
    base = {
        "custom_specialist_prompt_file": None,
        "custom_specialist_description": None,
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


def test_resume_with_a_new_definition_replaces_it_and_rearms_the_guarantee():
    state = SharedState(
        custom_specialist_prompt="old", custom_specialist_description="old desc", custom_specialist_dispatched=True
    )
    _apply_custom_specialist_resume(
        _ns(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION), state
    )
    assert (state.custom_specialist_prompt, state.custom_specialist_description) == (_PROMPT, _DESCRIPTION)
    assert state.custom_specialist_dispatched is False


@pytest.mark.parametrize("repassed", [True, False])
def test_resume_with_the_same_or_no_definition_keeps_the_guarantee_spent(repassed):
    state = SharedState(
        custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION, custom_specialist_dispatched=True
    )
    args = (
        _ns(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
        if repassed
        else _ns(custom_specialist_prompt="")
    )
    _apply_custom_specialist_resume(args, state)
    assert (state.custom_specialist_prompt, state.custom_specialist_dispatched) == (_PROMPT, True)


@pytest.mark.parametrize(
    ("overrides", "state_kw", "expected"),
    [
        ({}, {}, ""),
        (
            {"orch_prompt": "x"},
            {},
            "--orch-prompt replaces the Orchestration prompt, so custom_specialist would never be offered",
        ),
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
        (
            {"reset_state": True},
            {},
            "--reset-state wipes the custom specialist definition; reset first, then launch with the flags",
        ),
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


def test_conflicts_are_ignored_when_no_custom_specialist_is_configured():
    args = _ns(orch_prompt="x", no_framework_agent=True, research_lane_capacity=0, reset_state=True)
    assert _custom_specialist_conflict(args, SharedState()) == ""


def test_fresh_launch_conflict_exits_before_the_session_starts(tmp_path, monkeypatch):
    import hyperloom.inference_optimizer.cli as cli

    run_optimize = AsyncMock(return_value=0)
    monkeypatch.setattr(cli, "_run_optimize", run_optimize)
    prompt_file = tmp_path / "p.md"
    prompt_file.write_text(_PROMPT, encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "optimize",
                "--custom-specialist-prompt-file",
                str(prompt_file),
                "--custom-specialist-description",
                _DESCRIPTION,
                "--orch-prompt",
                "inline orchestration prompt",
            ]
        )

    assert exc.value.code == 2
    run_optimize.assert_not_awaited()


def test_custom_specialist_fields_roundtrip_and_default_on_old_state():
    state = SharedState(
        custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION, custom_specialist_dispatched=True
    )
    restored = SharedState.from_dict(state.to_dict())
    old = SharedState.from_dict({"session_id": "old"})

    assert (restored.custom_specialist_prompt, restored.custom_specialist_description) == (_PROMPT, _DESCRIPTION)
    assert restored.custom_specialist_dispatched is True
    assert (old.custom_specialist_prompt, old.custom_specialist_description, old.custom_specialist_dispatched) == (
        "",
        "",
        False,
    )


# At-least-once dispatch
@pytest.fixture
def guarantee_coord(tmp_path: Path, monkeypatch):
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
    )
    c.bus = SimpleNamespace(record_observation=AsyncMock())
    monkeypatch.setattr(
        SpecialistDispatchCollaborator,
        "warm_specialist_params",
        AsyncMock(side_effect=lambda params: params.setdefault("framework", "sglang")),
    )
    c.router.handle_intent = AsyncMock()  # type: ignore[assignment]
    return c


def _dispatched(coord) -> tuple[dict[str, Any], str]:
    coord.router.handle_intent.assert_awaited_once()
    src, intent = coord.router.handle_intent.call_args.args
    assert src == "orchestration"
    return intent.payload["params"], intent.payload["idempotency_key"]


_BASE_PARAMS = {
    "domain": "custom_specialist",
    "tags": ["custom"],
    "scope": "domain",
    "source": "coordinator_internal",
    "reason": "custom_specialist_guarantee",
    "framework": "sglang",
}


async def test_guarantee_dispatches_once_on_the_most_actionable_gap(guarantee_coord):
    state = guarantee_coord.shared_state
    state.upsert_gap({"canonical_id": "gap.low", "severity": "low"})
    state.upsert_gap({"canonical_id": "gap.high", "severity": "high"})

    await guarantee_coord.specialist_dispatch.maybe_ensure_custom_specialist()

    params, key = _dispatched(guarantee_coord)
    assert params == {**_BASE_PARAMS, "gap_canonical_id": "gap.high"}
    assert key == "custom-guarantee"
    guarantee_coord.bus.record_observation.assert_not_awaited()


async def test_guarantee_without_gaps_uses_the_session_gap(guarantee_coord):
    await guarantee_coord.specialist_dispatch.maybe_ensure_custom_specialist()

    params, _ = _dispatched(guarantee_coord)
    assert params == {
        **_BASE_PARAMS,
        "gap_canonical_id": "gap.custom.session",
        "gap_symptom": "No profiled gap; follow the operator-defined focus.",
        "gap_layer": "operator-defined",
    }


async def test_guarantee_key_is_cycle_scoped(guarantee_coord):
    guarantee_coord.shared_state.macro_cycle = 2

    await guarantee_coord.specialist_dispatch.maybe_ensure_custom_specialist()

    _, key = _dispatched(guarantee_coord)
    assert key == "custom-guarantee-c2"


async def test_a_denied_attempt_is_not_retried_until_the_next_cycle(guarantee_coord):
    # The mocked router creates no task, as an admission denial does.
    dispatch = guarantee_coord.specialist_dispatch
    await dispatch.maybe_ensure_custom_specialist()
    await dispatch.maybe_ensure_custom_specialist()
    guarantee_coord.shared_state.macro_cycle = 1
    await dispatch.maybe_ensure_custom_specialist()
    await dispatch.maybe_ensure_custom_specialist()

    keys = [call.args[1].payload["idempotency_key"] for call in guarantee_coord.router.handle_intent.await_args_list]
    assert keys == ["custom-guarantee", "custom-guarantee-c1"]


async def test_guarantee_skips_while_a_tag_only_custom_task_is_queued(guarantee_coord):
    guarantee_coord.tasks.queued.return_value = [SimpleNamespace(kind="specialist", params={"tags": ["custom"]})]

    await guarantee_coord.specialist_dispatch.maybe_ensure_custom_specialist()

    guarantee_coord.router.handle_intent.assert_not_awaited()


@pytest.mark.parametrize(
    ("state_kw", "phase"),
    [
        ({"custom_specialist_dispatched": True}, "FRAMEWORK_AGENT"),
        ({"custom_specialist_prompt": ""}, "FRAMEWORK_AGENT"),
        ({}, "ENABLEMENT"),
    ],
)
async def test_guarantee_is_a_no_op(guarantee_coord, state_kw, phase):
    state = guarantee_coord.shared_state
    for name, value in state_kw.items():
        setattr(state, name, value)
    state.phase = phase

    await guarantee_coord.specialist_dispatch.maybe_ensure_custom_specialist()

    guarantee_coord.router.handle_intent.assert_not_awaited()


@pytest.mark.parametrize("pruned", [True, False])
async def test_guarantee_falls_back_to_research_when_patches_are_impossible(guarantee_coord, pruned, monkeypatch):
    import hyperloom.orchestrator.specialists.runner as runner_mod

    # An installed or env-pointed framework tree on the host must not satisfy the preflight.
    monkeypatch.setattr(runner_mod, "resolve_framework_tree", lambda _framework: "")
    state = guarantee_coord.shared_state
    if pruned:
        state.add_pruned_family("source_patch")
        reason = "source_patch_pruned"
    else:
        state.framework_repo_path = ""
        from hyperloom.orchestrator.specialists.runner import NO_GIT_FRAMEWORK_SOURCE_ROOT

        reason = NO_GIT_FRAMEWORK_SOURCE_ROOT

    await guarantee_coord.specialist_dispatch.maybe_ensure_custom_specialist()

    params, _ = _dispatched(guarantee_coord)
    assert params["mode"] == "research"
    guarantee_coord.bus.record_observation.assert_awaited_once_with(
        "coordinator",
        "observation",
        {"kind": "custom_specialist_guarantee_research_only", "reason": reason},
    )


# Spawn marks the custom specialist as run
async def test_spawning_a_tag_only_custom_task_marks_it_dispatched(tmp_path):
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan

    state = SharedState(session_id="custom-spawn")
    state.max_minutes = 30
    state.save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    coord = Coordinator(
        session_dir=tmp_path,
        backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
        role_registry=default_role_registry(),
        knowledge_plane=None,
    )
    gate = asyncio.Event()

    async def _specialist(_ctx) -> dict:
        await gate.wait()
        return {"runner_status": "succeeded"}

    coord.sub.register_executor("specialist", _specialist)
    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={"tags": ["custom"], "gap_canonical_id": "gap.x"},
        idempotency_key="custom-spawn",
        requires_lanes=["research_lane"],
        lease_ttl_sec=600,
    )
    assert coord.shared_state.custom_specialist_dispatched is False

    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)

    assert coord.shared_state.custom_specialist_dispatched is True
    gate.set()
    await asyncio.wait_for(coord.dispatcher.wait_for_running_work(timeout=5), timeout=5)
