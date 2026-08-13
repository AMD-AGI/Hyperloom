# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Coordinator prompt describes the transport the Coordinator actually has.

``orchestration.md`` was written for the Claude tool surface and rendered
unconditionally: it told every OpenAI-only Coordinator turn that "every reply
MUST include at least one `emit_intent` tool_use block", and pointed it at
`get_recent_outcomes`, `run_action_now`, `WebSearch` and `WebFetch`. The Codex
session exposes none of those. The per-tick delta note did the same, telling
the model to "pull anything else you need with the read-only context tools"
even though ``CodexBackend`` has no ``set_context_provider`` and the
Coordinator's attach call was a no-op for it.

Retry parity is the third half of the same story: the Claude path retries
transient gateway failures with bounded backoff, while the Codex path failed
the whole tick on the first one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.codex_session import (
    CodexSession,
    CodexSessionError,
    CodexSessionResult,
    CodexSessionUnavailableError,
)
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.prompts.prompt_builder import (
    TRANSPORT_STRUCTURED_OUTPUT,
    TRANSPORT_TOOLS,
    build_orchestration_prompt,
    default_enabled_actions,
)
from hyperloom.orchestrator.roles import MockBackend, MockTurn, ScriptedPlan
from hyperloom.orchestrator.roles.agent_role import _ORCHESTRATION_INTENTS, default_role_registry
from hyperloom.orchestrator.roles.base import BackendError, LLMCallFailed, RetryPolicy
from hyperloom.orchestrator.roles.codex import CodexBackend
from hyperloom.orchestrator.roles.mcp_context_tools import CONTEXT_TOOL_NAMES
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir


# The tool surface the system prompt actually documents. ``Read`` is matched in
# backticks only: the prompt legitimately uses the English word.
_TOOL_SURFACE_TOKENS: tuple[str, ...] = (
    "emit_intent",
    "tool_use",
    "WebSearch",
    "WebFetch",
    "get_recent_outcomes",
    "get_running_tasks",
    "run_action_now",
    "read_reference",
)

# Anything a tool-less transport must never be told to call.
_FORBIDDEN_WITHOUT_TOOLS: tuple[str, ...] = (*_TOOL_SURFACE_TOKENS, *CONTEXT_TOOL_NAMES)


def _prompt(transport: str, *, phase: str = "") -> str:
    return build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        objective_kind="time_only",
        max_minutes=60,
        phase=phase,
        transport=transport,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
    )


# ---------------------------------------------------------------------------
# The prompt must not name a tool the transport does not mount.


@pytest.mark.parametrize("phase", ["", "PRELUDE", "EXPLORE", "KERNEL_AGENT", "FRAMEWORK_AGENT", "SWEEP", "CLOSE"])
def test_structured_output_prompt_names_no_claude_tool(phase: str) -> None:
    """The Codex Coordinator has no MCP tools; naming them wastes every turn."""
    prompt = _prompt(TRANSPORT_STRUCTURED_OUTPUT, phase=phase)
    for token in _FORBIDDEN_WITHOUT_TOOLS:
        assert token not in prompt, f"{token} in phase {phase or 'ALL'}"
    assert "`Read`" not in prompt


def test_structured_output_prompt_states_its_own_transport() -> None:
    """Replacing the tool contract with nothing would be worse than leaving it."""
    prompt = " ".join(_prompt(TRANSPORT_STRUCTURED_OUTPUT).split())
    assert "matching the enforced output schema" in prompt
    assert "at least one intent" in prompt


def test_tool_prompt_keeps_the_claude_tool_surface() -> None:
    """The Anthropic path must not lose its tool contract to the scoping."""
    prompt = _prompt(TRANSPORT_TOOLS)
    for token in _TOOL_SURFACE_TOKENS:
        assert token in prompt, token


def test_every_tool_the_prompt_names_is_one_policy_gate_grants() -> None:
    """The prompt and PolicyGate must describe the same tool surface."""
    granted = set(PolicyGate(role_registry=default_role_registry()).allowed_tools_for_agent("orchestration"))
    prompt = _prompt(TRANSPORT_TOOLS)
    named = {token for token in _TOOL_SURFACE_TOKENS if token != "tool_use" and token in prompt}
    assert named <= granted
    assert set(CONTEXT_TOOL_NAMES) <= granted


def test_transport_defaults_to_the_tool_surface() -> None:
    """An unspecified transport keeps the historical Claude rendering."""
    assert _prompt(TRANSPORT_TOOLS) == build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        objective_kind="time_only",
        max_minutes=60,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
    )


def test_backends_declare_their_transport(tmp_path: Path) -> None:
    """The prompt builder picks its rendering from the backend, not the role."""
    from hyperloom.orchestrator.roles.claude import ClaudeBackend

    codex = CodexBackend(allowed_intents=_ORCHESTRATION_INTENTS, cwd=tmp_path / "codex")
    assert codex.transport == TRANSPORT_STRUCTURED_OUTPUT
    assert ClaudeBackend.transport == TRANSPORT_TOOLS


# ---------------------------------------------------------------------------
# The delta note must not point at tools that were never mounted.


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _coordinator(session_dir: Path, orchestration: Any) -> Coordinator:
    plan = ScriptedPlan(turns=[MockTurn(intents=[])], default_intent=_heartbeat())
    return Coordinator(
        session_dir,
        backends={
            "orchestration": orchestration,
            "critic": MockBackend(plan, name="critic"),
            "robustness": MockBackend(plan, name="robustness"),
        },
    )


def _stub_session(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _start(session: CodexSession) -> None:
        return None

    async def _turn(session: CodexSession, prompt: str, **_kwargs: Any) -> CodexSessionResult:
        return CodexSessionResult(
            text='{"intents": [{"intent_type": "send_message", "payload": "{\\"topic\\": \\"h\\"}"}]}'
        )

    monkeypatch.setattr(CodexSession, "start", _start)
    monkeypatch.setattr(CodexSession, "turn", _turn)


async def test_delta_turn_without_context_tools_names_none(
    session_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telling a Codex turn to call a context tool is an instruction it cannot follow."""
    _stub_session(monkeypatch)
    backend = CodexBackend(allowed_intents=_ORCHESTRATION_INTENTS, cwd=tmp_path / "codex")
    coord = _coordinator(session_dir, backend)

    await backend.run(await coord._compose_prompt("orchestration"))
    coord._orchestration_seeded = True
    delta = await coord._compose_prompt("orchestration")

    assert "=== Context" in delta
    for tool in CONTEXT_TOOL_NAMES:
        assert tool not in delta


async def test_delta_turn_with_context_tools_still_names_them(session_dir: Path) -> None:
    """The Claude path's pull instruction must survive the conditional."""

    class _ToolBackend(MockBackend):
        conversational = True
        context_tools_mounted = True

    backend = _ToolBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="orchestration")
    coord = _coordinator(session_dir, backend)

    await coord._compose_prompt("orchestration")
    coord._orchestration_seeded = True
    delta = await coord._compose_prompt("orchestration")

    for tool in CONTEXT_TOOL_NAMES:
        assert tool in delta


# ---------------------------------------------------------------------------
# Retry parity with the Claude path.


def _retrying_backend(tmp_path: Path) -> CodexBackend:
    return CodexBackend(
        allowed_intents=_ORCHESTRATION_INTENTS,
        cwd=tmp_path / "codex",
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter_s=0.0),
    )


async def test_a_transient_gateway_failure_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single 502 must not cost the Coordinator its whole tick."""
    attempts = {"n": 0}

    async def _start(session: CodexSession) -> None:
        return None

    async def _turn(session: CodexSession, prompt: str, **_kwargs: Any) -> CodexSessionResult:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise CodexSessionError("Codex SDK turn failed: gateway 502")
        return CodexSessionResult(
            text='{"intents": [{"intent_type": "send_message", "payload": "{\\"topic\\": \\"h\\"}"}]}'
        )

    monkeypatch.setattr(CodexSession, "start", _start)
    monkeypatch.setattr(CodexSession, "turn", _turn)

    result = await _retrying_backend(tmp_path).run("tick")

    assert attempts["n"] == 3
    assert len(result.intents) == 1


async def test_exhausted_retries_surface_as_an_llm_call_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely dead gateway still fails, and only after the budget is spent."""
    attempts = {"n": 0}

    async def _start(session: CodexSession) -> None:
        return None

    async def _turn(session: CodexSession, prompt: str, **_kwargs: Any) -> CodexSessionResult:
        attempts["n"] += 1
        raise CodexSessionError("Codex SDK turn failed: gateway 502")

    monkeypatch.setattr(CodexSession, "start", _start)
    monkeypatch.setattr(CodexSession, "turn", _turn)

    with pytest.raises(LLMCallFailed, match="gateway 502"):
        await _retrying_backend(tmp_path).run("tick")

    assert attempts["n"] == 3


async def test_a_configuration_fault_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying an unusable sandbox policy only delays the operator's error."""
    attempts = {"n": 0}

    async def _start(session: CodexSession) -> None:
        attempts["n"] += 1
        raise CodexSessionUnavailableError("HYPERLOOM_CODEX_SANDBOX_MODE is unusable")

    monkeypatch.setattr(CodexSession, "start", _start)

    with pytest.raises(BackendError, match="SANDBOX_MODE"):
        await _retrying_backend(tmp_path).run("tick")

    assert attempts["n"] == 1


def test_both_providers_read_the_same_retry_knobs() -> None:
    """Parity means one policy, not a second set of env vars to discover."""
    from hyperloom.orchestrator.roles.claude import ClaudeBackend

    assert (
        CodexBackend.__dataclass_fields__["retry_policy"].default_factory
        == ClaudeBackend.__dataclass_fields__["retry_policy"].default_factory
        == RetryPolicy.from_env
    )
