# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Coordinator prompt describes the transport the Coordinator actually has."""

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
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.prompts.prompt_builder import (
    TRANSPORT_STRUCTURED_OUTPUT,
    TRANSPORT_TOOLS,
    build_orchestration_prompt,
    default_enabled_actions,
    PromptAssets,
    PromptCycle,
    SessionContext,
)
from hyperloom.orchestrator.roles.agent_role import _ORCHESTRATION_INTENTS, default_role_registry
from hyperloom.orchestrator.roles.base import BackendError, LLMCallFailed, RetryPolicy
from hyperloom.orchestrator.roles.codex import CodexBackend
from hyperloom.orchestrator.mcp_tool_names import (
    CONTEXT_TOOL_NAMES,
    CONTEXT_TOOL_QUALIFIED_NAMES,
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    qualified_context_tool_name,
)
from hyperloom.orchestrator.roles.mcp_context_tools import CONTEXT_TOOL_SPECS
from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir, asset_system_prompts_dir


# The tool surface the system prompt actually documents, under the names the
# runtime registers them with: a bare MCP name is a tool nothing answers to.
_TOOL_SURFACE_TOKENS: tuple[str, ...] = (
    EMIT_INTENT_TOOL_QUALIFIED,
    "tool_use",
    "WebSearch",
    "WebFetch",
    qualified_context_tool_name("get_recent_outcomes"),
    qualified_context_tool_name("get_running_tasks"),
    qualified_context_tool_name("read_reference"),
)

# Anything a tool-less transport must never be told to call. The bare names
# cover both spellings, a qualified name containing the bare one.
_FORBIDDEN_WITHOUT_TOOLS: tuple[str, ...] = (
    EMIT_INTENT_TOOL_NAME,
    "tool_use",
    "WebSearch",
    "WebFetch",
    *CONTEXT_TOOL_NAMES,
)


def _prompt(transport: str, *, phase: str = "") -> str:
    return build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=default_enabled_actions(no_kernel=False),
        session=SessionContext(framework="sglang", objective_kind="time_only", max_minutes=60, transport=transport),
        cycle=PromptCycle(phase=phase),
        assets=PromptAssets(rules_fragment_path=asset_system_prompts_dir() / "orchestration.md"),
    )


# The prompt must not name a tool the transport does not mount.


@pytest.mark.parametrize(
    "phase", ["", "PRELUDE", "ENABLEMENT", "EXPLORE", "KERNEL_AGENT", "FRAMEWORK_AGENT", "SWEEP", "CLOSE"]
)
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
    assert set(CONTEXT_TOOL_QUALIFIED_NAMES) <= granted


def _assert_mcp_tools_named_as_registered(text: str, where: str) -> None:
    """Fail when ``text`` names an MCP tool by anything but its registered name."""
    pairs = [(EMIT_INTENT_TOOL_NAME, EMIT_INTENT_TOOL_QUALIFIED)]
    pairs += [(name, qualified_context_tool_name(name)) for name in CONTEXT_TOOL_NAMES]
    for bare, qualified in pairs:
        # Every mention of the bare name is part of the qualified one.
        assert text.count(bare) == text.count(qualified), f"{bare} in {where}"


def test_tool_prompt_names_every_mcp_tool_under_its_registered_name() -> None:
    """A bare MCP name in the prompt advertises a tool the agent cannot call."""
    _assert_mcp_tools_named_as_registered(_prompt(TRANSPORT_TOOLS), "the tools prompt")


def test_tool_descriptions_name_every_mcp_tool_under_its_registered_name() -> None:
    """The SDK shows these beside the allow-list; a bare name there misroutes the call."""
    for tool_name, description, _schema, _method in CONTEXT_TOOL_SPECS:
        _assert_mcp_tools_named_as_registered(description, f"the {tool_name} description")


def test_reference_docs_name_every_mcp_tool_under_its_registered_name() -> None:
    """read_reference serves these verbatim, on the paths taken after a failure."""
    docs = sorted(asset_prompt_references_dir().glob("*.md"))
    assert docs, "no reference document to scan; read_reference serves this directory"
    for doc in docs:
        _assert_mcp_tools_named_as_registered(doc.read_text(encoding="utf-8"), doc.name)


def test_transport_defaults_to_the_tool_surface() -> None:
    """An unspecified transport keeps the historical Claude rendering."""
    assert _prompt(TRANSPORT_TOOLS) == build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=default_enabled_actions(no_kernel=False),
        session=SessionContext(framework="sglang", objective_kind="time_only", max_minutes=60),
        assets=PromptAssets(rules_fragment_path=asset_system_prompts_dir() / "orchestration.md"),
    )


def test_backends_declare_their_transport(tmp_path: Path) -> None:
    """The prompt builder picks its rendering from the backend, not the role."""
    from hyperloom.orchestrator.roles.claude import ClaudeBackend

    codex = CodexBackend(allowed_intents=_ORCHESTRATION_INTENTS, cwd=tmp_path / "codex")
    assert codex.transport == TRANSPORT_STRUCTURED_OUTPUT
    assert ClaudeBackend.transport == TRANSPORT_TOOLS


# The delta note must not point at tools that were never mounted.


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
