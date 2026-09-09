"""The provider-neutral run specification is an API other packages build against."""

from __future__ import annotations

from dataclasses import fields

from kernelforge.agent_backends.base import (
    AGENT_SAFETY_REJECTION_ATTR,
    AgentProviderError,
    AgentRunSpec,
    AgentToolPolicy,
)

#: Field order published before this branch added its flags. Positional callers
#: written against it must keep binding the same values to the same names.
_PUBLISHED_ORDER = (
    "system_prompt",
    "user_prompt",
    "cwd",
    "model",
    "writable",
    "timeout_sec",
    "reasoning_effort",
    "additional_directories",
    "target_files",
    "driver_script",
    "protected_globs",
    "allow_dirty_targets",
    "allow_untracked",
    "read_only_resume",
    "tool_policy",
    "hooks",
    "subagents",
    "mcp_servers",
    "provider_options",
)


def test_the_published_field_order_is_unchanged() -> None:
    """Keep every previously published field at the position it was published at."""
    names = [field.name for field in fields(AgentRunSpec)]

    assert names[: len(_PUBLISHED_ORDER)] == list(_PUBLISHED_ORDER)


def test_a_positional_caller_still_binds_its_tool_policy() -> None:
    """Bind a positionally supplied tool policy to tool_policy, not to a new flag."""
    policy = AgentToolPolicy(read=True, search=True, write=False, shell=False)

    spec = AgentRunSpec(
        "system",
        "user",
        "/tmp/workspace",
        "gpt-test",
        False,
        60,
        "high",
        ["/tmp/reference"],
        ["kernel.py"],
        "driver.py",
        ["*.json"],
        True,
        True,
        False,
        policy,
    )

    assert spec.tool_policy is policy
    assert spec.read_only_resume is False
    assert spec.allow_untracked is True


def test_the_safety_verdict_marker_is_declared_where_providers_can_find_it():
    """Publish the marker beside the provider base classes that must set it."""
    assert AGENT_SAFETY_REJECTION_ATTR == "agent_safety_rejection"
    assert "AGENT_SAFETY_REJECTION_ATTR" in (AgentProviderError.__doc__ or "")


def test_the_consumer_reads_the_published_marker():
    """Keep one spelling of the marker, so the two sides cannot drift apart."""
    from kernelforge.fusion import llm_failure

    assert llm_failure.AGENT_SAFETY_REJECTION_ATTR is AGENT_SAFETY_REJECTION_ATTR


def test_an_unmarked_provider_error_is_not_a_verdict():
    """Treat an unmarked error as retryable, which is the recoverable mistake."""
    from kernelforge.fusion.llm_failure import is_agent_safety_error

    assert is_agent_safety_error(AgentProviderError("something went wrong")) is False
