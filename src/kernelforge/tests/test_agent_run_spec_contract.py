"""The provider-neutral run specification is an API other packages build against.

``AgentRunSpec`` is constructed by every stage in this repository and by any
third-party backend registered through the ``kernelforge.agent_providers``
entry-point group. Its field order is therefore part of the contract: a new flag
inserted in the middle silently re-binds every positional argument after it, and
a caller passing a tool policy positionally would hand it to the new flag
instead. New fields go at the end.
"""

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
    """Publish the marker beside the provider base classes that must set it.

    Consumers stopped recognizing a workspace-safety verdict by matching
    ``*SafetyError`` on the class name, because a backend raises that same class
    for its own bookkeeping failures too -- a snapshot it could not read, a Git
    query that timed out -- and matching by name made a stalled call abandon a
    recipe. The verdict is now marked with an attribute instead. A provider
    outside this repository has no way to learn that from the consumer package,
    so the name lives with the contract it belongs to.
    """
    assert AGENT_SAFETY_REJECTION_ATTR == "agent_safety_rejection"
    assert "AGENT_SAFETY_REJECTION_ATTR" in (AgentProviderError.__doc__ or "")


def test_the_consumer_reads_the_published_marker():
    """Keep one spelling of the marker, so the two sides cannot drift apart."""
    from kernelforge.fusion import llm_failure

    assert llm_failure.AGENT_SAFETY_REJECTION_ATTR is AGENT_SAFETY_REJECTION_ATTR


def test_an_unmarked_provider_error_is_not_a_verdict():
    """Treat an unmarked error as retryable, which is the recoverable mistake.

    Retrying a genuine rejection costs one more attempt; abandoning a recipe over
    a transient failure discards work that would have finished.
    """
    from kernelforge.fusion.llm_failure import is_agent_safety_error

    assert is_agent_safety_error(AgentProviderError("something went wrong")) is False
