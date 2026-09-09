# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forge reads the environment contract Hyperloom publishes, not its own.

Forge does not ship next to Hyperloom any more, it ships inside it, and an
operator configuring a box should not have to learn a second vocabulary for the
same decision. These tests pin the ladder so a future call site cannot quietly
mint a third spelling.
"""

from __future__ import annotations

import pytest

from kernelforge.config import Config, resolve_agent_model, resolve_agent_reasoning_effort

_MODEL_VARS = (
    "FORGE_AGENT_MODEL",
    "FORGE_CLAUDE_MODEL",
    "CLAUDE_CONTEXT_WINDOW",
    "FORGE_CODEX_MODEL",
    "CLAUDE_MODEL",
    "CODEX_MODEL",
    "KERNEL_AGENTS_MODEL",
)
_EFFORT_VARS = ("FORGE_AGENT_REASONING_EFFORT", "HYPERLOOM_REASONING_EFFORT")


@pytest.fixture
def clean_env(monkeypatch):
    """Start from an environment that names no model and no effort."""
    for name in (*_MODEL_VARS, *_EFFORT_VARS):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_nothing_configured_defers_to_the_provider(clean_env) -> None:
    """No env var set means no model, so the provider default stands."""
    assert resolve_agent_model("claude") == ""
    assert resolve_agent_model("codex") == ""


def test_orchestration_model_is_inherited(clean_env) -> None:
    """The value every Hyperloom component already reads reaches Forge."""
    clean_env.setenv("CLAUDE_MODEL", "claude-opus-5")
    clean_env.setenv("CODEX_MODEL", "gpt-5.6")
    assert resolve_agent_model("claude") == "claude-opus-5"
    assert resolve_agent_model("codex") == "gpt-5.6"
    # An unrecognised backend reads the Claude ladder, which is both what
    # ``auto`` resolves to here and what Hyperloom's resolver does.
    assert resolve_agent_model("auto") == "claude-opus-5"


def test_the_forge_private_model_vars_are_not_read(clean_env) -> None:
    """Forge has no model variable of its own -- none of the three is read.

    ``FORGE_CLAUDE_MODEL`` / ``FORGE_CODEX_MODEL`` were Forge naming its own
    settings back when it was a separate project, and ``FORGE_AGENT_MODEL`` was
    the provider-neutral one above them. Inside Hyperloom each is a second
    spelling of ``CLAUDE_MODEL`` / ``CODEX_MODEL``, and a second spelling of one
    setting is only ever a second place for a box to be misconfigured. The
    Hyperloom-side resolver reads only the platform pair, so Forge reading more
    than that is the two ladders disagreeing.
    """
    clean_env.setenv("CLAUDE_MODEL", "claude-opus-5")
    clean_env.setenv("CODEX_MODEL", "gpt-5.6")
    clean_env.setenv("FORGE_AGENT_MODEL", "claude-opus-4-8")
    clean_env.setenv("FORGE_CLAUDE_MODEL", "claude-sonnet-5")
    clean_env.setenv("FORGE_CODEX_MODEL", "gpt-5.5")
    assert resolve_agent_model("claude") == "claude-opus-5"
    assert resolve_agent_model("codex") == "gpt-5.6"
    assert resolve_agent_model("auto") == "claude-opus-5"


def test_the_context_window_env_is_not_read(clean_env) -> None:
    """``CLAUDE_CONTEXT_WINDOW`` is gone; setting it changes nothing.

    It was the one rung this port added, and it turned out to name something
    Forge does not need: the window is only ever spelled as a suffix on the
    model id, this gateway rejects every bracketed id, and Forge has no
    compaction or token budget that would want the number for its own sake.
    """
    clean_env.setenv("CLAUDE_CONTEXT_WINDOW", "1m")
    config = Config.from_env(agent_backend="claude", workspace="/tmp")
    assert not hasattr(config, "agent_context_window")
    assert config.agent_runtime().model == "claude-opus-5"


def test_the_removed_alias_is_no_longer_read(clean_env) -> None:
    """``KERNEL_AGENTS_MODEL`` is gone, not merely deprecated.

    Nothing in this repository or in Hyperloom ever set it -- it was only ever
    read -- so keeping a spelling that no producer writes just gave the ladder a
    fourth rung to explain.
    """
    clean_env.setenv("KERNEL_AGENTS_MODEL", "claude-opus-4-5")
    assert resolve_agent_model("claude") == ""


def test_model_ladder_reaches_config(clean_env) -> None:
    """``Config.from_env`` uses the ladder for the backend it was given."""
    clean_env.setenv("CLAUDE_MODEL", "claude-sonnet-5")
    clean_env.setenv("CODEX_MODEL", "gpt-5.5")
    assert Config.from_env(agent_backend="claude", workspace="/tmp").agent_model == "claude-sonnet-5"
    assert Config.from_env(agent_backend="codex", workspace="/tmp").agent_model == "gpt-5.5"
    # An explicit override still outranks the environment entirely.
    override = Config.from_env(agent_backend="claude", agent_model="claude-opus-5", workspace="/tmp")
    assert override.agent_model == "claude-opus-5"


def test_effort_defaults_to_high(clean_env) -> None:
    """Unset means the campaign default, unchanged."""
    assert resolve_agent_reasoning_effort() == "high"


def test_hyperloom_effort_is_honoured(clean_env) -> None:
    """A box that names an effort once means it for Forge too."""
    clean_env.setenv("HYPERLOOM_REASONING_EFFORT", "medium")
    assert resolve_agent_reasoning_effort() == "medium"
    assert Config.from_env(agent_backend="claude", workspace="/tmp").agent_reasoning_effort == "medium"


def test_forge_effort_outranks_the_project_wide_one(clean_env) -> None:
    """Forge can still be turned up or down on its own."""
    clean_env.setenv("HYPERLOOM_REASONING_EFFORT", "medium")
    clean_env.setenv("FORGE_AGENT_REASONING_EFFORT", "low")
    assert resolve_agent_reasoning_effort() == "low"


@pytest.mark.parametrize("value", ["minimal", "none", "hgih"])
@pytest.mark.parametrize("name", _EFFORT_VARS)
def test_an_off_ladder_effort_is_refused_by_name(clean_env, name: str, value: str) -> None:
    """A campaign refuses to start rather than 400 hours in.

    ``HYPERLOOM_REASONING_EFFORT=minimal`` used to pass Hyperloom's own filter
    and then raise inside the Codex backend, mid-campaign. Both variables now
    answer for themselves, at startup, and the message names which one carried
    the bad value -- an operator sets both and would otherwise have to guess.
    """
    clean_env.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        resolve_agent_reasoning_effort()
    with pytest.raises(ValueError, match=name):
        Config.from_env(agent_backend="claude", workspace="/tmp")


def test_the_lower_variable_is_not_consulted_when_the_higher_one_is_valid(clean_env) -> None:
    """A stale project-wide value does not veto an explicit Forge one."""
    clean_env.setenv("HYPERLOOM_REASONING_EFFORT", "minimal")
    clean_env.setenv("FORGE_AGENT_REASONING_EFFORT", "low")
    assert resolve_agent_reasoning_effort() == "low"
