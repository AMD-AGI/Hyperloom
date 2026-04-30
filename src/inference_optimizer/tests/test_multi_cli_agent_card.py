"""Tests for orchestrator/multi_cli/agent_card.py — discovery + schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.agents import agents_root
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.multi_cli.agent_card import (
    AgentCard,
    AgentCardError,
    DEFAULT_INBOX,
    DEFAULT_OUTBOX,
    DEFAULT_PROMPT,
    discover_agent_cards,
    load_agent_card,
)


# ---------------------------------------------------------------------------
# Bundled cards (the four canonical roles)
# ---------------------------------------------------------------------------
def test_discovers_four_bundled_cards():
    """v0.4 MVP roster: executor / critic / triage / kernel (all Claude)."""
    cards = discover_agent_cards(agents_root())
    assert set(cards.keys()) == {
        "executor", "critic", "triage", "kernel",
    }


def test_kernel_card_has_expected_fields():
    cards = discover_agent_cards(agents_root())
    k = cards["kernel"]
    assert k.role == "kernel"
    assert k.backend == "claude"
    assert k.enabled is True
    # Plan A — kernel agent active in guided + marathon (NOT quick).
    assert ExecutionMode.GUIDED_KERNEL_OPT in k.allowed_modes
    assert ExecutionMode.MARATHON_MULTI_AGENT in k.allowed_modes
    assert ExecutionMode.QUICK_PARAM_SWEEP not in k.allowed_modes
    # Capabilities reflect responder role only.
    assert "response" in k.capabilities
    assert "delegate" not in k.capabilities
    assert "request" not in k.capabilities
    # Long Ray jobs need bumped backoff.
    assert k.restart_policy.backoff_seconds >= 30


def test_executor_card_has_expected_fields():
    cards = discover_agent_cards(agents_root())
    ex = cards["executor"]
    assert ex.role == "executor"
    assert ex.backend == "claude"
    assert ex.enabled is True
    assert ExecutionMode.QUICK_PARAM_SWEEP in ex.allowed_modes
    assert ExecutionMode.MARATHON_MULTI_AGENT in ex.allowed_modes
    assert "delegate" in ex.capabilities
    assert ex.restart_policy.continue_flag is True


def test_critic_card_is_claude_with_continue_v04():
    """v0.4 — critic flipped from Codex to Claude; continue_flag now True."""
    cards = discover_agent_cards(agents_root())
    cr = cards["critic"]
    assert cr.backend == "claude"
    assert cr.restart_policy.continue_flag is True
    assert ExecutionMode.QUICK_PARAM_SWEEP not in cr.allowed_modes
    assert ExecutionMode.GUIDED_KERNEL_OPT in cr.allowed_modes


def test_triage_always_on():
    """v0.4 — triage replaces watchdog, runs in every mode."""
    cards = discover_agent_cards(agents_root())
    t = cards["triage"]
    assert t.role == "triage"
    assert t.backend == "claude"
    assert ExecutionMode.QUICK_PARAM_SWEEP in t.allowed_modes
    assert ExecutionMode.GUIDED_KERNEL_OPT in t.allowed_modes
    assert ExecutionMode.MARATHON_MULTI_AGENT in t.allowed_modes
    assert "kill_task" in t.capabilities
    # Triage may emit alerts and observation messages.
    assert "alert" in t.capabilities


def test_system_prompt_path_resolves_under_card_dir():
    """The launcher resolves system_prompt relative to the card dir.

    Plan A: executor + kernel use the skill-style ``SKILL.md`` entrypoint;
    Critic / Watchdog / Sage still use the default ``system_prompt.md``.
    Both paths must resolve under ``card_dir`` and the file must exist.
    """
    cards = discover_agent_cards(agents_root())
    for name, card in cards.items():
        assert card.system_prompt_path.parent == card.card_dir, (
            f"{name}: system_prompt_path should live under card_dir"
        )
        assert card.system_prompt_path.is_file(), (
            f"{name}: system_prompt file {card.system_prompt_path} does not exist"
        )

    # Skill-style agents carry SKILL.md + actions/INDEX.md + reference/ + scripts/.
    for name in ("executor", "kernel"):
        card = cards[name]
        assert card.system_prompt_path.name == "SKILL.md", (
            f"{name}: skill-style entrypoint should be SKILL.md"
        )
        assert (card.card_dir / "actions" / "INDEX.md").is_file()
        assert (card.card_dir / "reference").is_dir()
        assert (card.card_dir / "scripts").is_dir()

    # Other roles still use the legacy single-file convention.
    for name in ("critic", "triage"):
        assert cards[name].system_prompt_path.name == DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# Loader (custom directories)
# ---------------------------------------------------------------------------
def _write_card(dir_: Path, body: str) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    p = dir_ / "agent_card.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_agent_card_minimal(tmp_path: Path):
    body = """
name: hello
role: executor
backend: claude
"""
    cardfile = _write_card(tmp_path / "hello", body)
    card = load_agent_card(cardfile)
    assert card.name == "hello"
    assert card.inbox_filename == DEFAULT_INBOX
    assert card.outbox_filename == DEFAULT_OUTBOX
    assert card.applies_to(ExecutionMode.QUICK_PARAM_SWEEP)  # no allowed_modes => any


def test_load_agent_card_rejects_name_dir_mismatch(tmp_path: Path):
    body = """
name: wrong
role: executor
backend: claude
"""
    cardfile = _write_card(tmp_path / "actual", body)
    with pytest.raises(AgentCardError, match="name"):
        load_agent_card(cardfile)


def test_load_agent_card_rejects_unknown_role(tmp_path: Path):
    body = """
name: alien
role: invader
backend: claude
"""
    cardfile = _write_card(tmp_path / "alien", body)
    with pytest.raises(AgentCardError, match="role"):
        load_agent_card(cardfile)


def test_load_agent_card_rejects_unknown_backend(tmp_path: Path):
    body = """
name: bee
role: executor
backend: ferrari
"""
    cardfile = _write_card(tmp_path / "bee", body)
    with pytest.raises(AgentCardError, match="backend"):
        load_agent_card(cardfile)


def test_load_agent_card_rejects_unknown_mode(tmp_path: Path):
    body = """
name: c
role: executor
backend: claude
allowed_modes:
  - turbo
"""
    cardfile = _write_card(tmp_path / "c", body)
    with pytest.raises(AgentCardError, match="allowed_modes"):
        load_agent_card(cardfile)


def test_load_agent_card_missing_name(tmp_path: Path):
    body = """
role: executor
backend: claude
"""
    cardfile = _write_card(tmp_path / "x", body)
    with pytest.raises(AgentCardError, match="name"):
        load_agent_card(cardfile)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def test_discover_returns_empty_for_missing_root(tmp_path: Path):
    assert discover_agent_cards(tmp_path / "missing") == {}


def test_discover_skips_dot_underscore_dirs(tmp_path: Path):
    body = """
name: real
role: executor
backend: claude
"""
    _write_card(tmp_path / "real", body)
    # These should be ignored
    (tmp_path / "_pycache_").mkdir()
    (tmp_path / ".hidden").mkdir()
    cards = discover_agent_cards(tmp_path)
    assert set(cards.keys()) == {"real"}


def test_discover_raises_on_duplicate_names(tmp_path: Path):
    # Two dirs each contain a card claiming the *same* name (illegal).
    body_a = "name: dup\nrole: executor\nbackend: claude\n"
    body_b = "name: dup\nrole: critic\nbackend: claude\n"
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "alpha" / "agent_card.yaml").write_text(body_a)
    (tmp_path / "beta" / "agent_card.yaml").write_text(body_b)
    with pytest.raises(AgentCardError, match="match directory"):
        discover_agent_cards(tmp_path)


def test_discover_skips_dirs_without_card(tmp_path: Path):
    body = "name: real\nrole: executor\nbackend: claude\n"
    _write_card(tmp_path / "real", body)
    (tmp_path / "no_card_here").mkdir()
    cards = discover_agent_cards(tmp_path)
    assert set(cards.keys()) == {"real"}
