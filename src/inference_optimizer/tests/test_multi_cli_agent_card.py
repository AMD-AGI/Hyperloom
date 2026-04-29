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
    cards = discover_agent_cards(agents_root())
    assert set(cards.keys()) == {"executor", "critic", "watchdog", "sage"}


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


def test_critic_card_disables_continue_flag_for_codex():
    cards = discover_agent_cards(agents_root())
    cr = cards["critic"]
    assert cr.backend == "codex"
    assert cr.restart_policy.continue_flag is False
    assert ExecutionMode.QUICK_PARAM_SWEEP not in cr.allowed_modes


def test_watchdog_marathon_only():
    cards = discover_agent_cards(agents_root())
    w = cards["watchdog"]
    assert w.allowed_modes == (ExecutionMode.MARATHON_MULTI_AGENT,)
    assert w.applies_to(ExecutionMode.MARATHON_MULTI_AGENT)
    assert not w.applies_to(ExecutionMode.QUICK_PARAM_SWEEP)


def test_system_prompt_path_resolves_under_card_dir():
    cards = discover_agent_cards(agents_root())
    ex = cards["executor"]
    assert ex.system_prompt_path == ex.card_dir / DEFAULT_PROMPT


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
    body_b = "name: dup\nrole: critic\nbackend: codex\n"
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
