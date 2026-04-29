"""Tests for orchestrator/multi_cli/codex_continuity.py — explicit
conversation.jsonl + budget-aware compaction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.multi_cli.codex_continuity import (
    CodexConversationLog,
    CodexPromptComposer,
    ConversationTurn,
    DEFAULT_CHAR_BUDGET,
    naive_summariser,
    update_after_restart,
)


# ---------------------------------------------------------------------------
# ConversationTurn shape
# ---------------------------------------------------------------------------
def test_turn_round_trip():
    t = ConversationTurn(role="user", content="hello", attempt=3)
    decoded = ConversationTurn.from_json(t.to_json())
    assert decoded.role == "user"
    assert decoded.content == "hello"
    assert decoded.attempt == 3


def test_turn_rejects_bad_role():
    with pytest.raises(ValueError, match="role"):
        ConversationTurn(role="random", content="x")


def test_turn_char_len_includes_overhead():
    t = ConversationTurn(role="user", content="hi")
    assert t.char_len() == 2 + 32


# ---------------------------------------------------------------------------
# CodexConversationLog basics
# ---------------------------------------------------------------------------
def test_append_and_read_back(tmp_path: Path):
    log = CodexConversationLog(path=tmp_path / "conv.jsonl")
    log.append(ConversationTurn(role="user", content="first"))
    log.append(ConversationTurn(role="assistant", content="reply"))
    turns = log.turns()
    assert [t.role for t in turns] == ["user", "assistant"]
    assert [t.content for t in turns] == ["first", "reply"]


def test_turns_skips_garbage_lines(tmp_path: Path):
    p = tmp_path / "conv.jsonl"
    log = CodexConversationLog(path=p)
    log.append(ConversationTurn(role="user", content="real"))
    with p.open("a", encoding="utf-8") as fh:
        fh.write("notjson\n")
    turns = log.turns()
    assert len(turns) == 1


def test_append_many_writes_all(tmp_path: Path):
    log = CodexConversationLog(path=tmp_path / "conv.jsonl")
    log.append_many([
        ConversationTurn(role="user", content="a"),
        ConversationTurn(role="assistant", content="b"),
        ConversationTurn(role="user", content="c"),
    ])
    assert [t.content for t in log.turns()] == ["a", "b", "c"]


def test_char_count_aggregates_per_turn(tmp_path: Path):
    log = CodexConversationLog(path=tmp_path / "conv.jsonl")
    log.append(ConversationTurn(role="user", content="x" * 100))
    log.append(ConversationTurn(role="assistant", content="y" * 200))
    assert log.char_count() == 100 + 200 + 32 * 2


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------
def test_no_compaction_under_budget(tmp_path: Path):
    log = CodexConversationLog(path=tmp_path / "conv.jsonl",
                               char_budget=DEFAULT_CHAR_BUDGET)
    log.append(ConversationTurn(role="user", content="small"))
    assert log.compact_if_over_budget() is False
    assert [t.content for t in log.turns()] == ["small"]


def test_compaction_collapses_middle_turns(tmp_path: Path):
    """When over budget the middle of the log is replaced by a summary,
    head + tail-K turns survive verbatim."""
    log = CodexConversationLog(
        path=tmp_path / "conv.jsonl",
        char_budget=600,           # tight enough to trigger
        keep_recent_turns=2,
    )
    # First turn is system so head_keep clauses fire.
    log.append(ConversationTurn(role="system", content="boot"))
    for i in range(10):
        log.append(ConversationTurn(role="user", content="x" * 200, attempt=i))
    assert log.char_count() > 600
    compacted = log.compact_if_over_budget()
    assert compacted is True
    after = log.turns()
    # head=system; +1 summary; +2 tail = 4 turns
    assert [t.role for t in after] == ["system", "summary", "user", "user"]
    # The two tail turns must be the most recent (attempts 8 + 9).
    assert after[-2].attempt == 8
    assert after[-1].attempt == 9
    # The summary turn must mention dropped turns.
    assert "summary" in after[1].content.lower()


def test_compaction_idempotent_after_first_pass(tmp_path: Path):
    log = CodexConversationLog(
        path=tmp_path / "conv.jsonl", char_budget=400, keep_recent_turns=1,
    )
    log.append(ConversationTurn(role="system", content="boot"))
    for i in range(8):
        log.append(ConversationTurn(role="user", content="x" * 100, attempt=i))
    assert log.compact_if_over_budget() is True
    # After first pass the file is small enough; second pass is a no-op.
    assert log.compact_if_over_budget() is False


def test_compaction_uses_custom_summariser(tmp_path: Path):
    log = CodexConversationLog(path=tmp_path / "conv.jsonl",
                               char_budget=200, keep_recent_turns=1)
    for i in range(8):
        log.append(ConversationTurn(role="user", content="x" * 100, attempt=i))
    log.compact_if_over_budget(summariser=lambda turns: f"SYNTHESIZED({len(turns)})")
    after = log.turns()
    summary_turn = next(t for t in after if t.role == "summary")
    assert summary_turn.content.startswith("SYNTHESIZED(")


def test_compaction_preserves_head_and_tail(tmp_path: Path):
    log = CodexConversationLog(path=tmp_path / "conv.jsonl",
                               char_budget=300, keep_recent_turns=3)
    for i in range(10):
        log.append(ConversationTurn(role="user", content="x" * 80, attempt=i))
    log.compact_if_over_budget()
    after = log.turns()
    # No system head -> head section empty; summary + last 3 tail = 4.
    assert [t.role for t in after] == ["summary", "user", "user", "user"]
    # Last three attempts == 7,8,9.
    assert [t.attempt for t in after[1:]] == [7, 8, 9]


def test_naive_summariser_truncates_long_lines():
    long_turn = ConversationTurn(role="user", content="x" * 500)
    text = naive_summariser([long_turn])
    assert "..." in text
    assert "user" in text


# ---------------------------------------------------------------------------
# CodexPromptComposer
# ---------------------------------------------------------------------------
def test_compose_includes_system_prompt_and_conversation(tmp_path: Path):
    log = CodexConversationLog(path=tmp_path / "conv.jsonl")
    log.append(ConversationTurn(role="user", content="hello"))
    log.append(ConversationTurn(role="assistant", content="hi"))
    composer = CodexPromptComposer(
        system_prompt="# Critic\nYou critique things.",
        conversation_log=log,
    )
    out = composer.compose(attempt=2)
    assert "# Critic" in out
    assert "==== protocol header ====" in out
    assert "==== prior conversation" in out
    assert "hello" in out and "hi" in out
    assert "==== current attempt=2" in out


def test_compose_skips_conversation_section_when_empty(tmp_path: Path):
    log = CodexConversationLog(path=tmp_path / "conv.jsonl")
    composer = CodexPromptComposer(
        system_prompt="# Sage", conversation_log=log,
    )
    out = composer.compose()
    assert "# Sage" in out
    assert "prior conversation" not in out


# ---------------------------------------------------------------------------
# update_after_restart
# ---------------------------------------------------------------------------
def test_update_after_restart_appends_assistant_turn(tmp_path: Path):
    p = tmp_path / "conv.jsonl"
    update_after_restart(p, new_assistant_turn="reply text", attempt=5)
    log = CodexConversationLog(path=p)
    turns = log.turns()
    assert len(turns) == 1
    assert turns[0].role == "assistant"
    assert turns[0].content == "reply text"
    assert turns[0].attempt == 5


def test_update_after_restart_with_no_new_turn(tmp_path: Path):
    p = tmp_path / "conv.jsonl"
    p.write_text('{"role":"user","content":"a","ts":"t"}\n', encoding="utf-8")
    compacted = update_after_restart(p, new_assistant_turn=None)
    assert compacted is False
    log = CodexConversationLog(path=p)
    assert len(log.turns()) == 1


def test_update_after_restart_triggers_compaction(tmp_path: Path):
    p = tmp_path / "conv.jsonl"
    log = CodexConversationLog(path=p, char_budget=300, keep_recent_turns=1)
    for i in range(8):
        log.append(ConversationTurn(role="user", content="x" * 100, attempt=i))
    compacted = update_after_restart(
        p, new_assistant_turn="latest reply",
        char_budget=300, keep_recent_turns=1,
    )
    assert compacted is True
    after = log.turns()
    # First (oldest) is summary; last is the new assistant turn.
    roles = [t.role for t in after]
    assert "summary" in roles
    assert after[-1].role == "assistant"
    assert after[-1].content == "latest reply"
