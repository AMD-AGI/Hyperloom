"""Tests for ``orchestrator.persona`` — IMPL-CHECKLIST §5.13‒5.19."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.persona import (
    HARD_TOKEN_LIMIT,
    KEEP_TAIL_TOKENS,
    SOFT_TIME_HOURS,
    archive_old_persona,
    distill_persona,
    estimate_tokens,
    should_distill,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------
def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_small_string():
    assert estimate_tokens("hello") == max(1, len("hello") // 4)


def test_estimate_tokens_growth():
    a = estimate_tokens("x" * 400)
    b = estimate_tokens("x" * 4000)
    assert b > a


# ---------------------------------------------------------------------------
# should_distill
# ---------------------------------------------------------------------------
def test_should_distill_returns_false_outside_marathon(tmp_path: Path):
    p = tmp_path / "executor.md"
    p.write_text("x" * (HARD_TOKEN_LIMIT * 4 + 1), encoding="utf-8")
    assert should_distill(p, ExecutionMode.QUICK_PARAM_SWEEP) is False
    assert should_distill(p, ExecutionMode.GUIDED_KERNEL_OPT) is False


def test_should_distill_marathon_oversized(tmp_path: Path):
    p = tmp_path / "executor.md"
    p.write_text("x" * (HARD_TOKEN_LIMIT * 4 + 1), encoding="utf-8")
    assert should_distill(p, ExecutionMode.MARATHON_MULTI_AGENT) is True


def test_should_distill_marathon_first_pass(tmp_path: Path):
    p = tmp_path / "executor.md"
    p.write_text("hello", encoding="utf-8")
    # last_distill_ts None → trigger
    assert should_distill(p, "marathon_multi_agent", last_distill_ts=None) is True


def test_should_distill_marathon_after_4h(tmp_path: Path):
    p = tmp_path / "executor.md"
    p.write_text("hello", encoding="utf-8")
    earlier = time.time() - (SOFT_TIME_HOURS * 3600.0 + 1)
    assert (
        should_distill(p, "marathon_multi_agent", last_distill_ts=earlier)
        is True
    )


def test_should_distill_marathon_recent_no_trigger(tmp_path: Path):
    p = tmp_path / "executor.md"
    p.write_text("hello", encoding="utf-8")
    assert (
        should_distill(p, "marathon_multi_agent", last_distill_ts=time.time())
        is False
    )


def test_should_distill_marathon_post_keep(tmp_path: Path):
    p = tmp_path / "executor.md"
    p.write_text("hello", encoding="utf-8")
    assert (
        should_distill(
            p,
            "marathon_multi_agent",
            last_distill_ts=time.time(),
            keep_just_happened=True,
        )
        is True
    )


def test_should_distill_missing_file(tmp_path: Path):
    p = tmp_path / "executor.md"
    assert should_distill(p, "marathon_multi_agent") is False


# ---------------------------------------------------------------------------
# archive_old_persona
# ---------------------------------------------------------------------------
def test_archive_old_persona_writes_dated_file(tmp_path: Path):
    archive = tmp_path / "archive"
    out = archive_old_persona("critic", "lots of words", archive)
    assert out.is_file()
    assert out.parent == archive
    assert out.name.startswith("critic-")
    assert out.read_text(encoding="utf-8") == "lots of words"


# ---------------------------------------------------------------------------
# distill_persona — fallback truncation
# ---------------------------------------------------------------------------
def test_distill_persona_truncate_path(tmp_path: Path):
    p = tmp_path / "agent.md"
    body = "lesson-A " * 5000  # ~10k tokens
    p.write_text(body, encoding="utf-8")
    out = distill_persona("agent", backend=None, persona_path=p)
    assert out.startswith("<!-- distilled")
    assert estimate_tokens(out) <= KEEP_TAIL_TOKENS + 50
    # file rewritten
    assert p.read_text(encoding="utf-8") == out


def test_distill_persona_short_input_returns_unchanged(tmp_path: Path):
    p = tmp_path / "agent.md"
    p.write_text("short body", encoding="utf-8")
    out = distill_persona("agent", backend=None, persona_path=p)
    assert "short body" in out


def test_distill_persona_archives_when_dir_set(tmp_path: Path):
    p = tmp_path / "agent.md"
    p.write_text("lots of words", encoding="utf-8")
    archive = tmp_path / "arch"
    distill_persona("agent", backend=None, persona_path=p, archive_dir=archive)
    assert archive.is_dir()
    saved = list(archive.glob("agent-*.md"))
    assert saved
    assert saved[0].read_text(encoding="utf-8") == "lots of words"


# ---------------------------------------------------------------------------
# distill_persona — backend path
# ---------------------------------------------------------------------------
class _IntentLike:
    def __init__(self, intent_type: str, payload: dict[str, Any]) -> None:
        self.type = type("T", (), {"value": intent_type})()
        self.payload = payload


class _StubBackend:
    """Backend stub that emits a single ``UPDATE_PERSONA`` intent."""

    def __init__(self, body: str) -> None:
        self._body = body

    async def run(
        self,
        prompt: str,
        *,
        agent_name: str,
        allowed_tools: tuple[str, ...] = (),
        extra: dict | None = None,
    ):
        from inference_optimizer.orchestrator.intent_parser import (
            Intent,
            IntentType,
        )
        return [Intent(IntentType.UPDATE_PERSONA, {"body_md": self._body})]


def test_distill_persona_uses_backend_when_provided(tmp_path: Path):
    p = tmp_path / "agent.md"
    p.write_text("ancient body of text", encoding="utf-8")
    backend = _StubBackend("crisper distilled body")
    out = distill_persona("agent", backend=backend, persona_path=p)
    assert "crisper distilled body" in out
