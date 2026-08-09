# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Codex orchestration runs as one conversation, not 256 cold starts.

The Coordinator decides between a full SEED push and a thin DELTA push from
``backend.conversational``. The Codex backend never had that attribute, so
``push_full`` was permanently True: an 8-hour OpenAI-only run re-pushed shared
state, resource pools, the policy-denial summary, warm start, the gaps ledger
and the cycle strategy on every one of 256 calls -- 5,053,654 prompt chars,
0 DELTA turns. The Anthropic-only run of the same shape sent 270,261 chars over
15 calls (7 SEED + 8 DELTA). The ``orchestration prompt mode=`` log line is
itself gated on ``conversational``, so the degradation left no trace.

The same attribute gates checkpoint compaction, which means the conversation
also grew without bound, and the checkpoint turn passes ``allow_no_intent``,
which the Codex backend did not accept -- enabling it would have raised
TypeError on the first compaction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.codex_session import CodexSession, CodexSessionResult
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import MockBackend, MockTurn, ScriptedPlan
from hyperloom.orchestrator.roles.agent_role import _ORCHESTRATION_INTENTS
from hyperloom.orchestrator.roles.codex import CodexBackend


_SEED_MARKER = "=== Shared session state ==="


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[MockTurn(intents=[])], default_intent=_heartbeat())


def _codex_backend(tmp_path: Path) -> CodexBackend:
    return CodexBackend(
        allowed_intents=_ORCHESTRATION_INTENTS,
        model="gpt-5.6-sol",
        cwd=tmp_path / "codex_workspace",
        sandbox_mode="workspace-write",
    )


def _coordinator(session_dir: Path, tmp_path: Path) -> Coordinator:
    """A Coordinator whose orchestration role runs on the Codex backend."""
    backends: dict[str, Any] = {
        "orchestration": _codex_backend(tmp_path),
        "critic": MockBackend(_silent_plan(), name="critic"),
        "robustness": MockBackend(_silent_plan(), name="robustness"),
    }
    return Coordinator(session_dir, backends=backends)


_HEARTBEAT_ENVELOPE = json.dumps({"intents": [{"intent_type": "send_message", "payload": '{"topic": "heartbeat"}'}]})


def _stub_turn(monkeypatch: pytest.MonkeyPatch, text: str) -> list[dict[str, Any]]:
    """Record every SDK turn the backend issues and reply with ``text``."""
    calls: list[dict[str, Any]] = []

    async def _start(session: CodexSession) -> None:
        return None

    async def _turn(
        session: CodexSession,
        prompt: str,
        *,
        timeout_sec: float,
        output_schema: dict[str, Any] | None = None,
    ) -> CodexSessionResult:
        calls.append({"prompt": prompt, "timeout_sec": timeout_sec, "output_schema": output_schema})
        return CodexSessionResult(text=text, thread_id="thread-1")

    async def _aclose(session: CodexSession) -> None:
        return None

    monkeypatch.setattr(CodexSession, "start", _start)
    monkeypatch.setattr(CodexSession, "turn", _turn)
    monkeypatch.setattr(CodexSession, "aclose", _aclose)
    return calls


def _stub_session(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the session lifecycle a run drives, replying with a heartbeat."""
    calls: list[str] = []

    async def _start(session: CodexSession) -> None:
        calls.append("start")

    async def _turn(
        session: CodexSession,
        prompt: str,
        *,
        timeout_sec: float,
        output_schema: dict[str, Any] | None = None,
    ) -> CodexSessionResult:
        calls.append("turn")
        return CodexSessionResult(text=_HEARTBEAT_ENVELOPE, thread_id="thread-1")

    async def _aclose(session: CodexSession) -> None:
        calls.append("close")

    def _reset_thread(session: CodexSession) -> None:
        calls.append("reset_thread")

    monkeypatch.setattr(CodexSession, "start", _start)
    monkeypatch.setattr(CodexSession, "turn", _turn)
    monkeypatch.setattr(CodexSession, "aclose", _aclose)
    monkeypatch.setattr(CodexSession, "reset_thread", _reset_thread)
    return calls


# ---------------------------------------------------------------------------
# The attribute the whole delta-gating decision hangs on.


def test_codex_orchestration_backend_is_conversational(tmp_path: Path) -> None:
    """Without this the Coordinator re-pushes the full SEED on every tick."""
    assert _codex_backend(tmp_path).conversational is True


def test_coordinator_sees_the_codex_backend_as_conversational(session_dir: Path, tmp_path: Path) -> None:
    assert _coordinator(session_dir, tmp_path)._orchestration_conversational() is True


# ---------------------------------------------------------------------------
# Prompt-mode gating across ticks.


async def _tick(coord: Coordinator) -> str:
    """Compose the orchestration prompt and run the turn it belongs to."""
    prompt = await coord._compose_prompt("orchestration")
    await coord.backends["orchestration"].run(prompt)
    coord._orchestration_seeded = True
    return prompt


async def test_second_orchestration_tick_takes_the_delta_path(
    session_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tick 1 seeds the conversation; tick 2 must send only the delta."""
    _stub_session(monkeypatch)
    coord = _coordinator(session_dir, tmp_path)

    seed_prompt = await _tick(coord)
    delta_prompt = await _tick(coord)

    assert _SEED_MARKER in seed_prompt
    assert _SEED_MARKER not in delta_prompt
    assert len(delta_prompt) < len(seed_prompt)
    assert coord.shared_state.orchestration_prompt_modes == {"seed": 1, "delta": 1}


async def test_reset_conversation_forces_the_next_turn_back_to_seed(
    session_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compaction and cycle restarts rebuild the conversation from a SEED."""
    _stub_session(monkeypatch)
    coord = _coordinator(session_dir, tmp_path)
    await _tick(coord)

    coord._reset_orchestration_conversation()

    assert coord._orchestration_seeded is False
    assert _SEED_MARKER in await coord._compose_prompt("orchestration")


async def test_reset_conversation_replaces_the_thread_not_the_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reset starts a fresh conversation; restarting the client would leak it."""
    calls = _stub_session(monkeypatch)
    backend = _codex_backend(tmp_path)

    await backend.run("tick 1")
    backend.reset_conversation()
    await backend.run("tick 2")

    assert calls.count("start") == 1
    assert calls.count("reset_thread") == 1


async def test_a_rescoped_system_prompt_opens_a_new_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thread instructions are fixed at thread_start, and phase entry rewrites them."""
    calls = _stub_session(monkeypatch)
    backend = _codex_backend(tmp_path)

    await backend.run("tick 1", system_prompt="EXPLORE scope")
    await backend.run("tick 2", system_prompt="EXPLORE scope")
    assert calls.count("reset_thread") == 0

    await backend.run("tick 3", system_prompt="KERNEL_AGENT scope")
    assert calls.count("reset_thread") == 1


async def test_a_replaced_thread_is_re_seeded_not_sent_a_delta(
    session_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delta into an empty thread would drop the whole session state silently."""
    _stub_session(monkeypatch)
    coord = _coordinator(session_dir, tmp_path)
    backend = coord.backends["orchestration"]

    await backend.run("tick 1")
    coord._orchestration_seeded = True
    assert _SEED_MARKER not in await coord._compose_prompt("orchestration")

    # A re-scoped system prompt replaces the thread behind the Coordinator's back.
    await backend.run("tick 2", system_prompt="KERNEL_AGENT scope")
    backend._thread_seeded = False

    assert backend.needs_seed is True
    assert _SEED_MARKER in await coord._compose_prompt("orchestration")


async def test_the_coordinator_closes_the_held_session(
    session_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session owns a child process, so the loop must release it."""
    calls = _stub_session(monkeypatch)
    coord = _coordinator(session_dir, tmp_path)
    await coord.backends["orchestration"].run("tick")

    await coord._close_backends()

    assert calls.count("close") == 1


# ---------------------------------------------------------------------------
# The checkpoint turn the conversational gate unlocks.


async def test_backend_accepts_allow_no_intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint turn passes it; without it compaction raises TypeError."""
    calls = _stub_turn(monkeypatch, json.dumps({"current_plan": "keep tuning attention"}))

    result = await _codex_backend(tmp_path).run("checkpoint please", allow_no_intent=True)

    assert result.intents == []
    assert "current_plan" in result.raw_text
    # A free-form summary cannot satisfy the intent envelope, so that turn must
    # not be schema-constrained.
    assert calls[0]["output_schema"] is None


async def test_orchestration_checkpoint_compacts_through_the_codex_backend(
    session_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the compaction path the conversational gate now reaches."""
    coord = _coordinator(session_dir, tmp_path)
    coord._orchestration_seeded = True
    _stub_turn(
        monkeypatch,
        json.dumps(
            {
                "current_plan": "drive TPOT down via chunked prefill",
                "hypotheses": ["scheduler blocks decode"],
                "tried_and_why": ["raised max-num-seqs: no gain"],
                "pending": ["kernel_agent request for gemm_a16w16"],
                "learnings": ["forge backend needs a resolved source file"],
                "next_cycle_directive": "",
            }
        ),
    )

    assert await coord._maybe_checkpoint_orchestration(tick=5, force=True) is True
    memory = coord.shared_state.orchestration_memory
    assert memory["current_plan"].startswith("drive TPOT down")
    # Compaction re-seeds, so the next tick pushes a full SEED again.
    assert coord._orchestration_seeded is False


async def test_turn_reports_a_context_size_the_checkpoint_policy_can_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Growth is tracked from the per-request context size, not the char count."""

    async def _start(session: CodexSession) -> None:
        return None

    async def _turn(session: CodexSession, prompt: str, **_kwargs: Any) -> CodexSessionResult:
        return CodexSessionResult(
            text=_HEARTBEAT_ENVELOPE,
            usage={"input_tokens": 900, "cache_read_input_tokens": 100},
        )

    monkeypatch.setattr(CodexSession, "start", _start)
    monkeypatch.setattr(CodexSession, "turn", _turn)

    result = await _codex_backend(tmp_path).run("tick")

    assert result.metadata["context_tokens_peak"] == 1000
