# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :class:`CriticAgentBackend`.

The backend drives a 3-step loop (prepare-review → Codex → commit-review);
subprocesses and Codex are bypassed via the *_factory injection points.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from hyperloom.orchestrator.roles import (
    CriticAgentBackend,
    RuntimeCall,
)
from hyperloom.orchestrator.roles.base import BackendError, LLMCallFailed
from hyperloom.common.llm_config import AnthropicMessageResult
from hyperloom.orchestrator.roles.critic_agent import (
    CRITIC_AGENT_LLM_CONNECT_TIMEOUT_SEC,
    CRITIC_AGENT_LLM_RW_TIMEOUT_SEC,
    CRITIC_AGENT_MAX_COMPLETION_TOKENS,
    CRITIC_AGENT_TRUNCATION_RETRY_FACTOR,
    _REVIEW_OUTPUT_INSTRUCTIONS,
    _extract_review_json,
    _is_truncated_finish,
    _reviewed_msg_ids_from_bundle,
    _verdict_references_kb,
)
from hyperloom.orchestrator.specialists.patch_safety import FORBIDDEN_PROPOSAL_FIELDS
from hyperloom.inference_optimizer.protocol.intent import IntentType


# Fakes — Codex client
@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeResp:
    choices: list[FakeChoice] = field(default_factory=list)


class FakeChatCompletions:
    def __init__(self, replies: list[str | tuple[str, str]]):
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        reply = self._replies.pop(0) if self._replies else ""
        # A reply may carry its own finish reason as ``(text, finish_reason)``;
        # a bare string keeps the default so existing callers read unchanged.
        text, finish_reason = reply if isinstance(reply, tuple) else (reply, "stop")
        return FakeResp(choices=[FakeChoice(message=FakeMessage(content=text), finish_reason=finish_reason)])


class FakeChat:
    def __init__(self, completions: FakeChatCompletions):
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, replies: list[str | tuple[str, str]]):
        self.completions = FakeChatCompletions(replies)
        self.chat = FakeChat(self.completions)


# Fakes — runtime.cli subprocess
def _build_envelope_from_review(
    review: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """Mirror critic-agent's commit-review envelope construction."""
    verdicts = review.get("review_verdicts") or review.get("verdicts") or []
    intents: list[dict[str, Any]] = []
    for item in verdicts:
        target = item.get("target_proposal_msg_id")
        verdict = item.get("verdict")
        if not target or not verdict:
            continue
        payload = {
            "target_proposal_msg_id": target,
            "verdict": verdict,
            "source": item.get("source", "critic"),
            "reasoning": item.get("reasoning", ""),
            "predicted_gain_pct": item.get("predicted_gain_pct"),
            "kb_evidence": item.get("kb_evidence") or [],
            "packet_evidence": item.get("packet_evidence") or [],
            "risks": item.get("risks") or [],
            "required_evidence": item.get("required_evidence") or [],
            "alternative_action": item.get("alternative_action"),
            "advice_text": item.get("advice_text", ""),
            "notes": item.get("notes") or [],
        }
        if "confidence" in item:
            payload["confidence"] = item["confidence"]
        intents.append({"intent_type": "review_verdict", "payload": payload})
    for advisory in review.get("advice") or []:
        body = advisory.get("body_md") or advisory.get("text")
        if not body:
            continue
        intents.append(
            {
                "intent_type": "send_message",
                "payload": {"topic": "advice", "body_md": body},
            }
        )
    if not intents:
        intents.append(
            {
                "intent_type": "send_message",
                "payload": {"topic": "heartbeat", "body_md": "ok (critic)"},
            }
        )
    return {"intents": intents}


def _make_fake_runtime(
    *,
    judge_bundle: dict[str, Any],
    fail_phase: str | None = None,
    capture: list[RuntimeCall] | None = None,
) -> Callable[[RuntimeCall], None]:
    """Return a ``RuntimeCaller`` that fakes prepare-review / commit-review."""

    def caller(call: RuntimeCall) -> None:
        if capture is not None:
            capture.append(call)
        if fail_phase == call.phase:
            raise BackendError(f"fake critic-agent runtime.cli {call.phase} exited rc=2: stderr='simulated failure'")
        if call.phase == "prepare-review":
            request = json.loads(call.request_path.read_text(encoding="utf-8"))
            bundle = dict(judge_bundle)
            bundle.setdefault("session_id", request.get("session_id"))
            bundle.setdefault("kind", request.get("kind"))
            call.out_path.write_text(json.dumps(bundle), encoding="utf-8")
        elif call.phase == "commit-review":
            request = json.loads(call.request_path.read_text(encoding="utf-8"))
            review = json.loads(call.review_path.read_text(encoding="utf-8"))
            envelope = _build_envelope_from_review(review, request["session_id"])
            emit = {
                "kind": "coordinator_inbox",
                "session_id": request["session_id"],
                "decision_id": None,
                "kb_writes": [],
                "notes": [],
                "intent_envelope": envelope,
            }
            call.out_path.write_text(json.dumps(emit), encoding="utf-8")
        else:  # pragma: no cover
            raise AssertionError(f"unexpected phase {call.phase!r}")

    return caller


# Fixtures
@pytest.fixture
def fake_critic_root(tmp_path: Path) -> Path:
    """Create a minimal critic-agent root with a stub runtime/cli.py."""
    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub for tests", encoding="utf-8")
    (root / "actions").mkdir()
    (root / "actions" / "review_coordinator_inbox.md").write_text(
        "# fake action prompt",
        encoding="utf-8",
    )
    (root / "SKILL.md").write_text("# fake skill prompt", encoding="utf-8")
    return root


@pytest.fixture
def fake_session_dir(tmp_path: Path) -> Path:
    """Coordinator-style session dir."""
    sd = tmp_path / "session-abc"
    sd.mkdir()
    return sd


def _make_backend(
    fake_critic_root: Path,
    fake_session_dir: Path,
    *,
    codex_replies: list[str | tuple[str, str]],
    judge_bundle: dict[str, Any],
    fail_phase: str | None = None,
    runtime_calls: list[RuntimeCall] | None = None,
    kb_mode: str = "inmemory",
    kb_env: dict[str, str] | None = None,
    known_actions: tuple[str, ...] = (),
) -> tuple[CriticAgentBackend, FakeOpenAIClient]:
    fake_client = FakeOpenAIClient(codex_replies)
    fake_caller = _make_fake_runtime(
        judge_bundle=judge_bundle,
        fail_phase=fail_phase,
        capture=runtime_calls,
    )
    backend = CriticAgentBackend(
        critic_agent_root=fake_critic_root,
        session_dir=fake_session_dir,
        codex_model="gpt-5.4",
        codex_client_factory=lambda: fake_client,
        kb_mode=kb_mode,
        kb_env=kb_env,
        runtime_caller_factory=lambda: fake_caller,
        known_actions=known_actions,
    )
    return backend, fake_client


# _extract_review_json
def test_extract_review_json_fenced():
    text = """Reasoning prose.
```json
{"review_verdicts": [{"target_proposal_msg_id": "abc", "verdict": "approve"}]}
```
trailing prose."""
    out = _extract_review_json(text)
    assert out is not None
    assert out["review_verdicts"][0]["target_proposal_msg_id"] == "abc"


def test_extract_review_json_bare():
    text = '{"review_verdicts": [{"target_proposal_msg_id": "x", "verdict": "reject"}]}'
    out = _extract_review_json(text)
    assert out is not None
    assert out["review_verdicts"][0]["verdict"] == "reject"


def test_extract_review_json_returns_none_on_empty():
    assert _extract_review_json("") is None
    assert _extract_review_json("just prose") is None


@pytest.mark.asyncio
async def test_run_writes_known_actions_into_request_options(
    fake_critic_root: Path,
    fake_session_dir: Path,
) -> None:
    runtime_calls: list[RuntimeCall] = []
    judge_bundle = {
        "kind": "coordinator_inbox",
        "session_id": fake_session_dir.name,
        "proposals": [],
        "review_constraints": {},
    }
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[],
        judge_bundle=judge_bundle,
        runtime_calls=runtime_calls,
        known_actions=("baseline", "validate_stack"),
    )
    await backend.run(prompt="=== inbox ===\n", system_prompt="You are critic.")

    assert runtime_calls
    request = json.loads(
        runtime_calls[0].request_path.read_text(encoding="utf-8"),
    )
    assert request["options"]["known_actions"] == ["baseline", "validate_stack"]


@pytest.mark.asyncio
async def test_run_omits_options_when_known_actions_empty(
    fake_critic_root: Path,
    fake_session_dir: Path,
) -> None:
    runtime_calls: list[RuntimeCall] = []
    judge_bundle = {
        "kind": "coordinator_inbox",
        "session_id": fake_session_dir.name,
        "proposals": [],
        "review_constraints": {},
    }
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[],
        judge_bundle=judge_bundle,
        runtime_calls=runtime_calls,
    )
    await backend.run(prompt="=== inbox ===\n", system_prompt="You are critic.")

    request = json.loads(
        runtime_calls[0].request_path.read_text(encoding="utf-8"),
    )
    assert "options" not in request


def test_extract_review_json_returns_none_when_key_absent():
    assert _extract_review_json('```json\n{"intents": []}\n```') is None


@pytest.mark.parametrize("finish", ["max_tokens", "length", "MAX_TOKENS", " length "])
def test_is_truncated_finish_covers_both_transport_spellings(finish: str):
    """OpenAI says ``length`` where Anthropic says ``max_tokens``."""
    assert _is_truncated_finish(finish) is True


@pytest.mark.parametrize("finish", [None, "", "stop", "end_turn", "tool_use"])
def test_is_truncated_finish_rejects_reasons_that_ended_on_their_own(finish: str | None):
    assert _is_truncated_finish(finish) is False


# Construction
def test_construct_missing_runtime_cli_raises(tmp_path: Path):
    bad_root = tmp_path / "no-runtime"
    bad_root.mkdir()
    sd = tmp_path / "sd"
    sd.mkdir()
    with pytest.raises(BackendError, match="runtime/cli.py not found"):
        CriticAgentBackend(
            critic_agent_root=bad_root,
            session_dir=sd,
            codex_client_factory=lambda: FakeOpenAIClient([]),
            runtime_caller_factory=lambda: lambda call: None,
        )


def test_construct_invalid_kb_mode(fake_critic_root: Path, fake_session_dir: Path):
    with pytest.raises(BackendError, match="kb_mode"):
        CriticAgentBackend(
            critic_agent_root=fake_critic_root,
            session_dir=fake_session_dir,
            kb_mode="bogus",  # type: ignore[arg-type]
            codex_client_factory=lambda: FakeOpenAIClient([]),
            runtime_caller_factory=lambda: lambda call: None,
        )


def test_construct_no_creds_no_factory_raises(monkeypatch, tmp_path: Path):
    """No codex_client_factory and no gateway creds at all -> construction fails fast."""
    for var in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_GATEWAY_KEY"):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub", encoding="utf-8")
    sd = tmp_path / "sd"
    sd.mkdir()
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        CriticAgentBackend(
            critic_agent_root=root,
            session_dir=sd,
            runtime_caller_factory=lambda: lambda call: None,
        )


def test_construct_prefers_explicit_openai_key_over_anthropic_token(monkeypatch, tmp_path: Path):
    """Explicit OPENAI_API_KEY wins over SAFE-filled ANTHROPIC_AUTH_TOKEN for Codex auth."""
    import openai

    monkeypatch.setenv("OPENAI_API_KEY", "openai-user-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "safe-filled")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    captured: dict = {}
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: captured.update(kw) or object())

    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub", encoding="utf-8")
    sd = tmp_path / "sd"
    sd.mkdir()
    CriticAgentBackend(
        critic_agent_root=root,
        session_dir=sd,
        runtime_caller_factory=lambda: lambda call: None,
    )
    assert captured["api_key"] == "openai-user-key"


def _construct_critic_capturing_sdk_kwargs(monkeypatch, tmp_path: Path) -> dict:
    """Build a real-SDK-path CriticAgentBackend, capturing the AsyncOpenAI kwargs."""
    import openai

    captured: dict = {}
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: captured.update(kw) or object())
    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub", encoding="utf-8")
    sd = tmp_path / "sd"
    sd.mkdir()
    CriticAgentBackend(
        critic_agent_root=root,
        session_dir=sd,
        runtime_caller_factory=lambda: lambda call: None,
    )
    return captured


def test_construct_forwards_llm_timeout_knobs_to_the_client(monkeypatch, tmp_path: Path):
    """CRITIC_AGENT_LLM_* seconds reach the SDK as one httpx.Timeout."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-user-key")
    monkeypatch.setenv("CRITIC_AGENT_LLM_CONNECT_TIMEOUT_S", "3")
    monkeypatch.setenv("CRITIC_AGENT_LLM_RW_TIMEOUT_S", "7")
    timeout = _construct_critic_capturing_sdk_kwargs(monkeypatch, tmp_path)["timeout"]
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (3.0, 7.0, 7.0, 7.0)


def test_construct_defaults_llm_timeouts_without_env_knobs(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-user-key")
    monkeypatch.delenv("CRITIC_AGENT_LLM_CONNECT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("CRITIC_AGENT_LLM_RW_TIMEOUT_S", raising=False)
    timeout = _construct_critic_capturing_sdk_kwargs(monkeypatch, tmp_path)["timeout"]
    assert timeout.connect == CRITIC_AGENT_LLM_CONNECT_TIMEOUT_SEC
    assert timeout.read == CRITIC_AGENT_LLM_RW_TIMEOUT_SEC


def test_construct_without_httpx_falls_back_to_sdk_default_timeouts(monkeypatch, tmp_path: Path):
    """httpx is an optional extra; a missing one must not block review entirely."""
    import sys

    monkeypatch.setenv("OPENAI_API_KEY", "openai-user-key")
    monkeypatch.setitem(sys.modules, "httpx", None)
    assert "timeout" not in _construct_critic_capturing_sdk_kwargs(monkeypatch, tmp_path)


def test_construct_missing_openai_sdk_raises_backend_error(monkeypatch, tmp_path: Path):
    import sys

    monkeypatch.setenv("OPENAI_API_KEY", "openai-user-key")
    monkeypatch.setitem(sys.modules, "openai", None)
    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub", encoding="utf-8")
    sd = tmp_path / "sd"
    sd.mkdir()
    with pytest.raises(BackendError, match="openai SDK not installed"):
        CriticAgentBackend(
            critic_agent_root=root,
            session_dir=sd,
            runtime_caller_factory=lambda: lambda call: None,
        )


def test_reviewed_msg_ids_from_bundle_dedups_and_orders():
    bundle = {
        "proposals": [
            {"msg_id": "m1"},
            {"msg_id": "m2"},
            {"msg_id": "m1"},  # dup dropped
            {"no_id": True},
            {"msg_id": ""},  # skipped
        ]
    }
    assert _reviewed_msg_ids_from_bundle(bundle) == ["m1", "m2"]


def test_reviewed_msg_ids_from_bundle_none_when_empty():
    assert _reviewed_msg_ids_from_bundle({"proposals": []}) is None
    assert _reviewed_msg_ids_from_bundle({}) is None
    assert _reviewed_msg_ids_from_bundle({"proposals": "nope"}) is None


# Case 1: Single proposal -> one approve verdict matching the msg_id
@pytest.mark.asyncio
async def test_single_proposal_yields_matching_verdict(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "Llama-3.1-8B", "framework": "sglang"},
        "missing_context": [],
        "required_context": [],
        "proposals": [
            {
                "msg_id": "abc1",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "predicted_gain_pct": 0.0,
                "payload": {"action_name": "baseline"},
            }
        ],
        "kb_priors_by_proposal": {"abc1": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {
            "allowed_verdicts": ["approve", "reject", "redirect", "advise", "needs_review"],
            "approve_requires": [
                "comparable_before_after_benchmark",
                "accuracy_gate_or_waiver",
                "active_path_proof_when_relevant",
                "rollback_plan",
            ],
            "ceiling_importance": 0.84,
        },
        "notes": [],
    }
    reply = """```json
{"review_verdicts": [
  {"target_proposal_msg_id": "abc1", "verdict": "approve",
   "source": "critic", "reasoning": "baseline is the canonical first step"}
]}
```"""
    runtime_calls: list[RuntimeCall] = []
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=judge_bundle,
        runtime_calls=runtime_calls,
    )

    res = await backend.run("prompt-with-proposal-abc1", system_prompt="critic system")

    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.REVIEW_VERDICT
    assert res.intents[0].payload["target_proposal_msg_id"] == "abc1"
    assert res.intents[0].payload["verdict"] == "approve"
    assert res.intents[0].payload["source"] == "critic"

    # Both subprocess phases ran in order, against the right paths.
    assert [c.phase for c in runtime_calls] == ["prepare-review", "commit-review"]
    assert runtime_calls[0].cwd == fake_critic_root
    assert runtime_calls[1].review_path is not None

    # Per-turn workdir was created under session/critic-workdir/000000.
    assert (fake_session_dir / "critic-workdir" / "000000" / "request.json").is_file()
    assert (fake_session_dir / "critic-workdir" / "000000" / "judge_bundle.json").is_file()
    assert (fake_session_dir / "critic-workdir" / "000000" / "review.json").is_file()
    assert (fake_session_dir / "critic-workdir" / "000000" / "emit.json").is_file()

    # The critic's token row records the reviewed proposal msg_id(s).
    import json as _json
    from hyperloom.inference_optimizer.session.session_paths import llm_calls_path

    token_rows = [
        _json.loads(line)
        for line in llm_calls_path(fake_session_dir).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    critic_rows = [r for r in token_rows if r["component"] == "critic"]
    assert critic_rows
    assert critic_rows[0]["reviewed_msg_ids"] == ["abc1"]

    # Default kb_mode injected into the runtime env.
    env = runtime_calls[0].env
    assert env["CRITIC_KB_CLIENT_MODE"] == "inmemory"
    assert env["CRITIC_SESSION_MEMORY_DIR"].endswith("critic-session-memory")


# Case 2: Multiple proposals -> one verdict each
@pytest.mark.asyncio
async def test_multiple_proposals_yield_one_verdict_each(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "p1",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            },
            {
                "msg_id": "p2",
                "from_agent": "orchestration",
                "action_name": "params",
                "payload": {},
                "predicted_gain_pct": 1.0,
            },
        ],
        "kb_priors_by_proposal": {"p1": [], "p2": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    reply = """```json
{"review_verdicts": [
  {"target_proposal_msg_id": "p1", "verdict": "approve", "source": "critic"},
  {"target_proposal_msg_id": "p2", "verdict": "advise",  "source": "critic"}
]}
```"""
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=judge_bundle,
    )
    res = await backend.run("prompt")
    verdicts = [i for i in res.intents if i.type == IntentType.REVIEW_VERDICT]
    assert len(verdicts) == 2
    assert {v.payload["target_proposal_msg_id"] for v in verdicts} == {"p1", "p2"}
    assert {v.payload["verdict"] for v in verdicts} == {"approve", "advise"}


# Case 3: Empty inbox -> heartbeat (LLM never called)
@pytest.mark.asyncio
async def test_empty_proposals_yields_heartbeat_no_llm(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [],
        "kb_priors_by_proposal": {},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    backend, client = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[],
        judge_bundle=judge_bundle,
    )
    res = await backend.run("prompt with no proposals")

    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.SEND_MESSAGE
    assert res.intents[0].payload["topic"] == "heartbeat"
    # LLM short-circuited.
    assert client.completions.calls == []


# Case 4: LLM returns garbage -> the turn fails
@pytest.mark.asyncio
async def test_unparseable_llm_reply_fails_the_turn(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    """Proposals were reviewed and no verdict came back — that is a failure.

    Emitting a heartbeat instead would make it indistinguishable from a critic
    that ran fine and had nothing to say, leaving the proposals pending while
    the loop reads the turn as successful.
    """
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "px",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {"px": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=["I am thinking… no JSON here."],
        judge_bundle=judge_bundle,
    )
    with pytest.raises(BackendError, match="no parseable review_verdicts"):
        await backend.run("prompt")


def _single_proposal_judge_bundle() -> dict[str, Any]:
    return {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "px",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {"px": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }


# Case 4b: the same truncation handling over the OpenAI transport, which spells
# the cut-off reason "length" and carries the cap as max_completion_tokens.
@pytest.mark.asyncio
async def test_openai_truncated_review_is_retried_with_a_bigger_cap(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    complete = (
        '{"review_verdicts": [{"target_proposal_msg_id": "px", '
        '"verdict": "approve", "source": "critic", "reasoning": "ok"}]}'
    )
    backend, client = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[('{"review_verdicts": [{"target_prop', "length"), complete],
        judge_bundle=_single_proposal_judge_bundle(),
    )

    res = await backend.run("prompt")

    verdicts = [i for i in res.intents if i.type == IntentType.REVIEW_VERDICT]
    assert len(verdicts) == 1
    assert verdicts[0].payload["target_proposal_msg_id"] == "px"
    calls = client.completions.calls
    assert len(calls) == 2
    assert calls[0]["kwargs"]["max_completion_tokens"] == CRITIC_AGENT_MAX_COMPLETION_TOKENS
    assert calls[1]["kwargs"]["max_completion_tokens"] == (
        CRITIC_AGENT_MAX_COMPLETION_TOKENS * CRITIC_AGENT_TRUNCATION_RETRY_FACTOR
    )


@pytest.mark.asyncio
async def test_openai_review_truncated_twice_fails_the_turn(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    truncated = ('{"review_verdicts": [{"target_prop', "length")
    backend, client = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[truncated, truncated],
        judge_bundle=_single_proposal_judge_bundle(),
    )

    with pytest.raises(BackendError, match="no parseable review_verdicts"):
        await backend.run("prompt")

    assert len(client.completions.calls) == 2


# Case 5: required_context non-empty -> needs_review + critic_unavailable
@pytest.mark.asyncio
async def test_missing_critical_context_yields_needs_review(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {},
        "proposals": [
            {
                "msg_id": "p1",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {},
        "kb_read_skipped_reason": "missing_critical_context",
        "review_constraints": {},
        "notes": ["model and/or framework unknown — KB priors not fetched"],
        "missing_context": ["model", "framework"],
        "required_context": ["model", "framework"],
    }
    reply = """```json
{"review_verdicts": [
  {"target_proposal_msg_id": "p1", "verdict": "needs_review",
   "source": "critic_unavailable",
   "reasoning": "missing model + framework",
   "notes": ["model", "framework"]}
]}
```"""
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=judge_bundle,
    )
    res = await backend.run("prompt")
    assert len(res.intents) == 1
    p = res.intents[0].payload
    assert p["verdict"] == "needs_review"
    assert p["source"] == "critic_unavailable"
    # Backend surfaces the runtime's reason in metadata for the Coordinator.
    assert res.metadata["kb_read_skipped_reason"] == "missing_critical_context"


# Case 6: subprocess exit code 2 -> BackendError
@pytest.mark.asyncio
async def test_prepare_review_subprocess_failure_raises(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {"proposals": []}  # never read because we fail first.
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[],
        judge_bundle=judge_bundle,
        fail_phase="prepare-review",
    )
    with pytest.raises(BackendError, match=r"rc=2"):
        await backend.run("prompt")


@pytest.mark.asyncio
async def test_commit_review_subprocess_failure_raises(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "z",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {"z": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    reply = '{"review_verdicts": [{"target_proposal_msg_id": "z", "verdict": "approve"}]}'
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=judge_bundle,
        fail_phase="commit-review",
    )
    with pytest.raises(BackendError, match=r"commit-review.*rc=2"):
        await backend.run("prompt")


# Case 7: kb_mode=live + kb_unreachable -> still emits, surfaces reason
@pytest.mark.asyncio
async def test_kb_unreachable_still_emits_verdict(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "p1",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {"p1": []},
        "kb_read_skipped_reason": "kb_unreachable",
        "review_constraints": {
            "kb_breaker": {"open": True, "until": 1234567890},
        },
        "notes": [
            "KB service unreachable (circuit breaker open); proceeding without priors",
        ],
        "missing_context": [],
        "required_context": [],
    }
    reply = """```json
{"review_verdicts": [
  {"target_proposal_msg_id": "p1", "verdict": "advise",
   "source": "critic",
   "reasoning": "approving without KB priors is risky; advise instead",
   "notes": ["KB unreachable"]}
]}
```"""
    runtime_calls: list[RuntimeCall] = []
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=judge_bundle,
        runtime_calls=runtime_calls,
        kb_mode="live",
        kb_env={"KB_BASE_URL": "http://127.0.0.1:1"},
    )
    res = await backend.run("prompt")

    assert len(res.intents) == 1
    assert res.intents[0].payload["verdict"] == "advise"
    assert res.metadata["kb_read_skipped_reason"] == "kb_unreachable"
    # KB env passed through to the subprocess
    env = runtime_calls[0].env
    assert env["CRITIC_KB_CLIENT_MODE"] == "live"
    assert env["KB_BASE_URL"] == "http://127.0.0.1:1"


@pytest.mark.asyncio
async def test_kb_live_without_url_raises(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {"proposals": []}
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[],
        judge_bundle=judge_bundle,
        kb_mode="live",
        kb_env={},
    )
    # No KB_BASE_URL in env either: ensure it's truly absent.
    saved = os.environ.pop("KB_BASE_URL", None)
    try:
        with pytest.raises(BackendError, match="KB_BASE_URL"):
            await backend.run("prompt")
    finally:
        if saved is not None:
            os.environ["KB_BASE_URL"] = saved


# Multi-turn: counter increments, workdirs scoped per-turn
@pytest.mark.asyncio
async def test_per_turn_workdirs_are_isolated(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "px",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {"px": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    reply = '{"review_verdicts": [{"target_proposal_msg_id": "px", "verdict": "approve"}]}'
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply, reply],
        judge_bundle=judge_bundle,
    )
    await backend.run("turn 1")
    await backend.run("turn 2")
    assert (fake_session_dir / "critic-workdir" / "000000" / "request.json").is_file()
    assert (fake_session_dir / "critic-workdir" / "000001" / "request.json").is_file()


# Output instructions are appended (verifies prompt construction)
@pytest.mark.asyncio
async def test_user_prompt_includes_judge_bundle_and_instructions(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "abc",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {"abc": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    reply = '{"review_verdicts": [{"target_proposal_msg_id": "abc", "verdict": "approve"}]}'
    backend, client = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=judge_bundle,
    )
    await backend.run("ignored", system_prompt="you are critic")
    call = client.completions.calls[0]
    assert call["model"] == "gpt-5.4"
    assert call["messages"][0] == {"role": "system", "content": "you are critic"}
    user_text = call["messages"][1]["content"]
    assert "JUDGE BUNDLE" in user_text
    assert "OUTPUT FORMAT" in user_text
    assert '"abc"' in user_text  # proposal msg_id from judge bundle


def test_the_output_schema_asks_for_the_rule_the_verdict_rests_on():
    """The Critic is told to reply with *exactly* this schema, and the
    Coordinator holds a reject to the verdict its cited rule declared by reading
    `failure_reason_code`. Documenting the field only in a reference file
    nothing loads is why prose-scanning became the only signal in production."""
    schema, _, rules = _REVIEW_OUTPUT_INSTRUCTIONS.partition("Rules (mirror")

    assert '"failure_reason_code"' in schema
    assert "failure_reason_code" in rules


def _bundle_reviewing(action_name: str) -> dict[str, Any]:
    """A judge bundle whose single proposal proposes ``action_name``."""
    return {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "abc",
                "from_agent": "orchestration",
                "action_name": action_name,
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {"abc": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }


async def _review_constraints_sent_for(
    action_name: str,
    fake_critic_root: Path,
    fake_session_dir: Path,
) -> dict[str, Any]:
    """Run one review turn and return the ``review_constraints`` the model saw."""
    backend, client = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=['{"review_verdicts": [{"target_proposal_msg_id": "abc", "verdict": "approve"}]}'],
        judge_bundle=_bundle_reviewing(action_name),
    )
    await backend.run("ignored", system_prompt="you are critic")
    user_text = client.completions.calls[0]["messages"][1]["content"]
    match = re.search(
        r"==== JUDGE BUNDLE ====\s*(\{.*?\})\s*==== END JUDGE BUNDLE ====",
        user_text,
        re.DOTALL,
    )
    assert match
    return json.loads(match.group(1))["review_constraints"]


@pytest.mark.asyncio
async def test_the_reviewed_bundle_carries_the_quantitative_claim_rule(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    """Delivered as data so the Critic's field list stays identical to the one
    the runner strips, and so a format slip is advisory rather than a reject
    that costs the round every proposal in the set."""
    constraints = await _review_constraints_sent_for("specialist", fake_critic_root, fake_session_dir)

    rule = constraints["quantitative_claim_rule"]
    assert rule["failure_verdict"] == "advise"
    assert set(rule["forbidden_proposal_fields"]) == set(FORBIDDEN_PROPOSAL_FIELDS)


@pytest.mark.asyncio
async def test_a_review_the_rule_cannot_apply_to_is_not_handed_the_rule(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    """The rule is about ``proposal_set[*]``, which a ``baseline`` proposal has
    no room for. Sending it anyway invites a citation the verdict path then has
    to read, so it goes only where it can be violated."""
    constraints = await _review_constraints_sent_for("baseline", fake_critic_root, fake_session_dir)

    assert "quantitative_claim_rule" not in constraints


# Static context propagation — backend sources model/framework from manifest.json or explicit static_context.
def _write_manifest(session_dir: Path, payload: dict[str, Any]) -> Path:
    """Write a minimal manifest.json the backend can ingest."""
    target = session_dir / "manifest.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


@pytest.mark.asyncio
async def test_run_populates_request_context_from_manifest(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    _write_manifest(
        fake_session_dir,
        {
            "schema_version": 1,
            "session_id": "sess-1",
            "model_name": "Llama-3.1-8B-Instruct",
            "model_path": "/models/llama-3.1-8b",
            "framework": "sglang",
            "gpu_type": "mi300x",
            "tp": 8,
            "workload": {
                "isl": 1024,
                "osl": 1024,
                "max_model_len": 4096,
                "precision": "fp8",
                "conc": 64,
            },
        },
    )
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "Llama-3.1-8B-Instruct", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "p1",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {"p1": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    reply = '{"review_verdicts": [{"target_proposal_msg_id": "p1", "verdict": "approve"}]}'
    runtime_calls: list[RuntimeCall] = []
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=judge_bundle,
        runtime_calls=runtime_calls,
    )
    await backend.run("prompt")

    # Read back the persisted request.json to verify context came from the manifest.
    request = json.loads((fake_session_dir / "critic-workdir" / "000000" / "request.json").read_text(encoding="utf-8"))
    ctx = request["context"]
    assert ctx["model"] == "Llama-3.1-8B-Instruct"
    assert ctx["framework"] == "sglang"
    assert ctx["gpu_type"] == "mi300x"
    assert ctx["model_path"] == "/models/llama-3.1-8b"
    assert ctx["tp"] == 8
    assert ctx["precision"] == "fp8"
    assert ctx["workload"]["isl"] == 1024
    assert ctx["workload"]["osl"] == 1024
    assert ctx["workload"]["conc"] == 64


@pytest.mark.asyncio
async def test_static_context_override_wins_over_manifest(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    # Manifest says sglang, but the explicit static_context overrides it.
    _write_manifest(
        fake_session_dir,
        {
            "schema_version": 1,
            "model_name": "ignored-by-test",
            "framework": "sglang",
        },
    )
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {},
        "proposals": [],
        "kb_priors_by_proposal": {},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    fake_caller = _make_fake_runtime(judge_bundle=judge_bundle)
    backend = CriticAgentBackend(
        critic_agent_root=fake_critic_root,
        session_dir=fake_session_dir,
        codex_client_factory=lambda: FakeOpenAIClient([]),
        runtime_caller_factory=lambda: fake_caller,
        static_context={"model": "explicit-m", "framework": "vllm", "gpu_type": "mi355x"},
    )
    backend.set_trace_context(tick=8, phase="FRAMEWORK_AGENT", macro_cycle=3)
    await backend.run("prompt")
    request = json.loads((fake_session_dir / "critic-workdir" / "000000" / "request.json").read_text(encoding="utf-8"))
    assert request["context"] == {
        "model": "explicit-m",
        "framework": "vllm",
        "gpu_type": "mi355x",
        "phase": "FRAMEWORK_AGENT",
        "macro_cycle": 3,
    }


@pytest.mark.asyncio
async def test_missing_manifest_falls_back_to_empty_context(
    fake_critic_root: Path,
    fake_session_dir: Path,
    caplog,
):
    # No manifest written: backend must not raise; request.context = {} + a WARNING is logged.
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {},
        "proposals": [],
        "kb_priors_by_proposal": {},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    fake_caller = _make_fake_runtime(judge_bundle=judge_bundle)
    with caplog.at_level("WARNING", logger="hyperloom.orchestrator.roles.critic_agent"):
        backend = CriticAgentBackend(
            critic_agent_root=fake_critic_root,
            session_dir=fake_session_dir,
            codex_client_factory=lambda: FakeOpenAIClient([]),
            runtime_caller_factory=lambda: fake_caller,
        )
    assert backend._static_context == {}
    assert any("manifest.json not found" in rec.getMessage() for rec in caplog.records)

    await backend.run("prompt")
    request = json.loads((fake_session_dir / "critic-workdir" / "000000" / "request.json").read_text(encoding="utf-8"))
    assert request["context"] == {}


@pytest.mark.asyncio
async def test_malformed_manifest_logs_warning_and_falls_back(
    fake_critic_root: Path,
    fake_session_dir: Path,
    caplog,
):
    # Corrupt manifest JSON → backend logs + falls back, doesn't crash the boot.
    (fake_session_dir / "manifest.json").write_text(
        "{ this is not json",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="hyperloom.orchestrator.roles.critic_agent"):
        backend = CriticAgentBackend(
            critic_agent_root=fake_critic_root,
            session_dir=fake_session_dir,
            codex_client_factory=lambda: FakeOpenAIClient([]),
            runtime_caller_factory=lambda: lambda call: None,
        )
    assert backend._static_context == {}
    assert any("failed to load manifest.json" in rec.getMessage() for rec in caplog.records)


def test_load_static_context_skips_unknown_and_empty_fields(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    # Empty / missing values are skipped upstream so the keys never appear in the JSON.
    _write_manifest(
        fake_session_dir,
        {
            "schema_version": 1,
            "model_name": "m",
            "framework": "",
            "gpu_type": None,
            "tp": 0,
            "workload": {"isl": 1024, "precision": ""},
        },
    )
    backend = CriticAgentBackend(
        critic_agent_root=fake_critic_root,
        session_dir=fake_session_dir,
        codex_client_factory=lambda: FakeOpenAIClient([]),
        runtime_caller_factory=lambda: lambda call: None,
    )
    ctx = backend._static_context
    assert ctx == {"model": "m", "workload": {"isl": 1024}}


# Diagnostic plumbing — required_context surfaces in metadata + log line.
@pytest.mark.asyncio
async def test_required_context_surfaces_in_metadata(
    fake_critic_root: Path,
    fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {},
        "proposals": [
            {
                "msg_id": "p1",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "payload": {},
                "predicted_gain_pct": 0.0,
            }
        ],
        "kb_priors_by_proposal": {},
        "kb_read_skipped_reason": "missing_critical_context",
        "review_constraints": {},
        "notes": [],
        "missing_context": ["model", "framework"],
        "required_context": ["model", "framework"],
    }
    reply = """```json
{"review_verdicts": [
  {"target_proposal_msg_id": "p1", "verdict": "needs_review",
   "source": "critic_unavailable", "reasoning": "no ctx"}
]}
```"""
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=judge_bundle,
    )
    res = await backend.run("prompt")
    assert res.metadata["required_context"] == ["model", "framework"]
    assert backend.calls[-1]["required_context"] == ["model", "framework"]


"""End-to-end test: real critic-agent runtime + mocked Codex + Coordinator.

Shells out to the real ``hyperloom/agents/critic/runtime/cli.py`` (only Codex is faked).
Marker ``critic_agent_e2e`` lets devs without the checkout skip it.
"""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.cli.credentials import _resolve_agent_root
from hyperloom.orchestrator.roles import (
    CriticAgentBackend,
    MockBackend,
    MockRobustnessBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


pytestmark = pytest.mark.critic_agent_e2e


# Fake Codex (only the LLM layer; the runtime is real).
@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg
    finish_reason: str = "stop"


@dataclass
class _Resp:
    choices: list[_Choice] = field(default_factory=list)


class _DeterministicCompletions:
    """Reads the user prompt's judge bundle and approves every proposal it sees."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages})
        user_text = messages[-1]["content"] if messages else ""
        # Pull the judge bundle JSON out of the marker-wrapped user prompt.
        m = re.search(
            r"==== JUDGE BUNDLE ====\s*(\{.*?\})\s*==== END JUDGE BUNDLE ====",
            user_text,
            re.DOTALL,
        )
        verdicts: list[dict[str, Any]] = []
        if m:
            try:
                bundle = json.loads(m.group(1))
            except json.JSONDecodeError:
                bundle = {}
            for proposal in bundle.get("proposals") or []:
                msg_id = proposal.get("msg_id")
                if not msg_id:
                    continue
                verdicts.append(
                    {
                        "target_proposal_msg_id": msg_id,
                        "verdict": "approve",
                        "source": "critic",
                        "reasoning": "deterministic e2e fixture — auto-approve",
                        "confidence": "medium",
                    }
                )
        body = json.dumps({"review_verdicts": verdicts})
        return _Resp(choices=[_Choice(message=_Msg(content=f"```json\n{body}\n```"))])


class _DeterministicChat:
    def __init__(self, completions):
        self.completions = completions


class _DeterministicClient:
    def __init__(self):
        self.completions = _DeterministicCompletions()
        self.chat = _DeterministicChat(self.completions)


@pytest.fixture
def critic_agent_root() -> Path:
    """Locate the real critic-agent checkout. Skip gracefully if absent."""
    root = _resolve_agent_root("critic")
    if root is None:
        pytest.skip(
            "critic-agent runtime not found — set CRITIC_AGENT_ROOT or check the src/hyperloom/agents/critic/ install"
        )
    return root


def _heartbeat() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


# scripted Orchestration -> real CriticAgentBackend -> approved
@pytest.mark.asyncio
async def test_critic_agent_real_runtime_clears_proposal(
    session_dir: Path,
    critic_agent_root: Path,
):
    """Orchestration proposes baseline -> real runtime emits review_verdict{approve} -> Coordinator materializes the task."""
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "baseline",
            "predicted_gain_pct": 0.0,
        },
    )

    critic_backend = CriticAgentBackend(
        critic_agent_root=critic_agent_root,
        session_dir=session_dir,
        codex_model="gpt-5.4",
        codex_client_factory=_DeterministicClient,
        kb_mode="inmemory",
        # No runtime_caller_factory: exercise the real subprocess path.
    )

    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(
                turns=[MockTurn(intents=[propose])],
                default_intent=_heartbeat(),
            ),
            name="orchestration",
        ),
        "critic": critic_backend,
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(session_dir, backends=backends)

    async def _runner(ctx) -> dict:
        return {"status": "succeeded", "tput": 1234.0, "kind": ctx.task.kind}

    c.sub.register_executor("baseline", _runner)

    try:
        # 3 ticks: propose → review_verdict{approve} → materialize + dispatch baseline.
        await c.tick(3)

        proposals = await c.bus.tail(topic="proposal")
        verdicts = await c.bus.tail(topic="review_verdict")
        decisions = await c.bus.tail(topic="decision")

        assert len(proposals) >= 1, f"no proposals on bus, got {proposals}"

        # Identify the real-runtime path via from=critic, the fake LLM's reasoning text, and the filesystem evidence asserted below.
        assert verdicts, "expected at least one review_verdict on the bus"
        approved = [v for v in verdicts if v.payload.get("verdict") == "approve"]
        assert approved, f"no approve verdict, got {[v.payload for v in verdicts]}"
        assert all(v.from_agent == "critic" for v in approved), "verdicts must originate from critic role"
        assert all("deterministic e2e fixture" in (v.payload.get("reasoning") or "") for v in approved), (
            f"verdict reasoning should match the fake LLM's text "
            f"('deterministic e2e fixture — auto-approve'); a 'mock critic' "
            f"reasoning would mean MockCriticBackend ran instead. "
            f"Payloads: {[v.payload for v in verdicts]}"
        )

        # Coordinator turned the approved proposal into a decision.
        assert any(d.payload.get("kind") == "approved_proposal" for d in decisions), (
            f"approved proposal didn't materialise into a decision: {[d.payload for d in decisions]}"
        )

    finally:
        await c.stop()

    # Filesystem assertions: real runtime wrote session memory.
    workdir = session_dir / "critic-workdir"
    assert workdir.is_dir()
    turn0 = workdir / "000000"
    for fname in ("request.json", "judge_bundle.json", "review.json", "emit.json"):
        assert (turn0 / fname).is_file(), f"missing per-turn artefact {fname}"

    memory_root = session_dir / "critic-session-memory"
    assert memory_root.is_dir(), f"session memory dir missing: {memory_root}"
    session_memories = list(memory_root.iterdir())
    assert session_memories, f"session memory dir is empty under {memory_root}"
    sm_dir = session_memories[0]
    # The runtime stamps decisions.jsonl + reviewed_msg_ids.json once a verdict commits.
    assert (sm_dir / "decisions.jsonl").is_file(), (
        f"decisions.jsonl missing under {sm_dir} (entries: {list(sm_dir.iterdir())})"
    )
    assert (sm_dir / "reviewed_msg_ids.json").is_file(), f"reviewed_msg_ids.json missing under {sm_dir}"

    # The Coordinator generates the msg_id, so just check the file is non-trivial.
    reviewed_raw = (sm_dir / "reviewed_msg_ids.json").read_text(encoding="utf-8")
    reviewed = json.loads(reviewed_raw)
    assert reviewed, f"reviewed_msg_ids.json should be non-empty, got {reviewed!r}"


@pytest.mark.asyncio
async def test_critic_agent_heartbeat_when_no_proposal(
    session_dir: Path,
    critic_agent_root: Path,
):
    """No proposals → real runtime falls back to a heartbeat envelope and short-circuits the LLM."""
    critic_backend = CriticAgentBackend(
        critic_agent_root=critic_agent_root,
        session_dir=session_dir,
        codex_model="gpt-5.4",
        codex_client_factory=_DeterministicClient,
        kb_mode="inmemory",
    )

    # Orchestration only heartbeats — no proposal ever surfaces.
    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[], default_intent=_heartbeat()),
            name="orchestration",
        ),
        "critic": critic_backend,
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(2)
        verdicts = await c.bus.tail(topic="review_verdict")
        assert not verdicts, f"unexpected verdicts in heartbeat-only run: {verdicts}"

        # Confirm the heartbeat path by checking the LLM was NOT called (zero proposals).
        client = critic_backend._client  # type: ignore[attr-defined]
        assert client.completions.calls == [], (
            f"LLM should be skipped when proposals are empty; calls={client.completions.calls}"
        )
    finally:
        await c.stop()


# KB trace + dry-run injection gate
def test_verdict_references_kb_helper():
    assert _verdict_references_kb(None) is False
    assert _verdict_references_kb({"review_verdicts": []}) is False
    assert _verdict_references_kb({"review_verdicts": [{"verdict": "approve"}]}) is False
    assert _verdict_references_kb({"review_verdicts": [{"verdict": "reject", "kb_evidence": ["kb_x"]}]}) is True


def test_build_kb_priors_trace_counts_and_reference():
    judge_bundle = {
        "kb_priors_trace": {
            "configured": True,
            "mode": "per_proposal",
            "client_mode": "live",
            "scope_filter": {"model": "m"},
            "limit": 5,
            "requests": [{"msg_id": "p1", "count": 2}],
        },
        "kb_priors_by_proposal": {"p1": [{"slug": "a"}, {"slug": "b"}]},
        "kb_priors_for_decision": [],
        "kb_read_skipped_reason": None,
    }
    review = {"review_verdicts": [{"verdict": "approve", "kb_evidence": ["k"]}]}
    trace = CriticAgentBackend._build_kb_priors_trace(judge_bundle, review)
    assert trace["configured"] is True
    assert trace["mode"] == "per_proposal"
    assert trace["prior_count"] == 2
    assert trace["referenced_in_verdict"] is True


def test_build_kb_priors_trace_carries_skip_reason():
    judge_bundle = {
        "kb_priors_trace": {},
        "kb_priors_by_proposal": {},
        "kb_priors_for_decision": [],
        "kb_read_skipped_reason": "kb_unreachable",
    }
    trace = CriticAgentBackend._build_kb_priors_trace(judge_bundle, None)
    assert trace["skipped_reason"] == "kb_unreachable"
    assert trace["prior_count"] == 0


def _kb_trace_judge_bundle() -> dict[str, Any]:
    return {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {
                "msg_id": "p1",
                "from_agent": "orchestration",
                "action_name": "sweep",
                "payload": {},
                "predicted_gain_pct": 1.0,
            },
        ],
        "kb_priors_by_proposal": {"p1": []},
        "kb_priors_for_decision": [],
        "kb_priors_trace": {
            "configured": True,
            "mode": "per_proposal",
            "client_mode": "",
            "scope_filter": {"model": "m"},
            "limit": 5,
            "requests": [{"msg_id": "p1", "cache": "miss", "count": 0}],
        },
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }


class _FakeKbEmitter:
    def __init__(self):
        self.enabled = True
        self.spans: list[dict[str, Any]] = []

    def record_kb_span(self, **kwargs):
        self.spans.append(kwargs)


@pytest.mark.asyncio
async def test_run_mirrors_kb_trace_to_langfuse(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch,
):
    fake_em = _FakeKbEmitter()
    from hyperloom.orchestrator.trace import langfuse_emitter as lfe

    monkeypatch.setattr(lfe, "get_emitter", lambda sd: fake_em)
    reply = '{"review_verdicts": [{"target_proposal_msg_id": "p1", "verdict": "approve", "source": "critic"}]}'
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=_kb_trace_judge_bundle(),
    )
    await backend.run("prompt")
    names = sorted(s["name"] for s in fake_em.spans)
    assert names == ["kb_priors:iter_0"]
    agents = {s["agent"] for s in fake_em.spans}
    assert agents == {"critic"}


@pytest.mark.asyncio
async def test_run_skips_langfuse_mirror_when_disabled(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch,
):
    fake_em = _FakeKbEmitter()
    fake_em.enabled = False
    from hyperloom.orchestrator.trace import langfuse_emitter as lfe

    monkeypatch.setattr(lfe, "get_emitter", lambda sd: fake_em)
    reply = '{"review_verdicts": [{"target_proposal_msg_id": "p1", "verdict": "approve", "source": "critic"}]}'
    backend, _ = _make_backend(
        fake_critic_root,
        fake_session_dir,
        codex_replies=[reply],
        judge_bundle=_kb_trace_judge_bundle(),
    )
    await backend.run("prompt")
    assert fake_em.spans == []


# Claude CLI review path (protocol="anthropic")
_CRITIC_MOD = "hyperloom.orchestrator.roles.critic_agent"


class FakeAnthropicCompletion:
    """Records ``aanthropic_completion`` calls and replays queued results."""

    def __init__(self, results: list[Any]):
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **params: Any) -> Any:
        self.calls.append(params)
        result = self._results.pop(0) if self._results else _anthropic_review_result("")
        if isinstance(result, Exception):
            raise result
        return result


def _anthropic_review_result(
    review_json: str,
    *,
    stop_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> AnthropicMessageResult:
    return AnthropicMessageResult(
        text=review_json,
        stop_reason=stop_reason,
        usage=usage if usage is not None else {"input_tokens": 21, "output_tokens": 7},
    )


def _make_anthropic_backend(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    results: list[Any],
    judge_bundle: dict[str, Any],
    claude_model: str = "claude-opus-4-8",
) -> tuple[CriticAgentBackend, FakeAnthropicCompletion]:
    """Wire a protocol=anthropic critic onto a recorded single-shot entry point.

    The credential probe is stubbed too: which transport llm_config would pick
    is its own concern, and the critic must not depend on the host's env.
    """
    fake_completion = FakeAnthropicCompletion(results)
    monkeypatch.setattr(f"{_CRITIC_MOD}.anthropic_transport_ready", lambda *_a, **_kw: True)
    monkeypatch.setattr(f"{_CRITIC_MOD}.aanthropic_completion", fake_completion)
    fake_caller = _make_fake_runtime(judge_bundle=judge_bundle)
    backend = CriticAgentBackend(
        critic_agent_root=fake_critic_root,
        session_dir=fake_session_dir,
        protocol="anthropic",
        claude_model=claude_model,
        codex_model="gpt-5.4",
        runtime_caller_factory=lambda: fake_caller,
    )
    return backend, fake_completion


@pytest.mark.asyncio
async def test_anthropic_protocol_single_proposal_yields_verdict(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "Llama-3.1-8B", "framework": "sglang"},
        "missing_context": [],
        "required_context": [],
        "proposals": [
            {
                "msg_id": "abc1",
                "from_agent": "orchestration",
                "action_name": "baseline",
                "predicted_gain_pct": 0.0,
                "payload": {"action_name": "baseline"},
            }
        ],
        "kb_priors_by_proposal": {"abc1": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {"allowed_verdicts": ["approve", "reject"]},
        "notes": [],
    }
    review_json = (
        '{"review_verdicts": [{"target_proposal_msg_id": "abc1", '
        '"verdict": "approve", "source": "critic", "reasoning": "canonical baseline"}]}'
    )
    backend, fake_completion = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[_anthropic_review_result(review_json)],
        judge_bundle=judge_bundle,
    )

    res = await backend.run("prompt-with-proposal-abc1", system_prompt="critic system")

    # Full KB+tools critic-agent produced the verdict via the Anthropic path.
    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.REVIEW_VERDICT
    assert res.intents[0].payload["target_proposal_msg_id"] == "abc1"
    assert res.intents[0].payload["verdict"] == "approve"
    assert res.metadata["model"] == "claude-opus-4-8"
    # One call carrying the system prompt in its own field, under the shared cap.
    assert len(fake_completion.calls) == 1
    call = fake_completion.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["system"] == "critic system"
    assert call["max_tokens"] == CRITIC_AGENT_MAX_COMPLETION_TOKENS
    user_content = call["messages"][0]["content"]
    assert "JUDGE BUNDLE" in user_content
    assert "critic system" not in user_content

    # Token usage from the turn metadata landed on the critic trace row.
    import json as _json
    from hyperloom.inference_optimizer.session.session_paths import llm_calls_path

    critic_rows = [
        _json.loads(line)
        for line in llm_calls_path(fake_session_dir).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    critic_rows = [r for r in critic_rows if r["component"] == "critic"]
    assert critic_rows
    assert critic_rows[0]["input_tokens"] == 21
    assert critic_rows[0]["output_tokens"] == 7
    assert critic_rows[0]["model"] == "claude-opus-4-8"


def _minimal_judge_bundle() -> dict[str, Any]:
    return {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [{"msg_id": "p1", "from_agent": "orchestration", "action_name": "baseline", "payload": {}}],
        "review_constraints": {},
    }


def _truncated_anthropic_result() -> AnthropicMessageResult:
    """A reply the model stopped writing when it ran out of output budget."""
    return _anthropic_review_result(
        '{"review_verdicts": [{"target_proposal',
        stop_reason="max_tokens",
        usage={"input_tokens": 9, "output_tokens": CRITIC_AGENT_MAX_COMPLETION_TOKENS},
    )


@pytest.mark.asyncio
async def test_truncated_review_is_retried_with_a_bigger_cap(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Re-asking under the same cap would truncate at the same byte, so the
    retry only earns its keep by raising the ceiling."""
    complete = _anthropic_review_result(
        '{"review_verdicts": [{"target_proposal_msg_id": "p1", '
        '"verdict": "approve", "source": "critic", "reasoning": "ok"}]}'
    )
    backend, fake_completion = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[_truncated_anthropic_result(), complete],
        judge_bundle=_minimal_judge_bundle(),
    )

    res = await backend.run("prompt", system_prompt="critic system")

    verdicts = [i for i in res.intents if i.type == IntentType.REVIEW_VERDICT]
    assert len(verdicts) == 1
    assert verdicts[0].payload["target_proposal_msg_id"] == "p1"
    assert len(fake_completion.calls) == 2
    assert fake_completion.calls[0]["max_tokens"] == CRITIC_AGENT_MAX_COMPLETION_TOKENS
    assert fake_completion.calls[1]["max_tokens"] == (
        CRITIC_AGENT_MAX_COMPLETION_TOKENS * CRITIC_AGENT_TRUNCATION_RETRY_FACTOR
    )


@pytest.mark.asyncio
async def test_review_truncated_twice_fails_the_turn(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verdicts that never arrive must not be reported as verdicts that say
    nothing: the loop would keep re-asking the question it already can't
    answer, and nothing on the record would say why."""
    backend, fake_completion = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[_truncated_anthropic_result(), _truncated_anthropic_result()],
        judge_bundle=_minimal_judge_bundle(),
    )

    with pytest.raises(BackendError, match="no parseable review_verdicts") as excinfo:
        await backend.run("prompt", system_prompt="critic system")

    # The message has to name the cap, or an operator cannot tell a truncated
    # review from a model that answered in prose.
    assert "max_tokens" in str(excinfo.value)
    assert len(fake_completion.calls) == 2


@pytest.mark.asyncio
async def test_rejected_retry_still_reports_the_truncation(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A model whose own output limit sits below the doubled cap rejects the
    retry. Surfacing only that rejection would send the reader after the retry
    instead of the truncation that forced it."""
    backend, fake_completion = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[
            _truncated_anthropic_result(),
            RuntimeError("max_tokens: 64000 > 32000, the maximum allowed for this model"),
        ],
        judge_bundle=_minimal_judge_bundle(),
    )

    with pytest.raises(BackendError, match="was truncated at") as excinfo:
        await backend.run("prompt", system_prompt="critic system")

    message = str(excinfo.value)
    assert str(CRITIC_AGENT_MAX_COMPLETION_TOKENS) in message
    assert str(CRITIC_AGENT_MAX_COMPLETION_TOKENS * CRITIC_AGENT_TRUNCATION_RETRY_FACTOR) in message
    # The provider's own words are kept, so the cap that was too large is named.
    assert "maximum allowed for this model" in message
    assert len(fake_completion.calls) == 2


@pytest.mark.asyncio
async def test_unparseable_review_is_not_retried(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A reply that ended on its own terms is a formatting failure, not a
    budget one, so a second call at a bigger cap buys nothing."""
    backend, fake_completion = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[_anthropic_review_result("no JSON here", stop_reason="end_turn")],
        judge_bundle=_minimal_judge_bundle(),
    )

    with pytest.raises(BackendError, match="no parseable review_verdicts"):
        await backend.run("prompt", system_prompt="critic system")

    assert len(fake_completion.calls) == 1


@pytest.mark.asyncio
async def test_max_completion_tokens_env_override_raises_the_cap(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A deployment can move the cap to fit its model without a code change —
    the knob this incident had no way to turn."""
    monkeypatch.setenv("CRITIC_AGENT_MAX_COMPLETION_TOKENS", "64000")
    review_json = (
        '{"review_verdicts": [{"target_proposal_msg_id": "p1", '
        '"verdict": "approve", "source": "critic", "reasoning": "ok"}]}'
    )
    backend, fake_completion = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[_anthropic_review_result(review_json)],
        judge_bundle=_minimal_judge_bundle(),
    )

    await backend.run("prompt", system_prompt="critic system")

    assert fake_completion.calls[0]["max_tokens"] == 64000


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("not-a-number", CRITIC_AGENT_MAX_COMPLETION_TOKENS),
        ("0", CRITIC_AGENT_MAX_COMPLETION_TOKENS),
        ("-5", CRITIC_AGENT_MAX_COMPLETION_TOKENS),
        ("  ", CRITIC_AGENT_MAX_COMPLETION_TOKENS),
        ("16000", 16000),
    ],
)
def test_max_completion_tokens_env_is_parsed_leniently(
    raw: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """A typo in the knob must cost the default, not the turn."""
    monkeypatch.setenv("CRITIC_AGENT_MAX_COMPLETION_TOKENS", raw)
    assert CriticAgentBackend._resolve_max_completion_tokens() == expected


@pytest.mark.asyncio
async def test_anthropic_protocol_traces_a_failed_completion(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A transport failure reaches the caller as LLMCallFailed, keeping its
    detail, and costs exactly one trace row."""
    backend, _ = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[LLMCallFailed("claude cli stream idle")],
        judge_bundle=_minimal_judge_bundle(),
    )
    with pytest.raises(LLMCallFailed, match="claude cli stream idle"):
        await backend.run("prompt", system_prompt="critic system")

    import json as _json

    from hyperloom.inference_optimizer.session.session_paths import llm_calls_path

    rows = [
        _json.loads(line)
        for line in llm_calls_path(fake_session_dir).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    critic_rows = [r for r in rows if r["component"] == "critic"]
    assert len(critic_rows) == 1
    assert critic_rows[0]["status"] == "error"
    assert critic_rows[0]["error_type"] == "LLMCallFailed"
    assert "claude cli stream idle" in critic_rows[0]["error_message"]


@pytest.mark.asyncio
async def test_anthropic_protocol_wraps_unexpected_transport_error(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend, _ = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[RuntimeError("boom")],
        judge_bundle=_minimal_judge_bundle(),
    )
    with pytest.raises(BackendError, match="Anthropic completion failed"):
        await backend.run("prompt", system_prompt="critic system")


def test_anthropic_protocol_refuses_a_host_without_a_usable_transport(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Constructing the backend must fail loudly rather than at the first review.

    The probe covers the transport, not just the credential: a subscription
    token with no claude CLI is exactly the case that used to pass here and
    fail later.
    """
    monkeypatch.setattr(f"{_CRITIC_MOD}.anthropic_transport_ready", lambda *_a, **_kw: False)
    with pytest.raises(BackendError, match="requires a usable Anthropic transport"):
        CriticAgentBackend(
            critic_agent_root=fake_critic_root,
            session_dir=fake_session_dir,
            protocol="anthropic",
            claude_model="claude-opus-4-8",
            codex_model="gpt-5.4",
            runtime_caller_factory=lambda: _make_fake_runtime(judge_bundle=_minimal_judge_bundle()),
        )


def test_anthropic_protocol_builds_no_review_client(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """llm_config owns the transport, so the critic holds nothing to leak."""
    backend, _ = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[],
        judge_bundle=_minimal_judge_bundle(),
    )
    assert backend._client is None


@pytest.mark.asyncio
async def test_raw_completion_max_turns_is_floored_by_the_real_backend(monkeypatch):
    """ClaudeBackend raises a literal max_turns=1 to its floor — Claude Code
    counts the model's own message as a turn, so 1 trips before any output.
    Pin the real value a raw-completion caller actually runs with."""
    from hyperloom.orchestrator.roles import claude as claude_mod

    seen: dict[str, int] = {}

    def fake_build_options(self, **kwargs):
        seen["max_turns"] = kwargs["max_turns"]
        raise RuntimeError("stop after options")

    monkeypatch.setattr(claude_mod.ClaudeBackend, "_build_options", fake_build_options)
    backend = claude_mod.ClaudeBackend(model="claude-opus-4-8", raw_completion=True, conversational=False)
    with pytest.raises(Exception):
        await backend.run("prompt", system_prompt="critic system", tools=[], max_turns=1)

    assert seen["max_turns"] == claude_mod._RAW_COMPLETION_MIN_MAX_TURNS
    assert seen["max_turns"] > 1


def test_accumulate_anthropic_usage_folds_tokens_and_tolerates_garbage():
    acc = {"input_tokens": 0, "output_tokens": 0}
    CriticAgentBackend._accumulate_anthropic_usage(acc, {"input_tokens": 3, "output_tokens": 4})
    CriticAgentBackend._accumulate_anthropic_usage(acc, {"input_tokens": "x", "output_tokens": None})
    CriticAgentBackend._accumulate_anthropic_usage(acc, None)
    assert acc["input_tokens"] == 3
    assert acc["output_tokens"] == 4


def test_accumulate_anthropic_usage_keeps_cache_counters_in_their_own_columns():
    """The judge bundle repeats across turns, so most of the input side arrives
    as cache reads. They stay split so a critic row can be compared with — and
    summed alongside — the orchestration rows ClaudeBackend writes."""
    acc = {"input_tokens": 0, "output_tokens": 0}
    CriticAgentBackend._accumulate_anthropic_usage(
        acc,
        {
            "input_tokens": 12,
            "cache_read_input_tokens": 4000,
            "cache_creation_input_tokens": 300,
            "output_tokens": 500,
        },
    )
    assert acc == {
        "input_tokens": 12,
        "output_tokens": 500,
        "cache_read_input_tokens": 4000,
        "cache_creation_input_tokens": 300,
    }


@pytest.mark.asyncio
async def test_anthropic_protocol_traces_cache_counters_separately(
    fake_critic_root: Path,
    fake_session_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A real subscription run reads most of its input from the prompt cache;
    folding it into input_tokens would make critic rows incomparable with the
    orchestration ones."""
    result = _anthropic_review_result(
        '{"review_verdicts": []}',
        usage={
            "input_tokens": 12,
            "cache_read_input_tokens": 4000,
            "cache_creation_input_tokens": 300,
            "output_tokens": 500,
        },
    )
    backend, _ = _make_anthropic_backend(
        fake_critic_root,
        fake_session_dir,
        monkeypatch,
        results=[result],
        judge_bundle=_minimal_judge_bundle(),
    )

    await backend.run("prompt", system_prompt="critic system")

    import json as _json

    from hyperloom.inference_optimizer.session.session_paths import llm_calls_path

    rows = [
        _json.loads(line)
        for line in llm_calls_path(fake_session_dir).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    critic_rows = [r for r in rows if r["component"] == "critic"]
    assert critic_rows
    row = critic_rows[0]
    assert row["input_tokens"] == 12
    assert row["cache_read_input_tokens"] == 4000
    assert row["cache_creation_input_tokens"] == 300
    assert row["output_tokens"] == 500
