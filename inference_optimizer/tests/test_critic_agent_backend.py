# Copyright Advanced Micro Devices, Inc. All rights reserved.

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

from inference_optimizer.orchestrator.backends import (
    CriticAgentBackend,
    RuntimeCall,
)
from inference_optimizer.orchestrator.backends.base import BackendError
from inference_optimizer.orchestrator.backends.critic_agent import (
    _extract_review_json,
)
from inference_optimizer.protocol.intent import IntentType


# Fakes — Codex client (mirrors test_p1_7_codex_backend.py)
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
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        text = self._replies.pop(0) if self._replies else ""
        return FakeResp(choices=[FakeChoice(message=FakeMessage(content=text))])


class FakeChat:
    def __init__(self, completions: FakeChatCompletions):
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, replies: list[str]):
        self.completions = FakeChatCompletions(replies)
        self.chat = FakeChat(self.completions)


# Fakes — runtime.cli subprocess
def _build_envelope_from_review(
    review: dict[str, Any], session_id: str,
) -> dict[str, Any]:
    """Mirror critic-agent's commit-review envelope construction so the fake produces the real runtime's shape."""
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
        intents.append({
            "intent_type": "send_message",
            "payload": {"topic": "advice", "body_md": body},
        })
    if not intents:
        intents.append({
            "intent_type": "send_message",
            "payload": {"topic": "heartbeat", "body_md": "ok (critic)"},
        })
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
            raise BackendError(
                f"fake critic-agent runtime.cli {call.phase} exited rc=2: "
                f"stderr='simulated failure'"
            )
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
        "# fake action prompt", encoding="utf-8",
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
    codex_replies: list[str],
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


def test_cortex_kb_url_propagated_to_runtime_env(
    fake_critic_root, fake_session_dir, monkeypatch
):
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    backend = CriticAgentBackend(
        critic_agent_root=fake_critic_root,
        session_dir=fake_session_dir,
        codex_client_factory=lambda: FakeOpenAIClient([]),
        runtime_caller_factory=lambda: (lambda call: None),
        cortex_kb_url="http://kb.local/",
    )
    env = backend._build_runtime_env()
    assert env["CORTEX_KB_URL"] == "http://kb.local/"


def test_explicit_cortex_kb_url_env_wins(
    fake_critic_root, fake_session_dir, monkeypatch
):
    monkeypatch.setenv("CORTEX_KB_URL", "http://from-env.local")
    backend = CriticAgentBackend(
        critic_agent_root=fake_critic_root,
        session_dir=fake_session_dir,
        codex_client_factory=lambda: FakeOpenAIClient([]),
        runtime_caller_factory=lambda: (lambda call: None),
        cortex_kb_url="http://from-flag.local",
    )
    env = backend._build_runtime_env()
    assert env["CORTEX_KB_URL"] == "http://from-env.local"


def test_no_cortex_kb_url_leaves_env_unset(
    fake_critic_root, fake_session_dir, monkeypatch
):
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    backend = CriticAgentBackend(
        critic_agent_root=fake_critic_root,
        session_dir=fake_session_dir,
        codex_client_factory=lambda: FakeOpenAIClient([]),
        runtime_caller_factory=lambda: (lambda call: None),
    )
    env = backend._build_runtime_env()
    assert "CORTEX_KB_URL" not in env


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
            runtime_caller_factory=lambda: (lambda call: None),
        )


def test_construct_invalid_kb_mode(fake_critic_root: Path, fake_session_dir: Path):
    with pytest.raises(BackendError, match="kb_mode"):
        CriticAgentBackend(
            critic_agent_root=fake_critic_root,
            session_dir=fake_session_dir,
            kb_mode="bogus",  # type: ignore[arg-type]
            codex_client_factory=lambda: FakeOpenAIClient([]),
            runtime_caller_factory=lambda: (lambda call: None),
        )


def test_construct_no_creds_no_factory_raises(monkeypatch, tmp_path: Path):
    """No codex_client_factory and no Codex creds → construction fails fast."""
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = tmp_path / "critic-agent"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "cli.py").write_text("# stub", encoding="utf-8")
    sd = tmp_path / "sd"
    sd.mkdir()
    with pytest.raises(BackendError, match="ANTHROPIC_AUTH_TOKEN"):
        CriticAgentBackend(
            critic_agent_root=root,
            session_dir=sd,
            runtime_caller_factory=lambda: (lambda call: None),
        )


# Case 1: Single proposal → one approve verdict matching the msg_id
@pytest.mark.asyncio
async def test_single_proposal_yields_matching_verdict(
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "Llama-3.1-8B", "framework": "sglang"},
        "missing_context": [],
        "required_context": [],
        "proposals": [{
            "msg_id": "abc1",
            "from_agent": "orchestration",
            "action_name": "baseline",
            "predicted_gain_pct": 0.0,
            "payload": {"action_name": "baseline"},
        }],
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
        fake_critic_root, fake_session_dir,
        codex_replies=[reply], judge_bundle=judge_bundle,
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

    # Default kb_mode injected into the runtime env.
    env = runtime_calls[0].env
    assert env["CRITIC_KB_CLIENT_MODE"] == "inmemory"
    assert env["CRITIC_SESSION_MEMORY_DIR"].endswith("critic-session-memory")


# Case 2: Multiple proposals → one verdict each
@pytest.mark.asyncio
async def test_multiple_proposals_yield_one_verdict_each(
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [
            {"msg_id": "p1", "from_agent": "orchestration", "action_name": "baseline",
             "payload": {}, "predicted_gain_pct": 0.0},
            {"msg_id": "p2", "from_agent": "orchestration", "action_name": "params",
             "payload": {}, "predicted_gain_pct": 1.0},
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
        fake_critic_root, fake_session_dir,
        codex_replies=[reply], judge_bundle=judge_bundle,
    )
    res = await backend.run("prompt")
    verdicts = [
        i for i in res.intents if i.type == IntentType.REVIEW_VERDICT
    ]
    assert len(verdicts) == 2
    assert {v.payload["target_proposal_msg_id"] for v in verdicts} == {"p1", "p2"}
    assert {v.payload["verdict"] for v in verdicts} == {"approve", "advise"}


# Case 3: Empty inbox → heartbeat (LLM never called)
@pytest.mark.asyncio
async def test_empty_proposals_yields_heartbeat_no_llm(
    fake_critic_root: Path, fake_session_dir: Path,
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
        fake_critic_root, fake_session_dir,
        codex_replies=[], judge_bundle=judge_bundle,
    )
    res = await backend.run("prompt with no proposals")

    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.SEND_MESSAGE
    assert res.intents[0].payload["topic"] == "heartbeat"
    # LLM short-circuited.
    assert client.completions.calls == []


# Case 4: LLM returns garbage → empty review → heartbeat
@pytest.mark.asyncio
async def test_unparseable_llm_reply_falls_back_to_heartbeat(
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [{
            "msg_id": "px", "from_agent": "orchestration",
            "action_name": "baseline", "payload": {},
            "predicted_gain_pct": 0.0,
        }],
        "kb_priors_by_proposal": {"px": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    backend, _ = _make_backend(
        fake_critic_root, fake_session_dir,
        codex_replies=["I am thinking… no JSON here."],
        judge_bundle=judge_bundle,
    )
    res = await backend.run("prompt")
    # Empty review_verdicts → commit-review fake falls back to heartbeat.
    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.SEND_MESSAGE
    assert res.intents[0].payload["topic"] == "heartbeat"


# Case 5: required_context non-empty → needs_review + critic_unavailable
@pytest.mark.asyncio
async def test_missing_critical_context_yields_needs_review(
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {},
        "proposals": [{
            "msg_id": "p1", "from_agent": "orchestration",
            "action_name": "baseline", "payload": {},
            "predicted_gain_pct": 0.0,
        }],
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
        fake_critic_root, fake_session_dir,
        codex_replies=[reply], judge_bundle=judge_bundle,
    )
    res = await backend.run("prompt")
    assert len(res.intents) == 1
    p = res.intents[0].payload
    assert p["verdict"] == "needs_review"
    assert p["source"] == "critic_unavailable"
    # Backend surfaces the runtime's reason in metadata for the Coordinator.
    assert res.metadata["kb_read_skipped_reason"] == "missing_critical_context"


# Case 6: subprocess exit code 2 → BackendError
@pytest.mark.asyncio
async def test_prepare_review_subprocess_failure_raises(
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {"proposals": []}  # never read because we fail first.
    backend, _ = _make_backend(
        fake_critic_root, fake_session_dir,
        codex_replies=[], judge_bundle=judge_bundle,
        fail_phase="prepare-review",
    )
    with pytest.raises(BackendError, match=r"rc=2"):
        await backend.run("prompt")


@pytest.mark.asyncio
async def test_commit_review_subprocess_failure_raises(
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [{
            "msg_id": "z", "from_agent": "orchestration",
            "action_name": "baseline", "payload": {},
            "predicted_gain_pct": 0.0,
        }],
        "kb_priors_by_proposal": {"z": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    reply = '{"review_verdicts": [{"target_proposal_msg_id": "z", "verdict": "approve"}]}'
    backend, _ = _make_backend(
        fake_critic_root, fake_session_dir,
        codex_replies=[reply], judge_bundle=judge_bundle,
        fail_phase="commit-review",
    )
    with pytest.raises(BackendError, match=r"commit-review.*rc=2"):
        await backend.run("prompt")


# Case 7: kb_mode=live + kb_unreachable → still emits, surfaces reason
@pytest.mark.asyncio
async def test_kb_unreachable_still_emits_verdict(
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [{
            "msg_id": "p1", "from_agent": "orchestration",
            "action_name": "baseline", "payload": {},
            "predicted_gain_pct": 0.0,
        }],
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
        fake_critic_root, fake_session_dir,
        codex_replies=[reply], judge_bundle=judge_bundle,
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
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {"proposals": []}
    backend, _ = _make_backend(
        fake_critic_root, fake_session_dir,
        codex_replies=[], judge_bundle=judge_bundle,
        kb_mode="live", kb_env={},
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
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [{
            "msg_id": "px", "from_agent": "orchestration",
            "action_name": "baseline", "payload": {},
            "predicted_gain_pct": 0.0,
        }],
        "kb_priors_by_proposal": {"px": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    reply = '{"review_verdicts": [{"target_proposal_msg_id": "px", "verdict": "approve"}]}'
    backend, _ = _make_backend(
        fake_critic_root, fake_session_dir,
        codex_replies=[reply, reply], judge_bundle=judge_bundle,
    )
    await backend.run("turn 1")
    await backend.run("turn 2")
    assert (fake_session_dir / "critic-workdir" / "000000" / "request.json").is_file()
    assert (fake_session_dir / "critic-workdir" / "000001" / "request.json").is_file()


# Output instructions are appended (verifies prompt construction)
@pytest.mark.asyncio
async def test_user_prompt_includes_judge_bundle_and_instructions(
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "m", "framework": "sglang"},
        "proposals": [{
            "msg_id": "abc", "from_agent": "orchestration",
            "action_name": "baseline", "payload": {},
            "predicted_gain_pct": 0.0,
        }],
        "kb_priors_by_proposal": {"abc": []},
        "kb_read_skipped_reason": None,
        "review_constraints": {},
        "notes": [],
        "missing_context": [],
        "required_context": [],
    }
    reply = '{"review_verdicts": [{"target_proposal_msg_id": "abc", "verdict": "approve"}]}'
    backend, client = _make_backend(
        fake_critic_root, fake_session_dir,
        codex_replies=[reply], judge_bundle=judge_bundle,
    )
    await backend.run("ignored", system_prompt="you are critic")
    call = client.completions.calls[0]
    assert call["model"] == "gpt-5.4"
    assert call["messages"][0] == {"role": "system", "content": "you are critic"}
    user_text = call["messages"][1]["content"]
    assert "JUDGE BUNDLE" in user_text
    assert "OUTPUT FORMAT" in user_text
    assert '"abc"' in user_text  # proposal msg_id from judge bundle


# Static context propagation — root-cause fix for the "every verdict is needs_review/critic_unavailable" loop bug; backend sources model/framework from manifest.json or explicit static_context.
def _write_manifest(session_dir: Path, payload: dict[str, Any]) -> Path:
    """Write a minimal manifest.json the backend can ingest."""
    target = session_dir / "manifest.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


@pytest.mark.asyncio
async def test_run_populates_request_context_from_manifest(
    fake_critic_root: Path, fake_session_dir: Path,
):
    _write_manifest(fake_session_dir, {
        "schema_version": 1,
        "session_id": "sess-1",
        "model_name": "Llama-3.1-8B-Instruct",
        "model_path": "/models/llama-3.1-8b",
        "framework": "sglang",
        "gpu_type": "mi300x",
        "tp": 8,
        "workload": {
            "isl": 1024, "osl": 1024, "max_model_len": 4096,
            "precision": "fp8", "conc": 64,
        },
    })
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {"model": "Llama-3.1-8B-Instruct", "framework": "sglang"},
        "proposals": [{
            "msg_id": "p1", "from_agent": "orchestration",
            "action_name": "baseline", "payload": {},
            "predicted_gain_pct": 0.0,
        }],
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
        fake_critic_root, fake_session_dir,
        codex_replies=[reply], judge_bundle=judge_bundle,
        runtime_calls=runtime_calls,
    )
    await backend.run("prompt")

    # Read back the persisted request.json to verify context came from the manifest.
    request = json.loads(
        (fake_session_dir / "critic-workdir" / "000000" / "request.json")
        .read_text(encoding="utf-8")
    )
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
    fake_critic_root: Path, fake_session_dir: Path,
):
    # Manifest says sglang, but the explicit static_context overrides it.
    _write_manifest(fake_session_dir, {
        "schema_version": 1,
        "model_name": "ignored-by-test",
        "framework": "sglang",
    })
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
        static_context={"model": "explicit-m", "framework": "vllm",
                        "gpu_type": "mi355x"},
    )
    await backend.run("prompt")
    request = json.loads(
        (fake_session_dir / "critic-workdir" / "000000" / "request.json")
        .read_text(encoding="utf-8")
    )
    assert request["context"] == {
        "model": "explicit-m",
        "framework": "vllm",
        "gpu_type": "mi355x",
    }


@pytest.mark.asyncio
async def test_missing_manifest_falls_back_to_empty_context(
    fake_critic_root: Path, fake_session_dir: Path, caplog,
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
    with caplog.at_level("WARNING", logger="inference_optimizer.orchestrator.backends.critic_agent"):
        backend = CriticAgentBackend(
            critic_agent_root=fake_critic_root,
            session_dir=fake_session_dir,
            codex_client_factory=lambda: FakeOpenAIClient([]),
            runtime_caller_factory=lambda: fake_caller,
        )
    assert backend._static_context == {}
    assert any(
        "manifest.json not found" in rec.getMessage()
        for rec in caplog.records
    )

    await backend.run("prompt")
    request = json.loads(
        (fake_session_dir / "critic-workdir" / "000000" / "request.json")
        .read_text(encoding="utf-8")
    )
    assert request["context"] == {}


@pytest.mark.asyncio
async def test_malformed_manifest_logs_warning_and_falls_back(
    fake_critic_root: Path, fake_session_dir: Path, caplog,
):
    # Corrupt manifest JSON → backend logs + falls back, doesn't crash the boot.
    (fake_session_dir / "manifest.json").write_text(
        "{ this is not json", encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="inference_optimizer.orchestrator.backends.critic_agent"):
        backend = CriticAgentBackend(
            critic_agent_root=fake_critic_root,
            session_dir=fake_session_dir,
            codex_client_factory=lambda: FakeOpenAIClient([]),
            runtime_caller_factory=lambda: (lambda call: None),
        )
    assert backend._static_context == {}
    assert any(
        "failed to load manifest.json" in rec.getMessage()
        for rec in caplog.records
    )


def test_load_static_context_skips_unknown_and_empty_fields(
    fake_critic_root: Path, fake_session_dir: Path,
):
    # Empty / missing values are skipped upstream so the keys never appear in the JSON.
    _write_manifest(fake_session_dir, {
        "schema_version": 1,
        "model_name": "m",
        "framework": "",
        "gpu_type": None,
        "tp": 0,
        "workload": {"isl": 1024, "precision": ""},
    })
    backend = CriticAgentBackend(
        critic_agent_root=fake_critic_root,
        session_dir=fake_session_dir,
        codex_client_factory=lambda: FakeOpenAIClient([]),
        runtime_caller_factory=lambda: (lambda call: None),
    )
    ctx = backend._static_context
    assert ctx == {"model": "m", "workload": {"isl": 1024}}


# Diagnostic plumbing — required_context surfaces in metadata + log line.
@pytest.mark.asyncio
async def test_required_context_surfaces_in_metadata(
    fake_critic_root: Path, fake_session_dir: Path,
):
    judge_bundle = {
        "kind": "coordinator_inbox",
        "merged_context": {},
        "proposals": [{
            "msg_id": "p1", "from_agent": "orchestration",
            "action_name": "baseline", "payload": {},
            "predicted_gain_pct": 0.0,
        }],
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
        fake_critic_root, fake_session_dir,
        codex_replies=[reply], judge_bundle=judge_bundle,
    )
    res = await backend.run("prompt")
    assert res.metadata["required_context"] == ["model", "framework"]
    assert backend.calls[-1]["required_context"] == ["model", "framework"]


# Merged from test_p2_critic_agent_e2e.py

"""End-to-end test: real critic-agent runtime + mocked Codex + Coordinator.

This shells out to the real ``critic-agent/runtime/cli.py`` (only Codex is
faked). Marker ``critic_agent_e2e`` lets devs without the checkout skip it.
"""


import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.cli import _resolve_critic_agent_root
from inference_optimizer.orchestrator.backends import (
    CriticAgentBackend,
    MockBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.paths import make_session_dir


pytestmark = pytest.mark.critic_agent_e2e


# Fake Codex (only the LLM layer; the runtime is real)
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
            user_text, re.DOTALL,
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
                verdicts.append({
                    "target_proposal_msg_id": msg_id,
                    "verdict": "approve",
                    "source": "critic",
                    "reasoning": "deterministic e2e fixture — auto-approve",
                    "confidence": "medium",
                })
        body = json.dumps({"review_verdicts": verdicts})
        return _Resp(choices=[_Choice(message=_Msg(content=f"```json\n{body}\n```"))])


class _DeterministicChat:
    def __init__(self, completions):
        self.completions = completions


class _DeterministicClient:
    def __init__(self):
        self.completions = _DeterministicCompletions()
        self.chat = _DeterministicChat(self.completions)


# Fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    from .conftest import seed_target_analysis_marker
    seed_target_analysis_marker(sd)
    return sd


@pytest.fixture
def critic_agent_root() -> Path:
    """Locate the real critic-agent checkout. Skip gracefully if absent."""
    root = _resolve_critic_agent_root()
    if root is None:
        pytest.skip(
            "critic-agent runtime not found — set CRITIC_AGENT_ROOT or "
            "place critic-agent/ next to inference_optimizer/"
        )
    return root


def _heartbeat() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


# E2E: scripted Orchestration → real CriticAgentBackend → approved
@pytest.mark.asyncio
async def test_critic_agent_real_runtime_clears_proposal(
    session_dir: Path, critic_agent_root: Path,
):
    """Orchestration proposes baseline → real runtime emits review_verdict{approve} → Coordinator materializes the task."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })

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
        "kernel":     MockKernelBackend(),
        "critic":     critic_backend,
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
        assert all(
            v.from_agent == "critic" for v in approved
        ), "verdicts must originate from critic role"
        assert all(
            "deterministic e2e fixture" in (v.payload.get("reasoning") or "")
            for v in approved
        ), (
            f"verdict reasoning should match the fake LLM's text "
            f"('deterministic e2e fixture — auto-approve'); a 'mock critic' "
            f"reasoning would mean MockCriticBackend ran instead. "
            f"Payloads: {[v.payload for v in verdicts]}"
        )

        # Coordinator turned the approved proposal into a decision.
        assert any(d.payload.get("kind") == "approved_proposal" for d in decisions), \
            f"approved proposal didn't materialise into a decision: {[d.payload for d in decisions]}"

    finally:
        await c.stop()

    # --- Filesystem assertions: real runtime wrote session memory ---
    workdir = session_dir / "critic-workdir"
    assert workdir.is_dir()
    turn0 = workdir / "000000"
    for fname in ("request.json", "judge_bundle.json", "review.json", "emit.json"):
        assert (turn0 / fname).is_file(), f"missing per-turn artefact {fname}"

    memory_root = session_dir / "critic-session-memory"
    assert memory_root.is_dir(), \
        f"session memory dir missing: {memory_root}"
    session_memories = list(memory_root.iterdir())
    assert session_memories, \
        f"session memory dir is empty under {memory_root}"
    sm_dir = session_memories[0]
    # The runtime stamps decisions.jsonl + reviewed_msg_ids.json once a verdict commits.
    assert (sm_dir / "decisions.jsonl").is_file(), \
        f"decisions.jsonl missing under {sm_dir} (entries: {list(sm_dir.iterdir())})"
    assert (sm_dir / "reviewed_msg_ids.json").is_file(), \
        f"reviewed_msg_ids.json missing under {sm_dir}"

    # The Coordinator generates the msg_id, so just check the file is non-trivial.
    reviewed_raw = (sm_dir / "reviewed_msg_ids.json").read_text(encoding="utf-8")
    reviewed = json.loads(reviewed_raw)
    assert reviewed, f"reviewed_msg_ids.json should be non-empty, got {reviewed!r}"


@pytest.mark.asyncio
async def test_critic_agent_heartbeat_when_no_proposal(
    session_dir: Path, critic_agent_root: Path,
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
        "kernel":     MockKernelBackend(),
        "critic":     critic_backend,
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(2)
        verdicts = await c.bus.tail(topic="review_verdict")
        assert not verdicts, f"unexpected verdicts in heartbeat-only run: {verdicts}"

        # Confirm the heartbeat path by checking the LLM was NOT called (zero proposals).
        client = critic_backend._client  # type: ignore[attr-defined]
        assert client.completions.calls == [], \
            f"LLM should be skipped when proposals are empty; calls={client.completions.calls}"
    finally:
        await c.stop()
