"""Unit tests for :class:`CriticAgentBackend`.

The backend drives a 3-step loop: prepare-review (subprocess) → Codex
chat completion → commit-review (subprocess). Both subprocess calls
are bypassed via ``runtime_caller_factory`` and Codex is bypassed via
``codex_client_factory`` so these tests don't pay for real Python /
network startup.

Cases (mirrors the plan):

1. Single proposal → exactly one review_verdict matching the msg_id.
2. Multiple proposals → one verdict each.
3. Empty inbox → heartbeat envelope (commit-review's own fallback,
   reproduced here by the fake).
4. LLM returns no parseable JSON → empty review_verdicts → heartbeat.
5. judge_bundle.required_context non-empty → all verdicts are
   needs_review with source=critic_unavailable.
6. prepare-review subprocess exits non-zero → BackendError.
7. kb_mode=live with kb_unreachable in judge bundle → verdicts still
   emitted, metadata records the reason.
8. Construction errors: missing critic-agent root, bad kb_mode.
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
from inference_optimizer.orchestrator.intent_parser import IntentType


# ---------------------------------------------------------------------------
# Fakes — Codex client (mirrors test_p1_7_codex_backend.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Fakes — runtime.cli subprocess
# ---------------------------------------------------------------------------
def _build_envelope_from_review(
    review: dict[str, Any], session_id: str,
) -> dict[str, Any]:
    """Mirror critic-agent's commit-review envelope construction.

    We replicate the relevant parts of
    ``critic-agent/runtime/decision_reviewer._commit_coordinator_inbox`` /
    ``intent_envelope.build_review_verdict_intent`` so the fake produces
    the same shape the real runtime would. This lets tests exercise the
    full review.json -> intent_envelope path without spawning subprocess.
    """
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
    """Return a ``RuntimeCaller`` that fakes prepare-review / commit-review.

    Optional ``fail_phase`` injects a :class:`BackendError` to simulate a
    non-zero subprocess exit. ``capture`` collects every call for asserts.
    """

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_critic_root(tmp_path: Path) -> Path:
    """Create a minimal critic-agent root with a stub runtime/cli.py.

    The backend's __post_init__ checks ``runtime/cli.py`` exists; since we
    inject ``runtime_caller_factory`` we don't actually invoke it.
    """
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
    )
    return backend, fake_client


# ---------------------------------------------------------------------------
# _extract_review_json
# ---------------------------------------------------------------------------
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


def test_extract_review_json_returns_none_when_key_absent():
    assert _extract_review_json('```json\n{"intents": []}\n```') is None


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
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
    """When codex_client_factory is None and there are no Codex creds in env,
    construction must fail fast."""
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


# ---------------------------------------------------------------------------
# Case 1: Single proposal → one approve verdict matching the msg_id
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Case 2: Multiple proposals → one verdict each
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Case 3: Empty inbox → heartbeat (LLM never called)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Case 4: LLM returns garbage → empty review → heartbeat
# ---------------------------------------------------------------------------
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
    # commit-review fake produced a heartbeat fallback because the review
    # had no review_verdicts.
    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.SEND_MESSAGE
    assert res.intents[0].payload["topic"] == "heartbeat"


# ---------------------------------------------------------------------------
# Case 5: required_context non-empty → needs_review + critic_unavailable
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Case 6: subprocess exit code 2 → BackendError
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Case 7: kb_mode=live + kb_unreachable → still emits, surfaces reason
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Multi-turn: counter increments, workdirs scoped per-turn
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Output instructions are appended (verifies prompt construction)
# ---------------------------------------------------------------------------
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
