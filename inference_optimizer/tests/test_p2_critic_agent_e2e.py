"""End-to-end test: real critic-agent runtime + mocked Codex + Coordinator.

Unlike :mod:`test_critic_agent_backend` (which fakes the runtime via
``runtime_caller_factory``), this test actually shells out to
``python -m runtime.cli prepare-review/commit-review`` against the real
``critic-agent/runtime/cli.py`` checked into the repo. Only the Codex
LLM is faked, with a deterministic reply that produces a
``review_verdict{verdict=approve}`` for whichever proposal the
Coordinator surfaces in the inbox.

Marker: ``critic_agent_e2e`` — devs without a critic-agent checkout in
their tree skip via ``pytest -m 'not critic_agent_e2e'``.

Verifies:

* ``pending_proposals`` get cleared with non-mock verdicts.
* The bus event log shows ``topic=review_verdict`` rows with
  ``from=critic`` and ``payload.source == "critic"``.
* ``<session_dir>/critic-session-memory/<session_id>/`` is populated
  with ``decisions.jsonl`` + ``reviewed_msg_ids.json`` (proof the
  runtime's session memory is wired through correctly).
* ``<session_dir>/critic-workdir/000000/{request,judge_bundle,review,emit}.json``
  exist (per-turn audit trail).
"""

from __future__ import annotations

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
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.paths import make_session_dir


pytestmark = pytest.mark.critic_agent_e2e


# ---------------------------------------------------------------------------
# Fake Codex (only the LLM layer; the runtime is real)
# ---------------------------------------------------------------------------
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
        # Pull the judge bundle JSON out of the user prompt. The backend
        # wraps it between markers we can pin against.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# E2E: scripted Orchestration → real CriticAgentBackend → approved
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_critic_agent_real_runtime_clears_proposal(
    session_dir: Path, critic_agent_root: Path,
):
    """Orchestration proposes baseline → real critic-agent runtime emits
    review_verdict{verdict=approve, source=critic} → Coordinator
    materializes the task."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })

    critic_backend = CriticAgentBackend(
        critic_agent_root=critic_agent_root,
        session_dir=session_dir,
        codex_model="gpt-5.4",
        codex_client_factory=_DeterministicClient,
        kb_mode="inmemory",
        # IMPORTANT: do NOT pass runtime_caller_factory — we want the
        # real subprocess path here.
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
        # tick 1: orchestration emits propose_action
        # tick 2: critic-agent runtime processes the proposal, emits
        #         review_verdict{approve}
        # tick 3: Coordinator materializes the task; dispatcher runs
        #         baseline; delegated_result lands.
        await c.tick(3)

        proposals = await c.bus.tail(topic="proposal")
        verdicts = await c.bus.tail(topic="review_verdict")
        decisions = await c.bus.tail(topic="decision")

        assert len(proposals) >= 1, f"no proposals on bus, got {proposals}"

        # The verdict came from the real runtime, not the mock. The
        # Coordinator strips most payload fields when mirroring to the bus
        # (only target / verdict / reasoning survive), so we identify the
        # real path by:
        #   1. ``from=critic`` (true for both mock + real).
        #   2. ``reasoning`` text matches our fake LLM's deterministic
        #      response — distinct from MockCriticBackend's auto-approve
        #      reasoning string.
        #   3. Filesystem evidence: critic-workdir/ + critic-session-memory/
        #      get populated only by the real runtime (asserted below).
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
    # The runtime stamps decisions.jsonl + reviewed_msg_ids.json once a
    # verdict commits.
    assert (sm_dir / "decisions.jsonl").is_file(), \
        f"decisions.jsonl missing under {sm_dir} (entries: {list(sm_dir.iterdir())})"
    assert (sm_dir / "reviewed_msg_ids.json").is_file(), \
        f"reviewed_msg_ids.json missing under {sm_dir}"

    # The reviewed_msg_ids file should contain the approved proposal's
    # msg_id (we don't know the exact msg_id ahead of time — the
    # Coordinator generates it — so we just check the file is non-trivial).
    reviewed_raw = (sm_dir / "reviewed_msg_ids.json").read_text(encoding="utf-8")
    reviewed = json.loads(reviewed_raw)
    assert reviewed, f"reviewed_msg_ids.json should be non-empty, got {reviewed!r}"


@pytest.mark.asyncio
async def test_critic_agent_heartbeat_when_no_proposal(
    session_dir: Path, critic_agent_root: Path,
):
    """No proposals in the inbox → real runtime falls back to heartbeat
    envelope. Verifies the LLM is short-circuited and the runtime still
    produces a valid intent envelope."""
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

        # critic backend was called at least once (it ticks every loop) and
        # produced a heartbeat — confirm by checking that the LLM was NOT
        # called (judge bundle had zero proposals).
        client = critic_backend._client  # type: ignore[attr-defined]
        assert client.completions.calls == [], \
            f"LLM should be skipped when proposals are empty; calls={client.completions.calls}"
    finally:
        await c.stop()
