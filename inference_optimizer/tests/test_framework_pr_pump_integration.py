# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Integration test for ``_pump_framework_pr_phase`` covering the full discover → Critic gate → enqueue path."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator import framework_agent_client as _fa_client
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType


# Cross-cutting framework parametrisation; add new frameworks here.
_FRAMEWORK_PARAMETRISATION: tuple[str, ...] = ("sglang", "vllm", "atom")


# Synthetic per-framework single-candidate batch fixtures (fa phase-discover shape).
_FRAMEWORK_CANDIDATES: dict[str, dict[str, Any]] = {
    "sglang": {
        "pr_url":   "https://github.com/sgl-project/sglang/pull/1",
        "diff_url": "https://github.com/sgl-project/sglang/pull/1.diff",
        "repo":     "sgl-project/sglang",
        "ref":      "perf/x",
        "title":    "perf: moe gemm fastpath",
        "framework": "sglang",
    },
    "vllm": {
        "pr_url":   "https://github.com/ROCm/vllm/pull/2",
        "diff_url": "https://github.com/ROCm/vllm/pull/2.diff",
        "repo":     "ROCm/vllm",
        "ref":      "perf/y",
        "title":    "perf: paged attention prefill",
        "framework": "vllm",
    },
    "atom": {
        "pr_url":   "https://github.com/ROCm/ATOM/pull/3",
        "diff_url": "https://github.com/ROCm/ATOM/pull/3.diff",
        "repo":     "ROCm/ATOM",
        "ref":      "perf/z",
        "title":    "perf: MTP scheduler + aiter fused_moe",
        "framework": "atom",
    },
}


class _StateStub:
    def __init__(self, framework: str = "sglang") -> None:
        self.phase = "FRAMEWORK_PR"
        self.framework_pr_phase_done = False
        self.framework_pr_discover_failures = 0
        self.framework_pr_batches: list[dict[str, Any]] = []
        self.framework_pr_phase_progress: list[dict[str, Any]] = []
        self.framework_pr_critic_decisions: list[dict[str, Any]] = []
        self.phase_history: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self.model = "test-model"
        self.framework = framework
        self.gpu_type = "MI300X"
        self.baseline_tput = 0.0
        self._saves = 0

    def save(self, _session_dir: Path) -> None:
        self._saves += 1


class _TasksStub:
    """Mimics the parts of ``Coordinator.tasks`` the pump touches."""

    def __init__(self) -> None:
        self._queued: list[Any] = []
        self._running: list[Any] = []
        self.created: list[dict[str, Any]] = []

    async def queued(self) -> list[Any]:
        return list(self._queued)

    async def running(self) -> list[Any]:
        return list(self._running)

    async def create_or_return_existing(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        # Mimic enqueue → queued so the pump's idempotency check fires next tick.
        self._queued.append(SimpleNamespace(
            kind=kwargs.get("kind"),
            task_id=f"t-{len(self.created)}",
        ))
        return self._queued[-1]


def _make_approve_intents(prompt: str) -> list[Intent]:
    m = re.search(r"msg_id=([a-f0-9]+)", prompt)
    msg_id = m.group(1) if m else "x"
    return [Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": msg_id,
            "verdict":                "approve",
            "reasoning":              "ok",
        },
    )]


def _make_reject_intents(prompt: str, reason: str = "out of scope") -> list[Intent]:
    m = re.search(r"msg_id=([a-f0-9]+)", prompt)
    msg_id = m.group(1) if m else "x"
    return [Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": msg_id,
            "verdict":                "reject",
            "reasoning":              reason,
        },
    )]


class _ScriptedCriticBackend:
    """Returns approve/reject in the order configured."""

    def __init__(self, verdicts: list[str]) -> None:
        self._verdicts = list(verdicts)
        self.call_count = 0

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> Any:
        self.call_count += 1
        v = self._verdicts.pop(0) if self._verdicts else "approve"
        intents = (
            _make_approve_intents(prompt)
            if v == "approve"
            else _make_reject_intents(prompt)
        )
        return SimpleNamespace(intents=intents, raw_text=f"({v})")


class _CoordinatorStub:
    """Glue stub carrying the attribute/method surface the pump touches."""

    _CRITIC_PRIORS_DECISION_TAIL = Coordinator._CRITIC_PRIORS_DECISION_TAIL
    _CRITIC_PRIORS_OUTCOME_TAIL = Coordinator._CRITIC_PRIORS_OUTCOME_TAIL
    _collect_framework_pr_priors = Coordinator._collect_framework_pr_priors
    _select_next_framework_pr_candidate = (
        Coordinator._select_next_framework_pr_candidate
    )
    _record_framework_pr_phase_done = (
        Coordinator._record_framework_pr_phase_done
    )
    _critic_review_framework_pr_candidate = (
        Coordinator._critic_review_framework_pr_candidate
    )
    _discover_next_framework_pr_batch = (
        Coordinator._discover_next_framework_pr_batch
    )
    _enqueue_framework_pr_task = Coordinator._enqueue_framework_pr_task

    def __init__(
        self,
        tmp_path: Path,
        critic: _ScriptedCriticBackend | None,
        *,
        framework: str = "sglang",
    ) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub(framework=framework)
        self.tasks = _TasksStub()
        self.framework_pr_discover_timeout_sec = 0.0
        self._framework = framework
        self.backends: dict[str, Any] = {}
        if critic is not None:
            self.backends["critic"] = critic

    def _framework_pr_discover_repo_urls(self, framework: str) -> list[str]:
        # Pin to a single repo so these scenarios keep one-batch/one-task accounting.
        return [_fa_client.repo_url_for_framework(framework or self._framework)]

    def _framework_pr_known_candidate_ids(self) -> set[str]:
        return Coordinator._framework_pr_known_candidate_ids(self)  # type: ignore[arg-type]

    def _framework_pr_tried_refs(self) -> list[str]:
        return Coordinator._framework_pr_tried_refs(self)  # type: ignore[arg-type]


def _pump(stub: _CoordinatorStub) -> None:
    asyncio.run(Coordinator._pump_framework_pr_phase(stub))  # type: ignore[arg-type]


# Scenario 1 — discover → approve → enqueue (parametrised across frameworks)
@pytest.mark.parametrize("framework", _FRAMEWORK_PARAMETRISATION)
def test_pump_happy_path_discover_approve_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
):
    """A single-candidate batch is approved and enqueued as one ``framework_pr`` task (parametrised sglang/vllm/atom)."""
    captured_framework: dict[str, str] = {}

    async def _discover(**kwargs: Any) -> dict[str, Any]:
        captured_framework["framework"] = str(kwargs.get("framework") or "")
        return {
            "batch_id": f"batch-{framework}",
            "candidates": [dict(_FRAMEWORK_CANDIDATES[framework])],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    critic = _ScriptedCriticBackend(["approve"])
    stub = _CoordinatorStub(tmp_path, critic, framework=framework)

    _pump(stub)

    assert captured_framework["framework"] == framework
    assert len(stub.shared_state.framework_pr_batches) == 1
    assert critic.call_count == 1
    assert len(stub.tasks.created) == 1
    assert stub.tasks.created[0]["kind"] == "framework_pr"
    enqueued = stub.tasks.created[0]["params"]["candidate"]
    assert enqueued["framework"] == framework
    assert enqueued["pr_url"] == _FRAMEWORK_CANDIDATES[framework]["pr_url"]
    # No progress rows yet (the executor writes those).
    assert stub.shared_state.framework_pr_phase_progress == []
    decisions = stub.shared_state.framework_pr_critic_decisions
    assert len(decisions) == 1
    assert decisions[0]["verdict"] == "approve"


def test_pump_integration_parametrised_over_all_three_frameworks():
    """G4 static guard: the parametrisation list covers exactly the three frameworks (and matches the fixture dict)."""
    assert set(_FRAMEWORK_PARAMETRISATION) == {"sglang", "vllm", "atom"}
    assert set(_FRAMEWORK_CANDIDATES.keys()) == set(_FRAMEWORK_PARAMETRISATION)


# Scenario 2 — discover → reject → skip → next on the next tick
def test_pump_reject_writes_progress_then_next_tick_picks_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Two-candidate batch: first rejected (``critic_denied``), next tick picks and enqueues the second."""
    discover_calls = SimpleNamespace(n=0)

    async def _discover(**_: Any) -> dict[str, Any]:
        discover_calls.n += 1
        return {
            "batch_id": "batch-2",
            "candidates": [
                {"pr_url": "https://example.com/pr/1", "repo": "a/b", "ref": "x1"},
                {"pr_url": "https://example.com/pr/2", "repo": "a/b", "ref": "x2"},
            ],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    critic = _ScriptedCriticBackend(["reject", "approve"])
    stub = _CoordinatorStub(tmp_path, critic)

    # Tick 1 — discover + Critic reject + progress row, no task.
    _pump(stub)
    assert discover_calls.n == 1
    assert critic.call_count == 1
    assert stub.tasks.created == []
    denied = [
        r for r in stub.shared_state.framework_pr_phase_progress
        if r.get("status") == "critic_denied"
    ]
    assert len(denied) == 1
    assert denied[0]["candidate_id"] == "https://example.com/pr/1"

    # Tick 2 — selector skips the denied candidate, Critic approves
    # the second, task is enqueued. discover is NOT called again
    # because the batch is not exhausted.
    _pump(stub)
    assert discover_calls.n == 1, "discover must not be called again — batch still has work"
    assert critic.call_count == 2
    assert len(stub.tasks.created) == 1
    # The enqueued task's candidate must be #2.
    enqueued_candidate = stub.tasks.created[0]["params"]["candidate"]
    assert enqueued_candidate["pr_url"] == "https://example.com/pr/2"


# Scenario 3 — already-in-flight task → pump no-op
def test_pump_is_noop_when_framework_pr_task_already_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """An already-running ``framework_pr`` task makes the pump return early without discover or Critic (idempotency guard)."""
    discover_called = SimpleNamespace(n=0)

    async def _discover(**_: Any) -> dict[str, Any]:
        discover_called.n += 1
        return {"batch_id": "unused", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    critic = _ScriptedCriticBackend(["approve"])
    stub = _CoordinatorStub(tmp_path, critic)
    # Pretend a framework_pr task is already running.
    stub.tasks._running.append(
        SimpleNamespace(kind="framework_pr", task_id="t-existing"),
    )

    _pump(stub)

    assert discover_called.n == 0
    assert critic.call_count == 0
    assert stub.tasks.created == []
    assert stub.shared_state.framework_pr_phase_done is False


# Scenario 4 — discover empty payload → phase_history row + phase_done
def test_pump_marks_phase_done_with_history_row_on_empty_discover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A clean empty discover payload flips ``framework_pr_phase_done`` and records a phase_history row; no Critic call."""
    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    critic = _ScriptedCriticBackend([])
    stub = _CoordinatorStub(tmp_path, critic)

    _pump(stub)

    assert stub.shared_state.framework_pr_phase_done is True
    assert critic.call_count == 0
    rows = [
        r for r in stub.shared_state.phase_history
        if r.get("event") == "framework_pr_phase_done"
    ]
    assert len(rows) == 1
    assert rows[0]["reason"] == "discover_empty_payload"
