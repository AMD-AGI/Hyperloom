# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Integration test for ``_pump_framework_agent_phase`` (converged async gate).

The pump discovers a batch then submits the chosen candidate as a normal
``framework_agent`` proposal; the async Critic verdict (handled on a later tick)
drives the apply/author enqueue or the ``critic_denied`` row. The pump itself
no longer calls the Critic synchronously nor creates a ``framework_agent`` task
inline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator import framework_agent_client as _fa_client
from hyperloom.orchestrator.coordinator import Coordinator


# Cross-cutting framework parametrisation; add new frameworks here.
_FRAMEWORK_PARAMETRISATION: tuple[str, ...] = ("sglang", "vllm", "atom")


# Synthetic per-framework single-candidate batch fixtures (fa phase-discover shape).
_FRAMEWORK_CANDIDATES: dict[str, dict[str, Any]] = {
    "sglang": {
        "pr_url": "https://github.com/sgl-project/sglang/pull/1",
        "diff_url": "https://github.com/sgl-project/sglang/pull/1.diff",
        "repo": "sgl-project/sglang",
        "ref": "perf/x",
        "title": "perf: moe gemm fastpath",
        "framework": "sglang",
    },
    "vllm": {
        "pr_url": "https://github.com/ROCm/vllm/pull/2",
        "diff_url": "https://github.com/ROCm/vllm/pull/2.diff",
        "repo": "ROCm/vllm",
        "ref": "perf/y",
        "title": "perf: paged attention prefill",
        "framework": "vllm",
    },
    "atom": {
        "pr_url": "https://github.com/ROCm/ATOM/pull/3",
        "diff_url": "https://github.com/ROCm/ATOM/pull/3.diff",
        "repo": "ROCm/ATOM",
        "ref": "perf/z",
        "title": "perf: MTP scheduler + aiter fused_moe",
        "framework": "atom",
    },
}


class _StateStub:
    def __init__(self, framework: str = "sglang") -> None:
        self.phase = "FRAMEWORK_AGENT"
        self.framework_agent_phase_done = False
        self.framework_agent_authoring_enabled = False
        self.framework_agent_discover_failures = 0
        self.framework_agent_empty_discoveries = 0
        self.framework_agent_batches: list[dict[str, Any]] = []
        self.framework_agent_phase_progress: list[dict[str, Any]] = []
        self.framework_agent_critic_decisions: list[dict[str, Any]] = []
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
        self._queued.append(
            SimpleNamespace(
                kind=kwargs.get("kind"),
                task_id=f"t-{len(self.created)}",
            )
        )
        return self._queued[-1]


class _BusStub:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def append_and_seq(self, msg: Any) -> Any:
        self.messages.append(msg)
        return msg


class _CoordinatorStub:
    """Glue stub carrying the attribute/method surface the pump touches."""

    _CRITIC_PRIORS_DECISION_TAIL = Coordinator._CRITIC_PRIORS_DECISION_TAIL
    _CRITIC_PRIORS_OUTCOME_TAIL = Coordinator._CRITIC_PRIORS_OUTCOME_TAIL
    _MAX_REPEATED_REVIEW_SUBMISSIONS = Coordinator._MAX_REPEATED_REVIEW_SUBMISSIONS
    _collect_framework_agent_candidate_priors = Coordinator._collect_framework_agent_candidate_priors
    _framework_candidate_key = staticmethod(Coordinator._framework_candidate_key)
    _framework_processed_candidate_keys = Coordinator._framework_processed_candidate_keys
    _stamp_framework_progress = Coordinator._stamp_framework_progress
    _unprocessed_framework_agent_candidates = Coordinator._unprocessed_framework_agent_candidates
    _select_next_framework_agent_candidate = Coordinator._select_next_framework_agent_candidate
    _select_best_framework_agent_candidate = Coordinator._select_best_framework_agent_candidate
    _record_framework_agent_phase_done = Coordinator._record_framework_agent_phase_done
    _submit_framework_agent_candidate_for_review = Coordinator._submit_framework_agent_candidate_for_review
    _record_framework_agent_critic_denied = Coordinator._record_framework_agent_critic_denied
    _discover_next_framework_batch = Coordinator._discover_next_framework_batch
    _enqueue_framework_agent_task = Coordinator._enqueue_framework_agent_task

    def __init__(self, tmp_path: Path, *, framework: str = "sglang") -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub(framework=framework)
        self.tasks = _TasksStub()
        self.bus = _BusStub()
        self.state = SimpleNamespace(pending_proposals={})
        self.framework_agent_discover_timeout_sec = 0.0
        self._framework = framework
        self.backends: dict[str, Any] = {}

    async def _record_observation(self, *_a: Any, **_k: Any) -> None:
        return None

    async def _rank_framework_agent_candidates_llm(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        # Hermetic: force the deterministic discovery-order fallback.
        return None

    async def _audit_framework_agent_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        v = getattr(self, "_audit_verdict", None) or {"recommended_next_step": ""}
        try:
            candidate["_audit"] = v
        except Exception:
            pass
        return v

    def _framework_agent_discover_repo_urls(self, framework: str) -> list[str]:
        return [_fa_client.repo_url_for_framework(framework or self._framework)]

    def _framework_known_candidate_ids(self) -> set[str]:
        return Coordinator._framework_known_candidate_ids(self)  # type: ignore[arg-type]

    def _framework_tried_refs(self) -> list[str]:
        return Coordinator._framework_tried_refs(self)  # type: ignore[arg-type]


def _pump(stub: _CoordinatorStub) -> None:
    asyncio.run(Coordinator._pump_framework_agent_phase(stub))  # type: ignore[arg-type]


def _framework_agent_pendings(stub: _CoordinatorStub) -> list[Any]:
    return [
        p
        for p in stub.state.pending_proposals.values()
        if getattr(p, "action_name", "") == "framework_agent"
    ]


# Scenario 1 — discover → submit candidate proposal (parametrised across frameworks)
@pytest.mark.parametrize("framework", _FRAMEWORK_PARAMETRISATION)
def test_pump_happy_path_discover_submits_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
):
    """A single-candidate batch is submitted as one ``framework_agent`` proposal (no task, no sync Critic call)."""
    captured_framework: dict[str, str] = {}

    async def _discover(**kwargs: Any) -> dict[str, Any]:
        captured_framework["framework"] = str(kwargs.get("framework") or "")
        return {
            "batch_id": f"batch-{framework}",
            "candidates": [dict(_FRAMEWORK_CANDIDATES[framework])],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _CoordinatorStub(tmp_path, framework=framework)

    _pump(stub)

    assert captured_framework["framework"] == framework
    assert len(stub.shared_state.framework_agent_batches) == 1
    # No task is enqueued by the pump; the async verdict drives that later.
    assert stub.tasks.created == []
    pendings = _framework_agent_pendings(stub)
    assert len(pendings) == 1
    payload = pendings[0].payload
    assert payload["candidate"]["pr_url"] == _FRAMEWORK_CANDIDATES[framework]["pr_url"]
    assert payload["framework_agent_candidate_id"] == _FRAMEWORK_CANDIDATES[framework]["pr_url"]
    # No progress rows yet (verdict/executor write those).
    assert stub.shared_state.framework_agent_phase_progress == []


def test_pump_integration_parametrised_over_all_three_frameworks():
    """G4 static guard: the parametrisation list covers exactly the three frameworks (and matches the fixture dict)."""
    assert set(_FRAMEWORK_PARAMETRISATION) == {"sglang", "vllm", "atom"}
    assert set(_FRAMEWORK_CANDIDATES.keys()) == set(_FRAMEWORK_PARAMETRISATION)


# Scenario 2 — submit → (async reject) → next tick submits the next candidate
def test_pump_reject_then_next_tick_submits_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Two-candidate batch: c1 submitted; an async reject records ``critic_denied``; next tick submits c2."""
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
    stub = _CoordinatorStub(tmp_path)

    # Tick 1 — discover + submit c1 proposal; no task.
    _pump(stub)
    assert discover_calls.n == 1
    assert stub.tasks.created == []
    pendings = _framework_agent_pendings(stub)
    assert len(pendings) == 1
    p1 = pendings[0]
    assert p1.payload["framework_agent_candidate_id"] == "https://example.com/pr/1"

    # Async reject arrives: critic_denied row + the proposal is decided.
    asyncio.run(
        Coordinator._record_framework_agent_critic_denied(stub, p1, "out of scope")  # type: ignore[arg-type]
    )
    p1.decided = True
    denied = [r for r in stub.shared_state.framework_agent_phase_progress if r.get("status") == "critic_denied"]
    assert len(denied) == 1
    assert denied[0]["candidate_id"] == "https://example.com/pr/1"

    # Tick 2 — selector skips the denied candidate and submits c2 (discover NOT re-called).
    _pump(stub)
    assert discover_calls.n == 1, "discover must not be called again — batch still has work"
    new_pendings = [p for p in _framework_agent_pendings(stub) if not getattr(p, "decided", False)]
    assert len(new_pendings) == 1
    assert new_pendings[0].payload["framework_agent_candidate_id"] == "https://example.com/pr/2"


# Scenario 3 — already-in-flight task → pump no-op
def test_pump_is_noop_when_framework_agent_task_already_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """An already-running ``framework_agent`` task makes the pump return early without discover or submit (idempotency guard)."""
    discover_called = SimpleNamespace(n=0)

    async def _discover(**_: Any) -> dict[str, Any]:
        discover_called.n += 1
        return {"batch_id": "unused", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _CoordinatorStub(tmp_path)
    stub.tasks._running.append(
        SimpleNamespace(kind="framework_agent", task_id="t-existing"),
    )

    _pump(stub)

    assert discover_called.n == 0
    assert stub.tasks.created == []
    assert _framework_agent_pendings(stub) == []
    assert stub.shared_state.framework_agent_phase_done is False


# Scenario 3b — a pending candidate proposal serializes the pump (no second submit)
def test_pump_is_noop_while_candidate_proposal_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    discover_called = SimpleNamespace(n=0)

    async def _discover(**_: Any) -> dict[str, Any]:
        discover_called.n += 1
        return {
            "batch_id": "b",
            "candidates": [{"pr_url": "https://example.com/pr/9", "repo": "a/b", "ref": "x"}],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _CoordinatorStub(tmp_path)

    _pump(stub)
    _pump(stub)  # pending proposal -> serialized, no second submit
    assert len(_framework_agent_pendings(stub)) == 1
    assert discover_called.n == 1


# Scenario 4 — discover empty payload → phase_history row + phase_done
def test_pump_retries_empty_discover_before_marking_phase_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A clean empty discover payload is tolerated for up to
    ``DISCOVER_FAILURE_RETRY_LIMIT`` consecutive ticks (transient upstream
    blip) before flipping ``framework_agent_phase_done`` and recording a
    phase_history row. This guards against one momentary empty result silently
    skipping the entire FRAMEWORK phase.
    """

    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _CoordinatorStub(tmp_path)

    # First (limit - 1) empty batches retry without ending the phase.
    for i in range(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT - 1):
        _pump(stub)
        assert stub.shared_state.framework_agent_phase_done is False
        assert stub.shared_state.framework_agent_empty_discoveries == i + 1
        assert _framework_agent_pendings(stub) == []
        assert [
            r for r in stub.shared_state.phase_history if r.get("event") == "framework_agent_phase_done"
        ] == []

    # The limit-th consecutive empty batch ends the phase with a summary row.
    _pump(stub)
    assert stub.shared_state.framework_agent_phase_done is True
    assert _framework_agent_pendings(stub) == []
    rows = [r for r in stub.shared_state.phase_history if r.get("event") == "framework_agent_phase_done"]
    assert len(rows) == 1
    assert rows[0]["reason"] == "discover_empty_payload"


def test_pump_empty_then_nonempty_discover_resets_streak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A non-empty batch after some empty ones clears the empty-discovery streak and submits a candidate."""
    calls = SimpleNamespace(n=0)

    async def _discover(**_: Any) -> dict[str, Any]:
        calls.n += 1
        if calls.n == 1:
            return {"batch_id": "", "candidates": []}
        return {
            "batch_id": "batch-recover",
            "candidates": [{"pr_url": "https://example.com/pr/1", "repo": "a/b", "ref": "x1"}],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _CoordinatorStub(tmp_path)

    _pump(stub)  # empty -> retry
    assert stub.shared_state.framework_agent_phase_done is False
    assert stub.shared_state.framework_agent_empty_discoveries == 1

    _pump(stub)  # non-empty -> resets streak, submits candidate
    assert stub.shared_state.framework_agent_empty_discoveries == 0
    assert stub.shared_state.framework_agent_phase_done is False
    assert len(_framework_agent_pendings(stub)) == 1
