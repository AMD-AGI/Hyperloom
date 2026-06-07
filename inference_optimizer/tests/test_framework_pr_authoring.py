"""Tests for the FRAMEWORK_PR authoring track.

The authoring track widens the FRAMEWORK_PR phase from "apply an existing
PR diff" to "read source + author a new patch". For each discovered +
Critic-approved candidate the pump dispatches a write-capable
``serving_specialist`` (in addition to the diff-only ``framework_pr``
task); the authored patch then flows through the existing
autosubmit → Critic → integrate_patch → bench → KEEP/REVERT chain.

Coverage:

1. **pump dispatches authoring specialist** — happy path enqueues BOTH a
   ``framework_pr`` task (diff track) and a ``specialist`` task
   (authoring track), and the specialist params carry the FRAMEWORK_PR
   provenance markers + PR seed notes.
2. **authoring disabled** — with the flag off, only the diff track runs
   (preserves the legacy behaviour the other pump tests assert).
3. **_framework_pr_authoring_inflight** — True while a specialist /
   integrate_patch task is in flight or an integrate_patch proposal is
   pending Critic review; False otherwise.
4. **_record_framework_pr_authored_outcome** — a KEEP integrate_patch
   result writes a progress row + rolls the batch max-gain stat the
   plateau judge reads; non-terminal statuses are ignored.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator import framework_agent_client as _fa_client
from inference_optimizer.orchestrator.coordinator import Coordinator


class _StateStub:
    def __init__(self, *, authoring: bool = True) -> None:
        self.phase = "FRAMEWORK_PR"
        self.framework_pr_phase_done = False
        self.framework_pr_discover_failures = 0
        self.framework_pr_batches: list[dict[str, Any]] = []
        self.framework_pr_phase_progress: list[dict[str, Any]] = []
        self.framework_pr_critic_decisions: list[dict[str, Any]] = []
        self.framework_pr_authoring_enabled = authoring
        self.phase_history: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self.model = "test-model"
        self.framework = "sglang"
        self.gpu_type = "MI300X"
        self.baseline_tput = 1000.0
        self._saves = 0

    def save(self, _session_dir: Path) -> None:
        self._saves += 1


class _TasksStub:
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
        t = SimpleNamespace(
            kind=kwargs.get("kind"),
            task_id=f"t-{len(self.created)}",
            params=kwargs.get("params") or {},
        )
        self._queued.append(t)
        return t


def _make_intent(prompt: str, verdict: str):
    import re

    from inference_optimizer.protocol.intent import Intent, IntentType

    m = re.search(r"msg_id=([a-f0-9]+)", prompt)
    msg_id = m.group(1) if m else "x"
    return Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": msg_id,
            "verdict": verdict,
            "reasoning": "ok",
        },
    )


class _ApproveCritic:
    def __init__(self) -> None:
        self.call_count = 0

    async def run(self, prompt: str, **_: Any) -> Any:
        self.call_count += 1
        return SimpleNamespace(
            intents=[_make_intent(prompt, "approve")], raw_text="(approve)",
        )


class _Stub:
    """Binds the Coordinator methods the pump + helpers touch."""

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
    _enqueue_framework_pr_authoring_specialist = (
        Coordinator._enqueue_framework_pr_authoring_specialist
    )
    _framework_pr_authoring_inflight = (
        Coordinator._framework_pr_authoring_inflight
    )
    _record_framework_pr_authored_outcome = (
        Coordinator._record_framework_pr_authored_outcome
    )
    _pump_framework_pr_phase = Coordinator._pump_framework_pr_phase

    def __init__(self, tmp_path: Path, *, authoring: bool = True) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub(authoring=authoring)
        self.tasks = _TasksStub()
        self.framework_pr_discover_timeout_sec = 0.0
        self.backends: dict[str, Any] = {"critic": _ApproveCritic()}
        self.state = SimpleNamespace(pending_proposals={})

    async def _warm_specialist_params(self, params: dict[str, Any]) -> None:
        # No-op: avoid pulling in KnowledgePlane in the unit test.
        return None

    def _framework_pr_discover_repo_urls(self, framework: str) -> list[str]:
        return [_fa_client.repo_url_for_framework(framework or "sglang")]

    def _framework_pr_known_candidate_ids(self) -> set[str]:
        return Coordinator._framework_pr_known_candidate_ids(self)  # type: ignore[arg-type]

    def _framework_pr_tried_refs(self) -> list[str]:
        return Coordinator._framework_pr_tried_refs(self)  # type: ignore[arg-type]


_CANDIDATE = {
    "pr_url": "https://github.com/sgl-project/sglang/pull/42",
    "diff_url": "https://github.com/sgl-project/sglang/pull/42.diff",
    "repo": "sgl-project/sglang",
    "ref": "perf/moe",
    "title": "perf: fused moe gemm fastpath",
    "framework": "sglang",
}


def _pump(stub: _Stub) -> None:
    asyncio.run(Coordinator._pump_framework_pr_phase(stub))  # type: ignore[arg-type]


def test_pump_dispatches_authoring_specialist_alongside_diff_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)

    _pump(stub)

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds.count("framework_pr") == 1
    assert kinds.count("specialist") == 1

    spec = next(c for c in stub.tasks.created if c["kind"] == "specialist")
    params = spec["params"]
    assert params["framework_pr_authoring"] is True
    assert params["domain"] == "serving_specialist"
    assert params["readonly"] is False
    assert params["framework_pr_candidate_id"] == _CANDIDATE["pr_url"]
    assert _CANDIDATE["pr_url"] in params["notes"]
    assert _CANDIDATE["diff_url"] in params["notes"]
    # Write tools are granted so the specialist can author a patch.
    assert "Write" in spec["allowed_tools"]
    assert "Edit" in spec["allowed_tools"]
    # Worktree authoring lane (not the serving lane — integrate_patch
    # grabs that later).
    assert spec["requires_lanes"] == ["research_lane"]


def test_pump_authoring_disabled_runs_diff_track_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=False)

    _pump(stub)

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds == ["framework_pr"]


def test_authoring_inflight_detects_specialist_and_proposals(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True)

    # Nothing in flight.
    assert asyncio.run(
        Coordinator._framework_pr_authoring_inflight(stub)  # type: ignore[arg-type]
    ) is False

    # A running specialist counts.
    stub.tasks._running.append(SimpleNamespace(kind="specialist", task_id="s1"))
    assert asyncio.run(
        Coordinator._framework_pr_authoring_inflight(stub)  # type: ignore[arg-type]
    ) is True
    stub.tasks._running.clear()

    # A queued integrate_patch counts.
    stub.tasks._queued.append(
        SimpleNamespace(kind="integrate_patch", task_id="i1"),
    )
    assert asyncio.run(
        Coordinator._framework_pr_authoring_inflight(stub)  # type: ignore[arg-type]
    ) is True
    stub.tasks._queued.clear()

    # A pending integrate_patch Critic proposal counts (the gap between
    # specialist completion and integrate_patch task creation).
    stub.state.pending_proposals = {
        "m1": SimpleNamespace(action_name="integrate_patch"),
    }
    assert asyncio.run(
        Coordinator._framework_pr_authoring_inflight(stub)  # type: ignore[arg-type]
    ) is True


def test_record_authored_outcome_writes_progress_and_rolls_max_gain(
    tmp_path: Path,
):
    stub = _Stub(tmp_path, authoring=True)
    stub.shared_state.framework_pr_batches = [
        {"batch_id": "b1", "max_gain_pct_observed_in_batch": 1.0},
    ]
    task = SimpleNamespace(
        task_id="i-1",
        params={
            "framework_pr_candidate_id": "pr-42",
            "framework_pr_batch_id": "b1",
            "specialist_task_id": "s-1",
        },
    )
    result = SimpleNamespace(
        state="succeeded",
        result={"status": "kept", "delta_pct": 6.5, "output_throughput": 1065.0},
    )

    Coordinator._record_framework_pr_authored_outcome(  # type: ignore[arg-type]
        stub, task=task, result=result,
    )

    progress = stub.shared_state.framework_pr_phase_progress
    assert len(progress) == 1
    row = progress[0]
    assert row["status"] == "kept"
    assert row["kept"] is True
    assert row["provenance"] == "authored"
    assert row["candidate_id"] == "pr-42"
    assert row["gain_pct"] == pytest.approx(6.5)
    # Batch max-gain rolled from 1.0 → 6.5.
    assert stub.shared_state.framework_pr_batches[0][
        "max_gain_pct_observed_in_batch"
    ] == pytest.approx(6.5)


def test_record_authored_outcome_ignores_non_terminal_status(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True)
    task = SimpleNamespace(task_id="i-2", params={"framework_pr_batch_id": "b1"})
    result = SimpleNamespace(
        state="succeeded", result={"status": "apply_failed"},
    )

    Coordinator._record_framework_pr_authored_outcome(  # type: ignore[arg-type]
        stub, task=task, result=result,
    )

    assert stub.shared_state.framework_pr_phase_progress == []
