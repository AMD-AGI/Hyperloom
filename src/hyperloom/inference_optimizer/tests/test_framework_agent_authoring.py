"""Tests for the FRAMEWORK authoring track."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.framework import client as _fa_client
from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult

from hyperloom.orchestrator.state.shared_state import SharedState

from ._optimize_fixtures import FakeCoordinator, optimize_state


def _state(*, authoring: bool = True) -> SharedState:
    """Real ``SharedState`` seeded for the authoring track.

    The local-exploration arm is off: this suite exercises the PR-authoring
    track, and the arm has dedicated coverage elsewhere.
    """
    return optimize_state(
        framework_agent_authoring_enabled=authoring,
        framework_local_explore_enabled=False,
        model="test-model",
        framework="sglang",
        gpu_type="MI300X",
        baseline_tput=1000.0,
    )


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
            state="queued",
        )
        self._queued.append(t)
        return t, False


def _make_intent(prompt: str, verdict: str):
    import re

    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

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
            intents=[_make_intent(prompt, "approve")],
            raw_text="(approve)",
        )


class _BusStub:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def append_and_seq(self, msg: Any) -> Any:
        self.messages.append(msg)
        return msg

    async def tail(self, n: int = 200, *, topic: str | None = None, **_: Any) -> list[Any]:
        messages = [msg for msg in self.messages if topic is None or getattr(msg, "topic", topic) == topic]
        return list(reversed(messages[-n:]))


class _Stub(FakeCoordinator):
    """The state a FRAMEWORK pump tick reads; the rest resolves for real.

    Only genuine boundaries are doubled here: the task store, the bus, the
    Critic backend and the discovery call. Every Coordinator method the pump
    reaches for is served by its real collaborator, so a helper added to the
    call chain needs no edit in this file.
    """

    def __init__(self, tmp_path: Path, *, authoring: bool = True) -> None:
        super().__init__(
            tmp_path,
            shared_state=_state(authoring=authoring),
            tasks=_TasksStub(),
            bus=_BusStub(),
            backends={"critic": _ApproveCritic()},
            state=SimpleNamespace(pending_proposals={}),
            framework_agent_discover_timeout_sec=0.0,
            # No GPU pool: authoring degrades to the research-lane-only path.
            framework_gpu_pool=None,
        )

    async def _record_observation(self, *_a: Any, **_k: Any) -> None:
        return None

    async def _warm_specialist_params(self, params: dict[str, Any]) -> None:
        return None

    def _framework_agent_discover_repo_urls(self, framework: str) -> list[str]:
        return [_fa_client.repo_url_for_framework(framework or "sglang")]


_CANDIDATE = {
    "pr_url": "https://github.com/sgl-project/sglang/pull/42",
    "diff_url": "https://github.com/sgl-project/sglang/pull/42.diff",
    "repo": "sgl-project/sglang",
    "ref": "perf/moe",
    "title": "perf: fused moe gemm fastpath",
    "framework": "sglang",
}


def _seed_batch(stub: _Stub, *candidates: dict[str, Any]) -> None:
    """Put a discovered batch in state.

    Discovery is the candidate-discovery specialist's deliverable, landing in
    ``framework_agent_batches``; the pump reads that. A test that patched
    ``fa phase-discover`` would be patching a call the pump no longer makes.
    """
    stub.shared_state.framework_agent_batches = [{"batch_id": "b1", "candidates": [dict(c) for c in candidates]}]


def _pump(stub: _Stub) -> None:
    asyncio.run(stub._pump_framework_agent_phase())


def _pump_then_materialize(stub: _Stub) -> None:
    """Run the pump (resolves audit route + submits a candidate proposal) then materialise it.

    The pump submits the candidate carrying its resolved ``audit_step``; an
    approve verdict materialises it via ``_materialize_framework_agent_candidate``,
    which performs the apply/author dispatch.
    """
    asyncio.run(stub._pump_framework_agent_phase())
    pendings = [
        p
        for p in stub.state.pending_proposals.values()
        if getattr(p, "action_name", "") == "integrate_patch" and not getattr(p, "decided", False)
    ]
    for p in pendings:
        asyncio.run(stub._materialize_framework_agent_candidate(p))
        p.decided = True


def _materialize(stub: _Stub, *, audit_step: str = "") -> None:
    from hyperloom.orchestrator.loop.coordinator import PendingProposal

    pending = PendingProposal(
        proposal_msg_id="m-fpr",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={
            "candidate": dict(_CANDIDATE),
            "framework_agent_candidate_id": _CANDIDATE["pr_url"],
            "batch_id": "b1",
            "audit": {},
            "audit_step": audit_step,
        },
    )
    asyncio.run(stub._materialize_framework_agent_candidate(pending))


def test_pump_submits_candidate_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The pump submits the candidate as a ``framework_agent`` proposal; no task is created inline."""

    stub = _Stub(tmp_path, authoring=True)
    _seed_batch(stub, _CANDIDATE)

    _pump(stub)

    assert stub.tasks.created == []
    pendings = [p for p in stub.state.pending_proposals.values() if p.action_name == "integrate_patch"]
    assert len(pendings) == 1
    assert pendings[0].payload["framework_agent_candidate_id"] == _CANDIDATE["pr_url"]


def test_materialize_unknown_route_dispatches_both_tracks(
    tmp_path: Path,
):
    """An approved candidate with an unknown audit route runs the raw-diff + authoring tracks."""
    stub = _Stub(tmp_path, authoring=True)

    _materialize(stub, audit_step="")

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds.count("integrate_patch") == 1
    assert kinds.count("specialist") == 1

    spec = next(c for c in stub.tasks.created if c["kind"] == "specialist")
    params = spec["params"]
    assert params["framework_agent_authoring"] is True
    assert params["domain"] == "serving_specialist"
    assert params["source_phase"] == "FRAMEWORK_AGENT"
    assert params["framework_agent_candidate_id"] == _CANDIDATE["pr_url"]
    assert params.get("task_kind") == "framework_authoring"
    pr_lead = params.get("pr_lead") or {}
    assert _CANDIDATE["pr_url"] == pr_lead.get("url") or _CANDIDATE["pr_url"] in params.get("notes", "")
    assert _CANDIDATE["diff_url"] == pr_lead.get("diff_url") or _CANDIDATE["diff_url"] in params.get("notes", "")
    assert spec["requires_lanes"] == ["research_lane"]


def test_materialize_authoring_disabled_runs_diff_track_only(
    tmp_path: Path,
):
    stub = _Stub(tmp_path, authoring=False)

    _materialize(stub, audit_step="")

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds == ["integrate_patch"]


def test_reauthor_attempt_propagates_into_specialist_and_integrate_params(tmp_path: Path):
    from hyperloom.orchestrator.phases.explore import _forward_integrate_source

    stub = _Stub(tmp_path, authoring=True)

    task_id = asyncio.run(
        stub._enqueue_framework_agent_authoring_specialist(
            dict(_CANDIDATE),
            {},
            reauthor_attempt=1,
        )
    )

    specialist_task = stub.tasks._queued[-1]
    assert task_id == specialist_task.task_id
    assert specialist_task.params["reauthor_attempt"] == 1
    round_entry = stub._build_specialist_round_entry(
        task=specialist_task,
        done_payload={"proposal_set": [], "empty": True},
        source=f"specialist:{task_id}",
    )
    assert round_entry["reauthor_attempt"] == 1
    integrate_params: dict[str, Any] = {}
    _forward_integrate_source(specialist_task.params, integrate_params)
    assert integrate_params["reauthor_attempt"] == 1


@pytest.mark.parametrize(
    ("route", "authoring", "kinds"),
    [
        ("direct_framework", True, ["integrate_patch"]),
        ("author_via_specialist", True, ["specialist"]),
        # Author route with the arm off must still land on the raw-diff track:
        # the alternative is a candidate that is approved and then stranded.
        ("author_via_specialist", False, ["integrate_patch"]),
        # An unlabelled candidate takes the author route: rewriting against
        # live source is the safe default for one nobody vetted.
        ("", True, ["specialist"]),
        # A route the pump does not recognise runs both tracks rather than
        # picking one on a guess.
        ("unrecognised", True, ["integrate_patch", "specialist"]),
    ],
)
def test_the_route_the_discovery_specialist_returns_picks_the_track(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    authoring: bool,
    kinds: list[str],
):
    """``candidate["route"]`` is the only input that selects apply-vs-author.

    The pump neither re-audits nor re-ranks -- the route travels from the
    discovery specialist's batch through the proposal payload to the dispatch.
    """
    stub = _Stub(tmp_path, authoring=authoring)
    _seed_batch(stub, dict(_CANDIDATE, route=route))

    _pump_then_materialize(stub)

    assert sorted(c["kind"] for c in stub.tasks.created) == sorted(kinds)


def test_authoring_inflight_detects_specialist_and_proposals(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True)
    # One unprocessed candidate so the signal has a valid target.
    _CAND_ID = "https://github.com/ROCm/vllm/pull/999"
    stub.shared_state.framework_agent_batches = [{"batch_id": "b1", "candidates": [{"candidate_id": _CAND_ID}]}]
    stub.shared_state.framework_agent_phase_progress = []

    assert asyncio.run(stub._framework_agent_authoring_inflight()) is False

    # A running framework-owned specialist for the unprocessed candidate counts.
    stub.tasks._running.append(
        SimpleNamespace(
            kind="specialist",
            task_id="s1",
            params={"framework_agent_authoring": True, "framework_agent_candidate_id": _CAND_ID},
        )
    )
    assert asyncio.run(stub._framework_agent_authoring_inflight()) is True
    stub.tasks._running.clear()

    # A kernel-phase specialist (no framework_agent_authoring) does NOT count.
    stub.tasks._running.append(SimpleNamespace(kind="specialist", task_id="k1", params={}))
    assert asyncio.run(stub._framework_agent_authoring_inflight()) is False
    stub.tasks._running.clear()

    # A queued framework-owned integrate_patch for the unprocessed candidate counts.
    stub.tasks._queued.append(
        SimpleNamespace(
            kind="integrate_patch",
            task_id="i1",
            params={"framework_agent_authoring": True, "framework_agent_candidate_id": _CAND_ID},
        )
    )
    assert asyncio.run(stub._framework_agent_authoring_inflight()) is True
    stub.tasks._queued.clear()

    # A bare kernel integrate_patch task (no framework_agent_authoring) does NOT count.
    stub.tasks._queued.append(SimpleNamespace(kind="integrate_patch", task_id="k2", params={}))
    assert asyncio.run(stub._framework_agent_authoring_inflight()) is False
    stub.tasks._queued.clear()

    # A pending framework_agent Critic proposal for the unprocessed candidate counts.
    stub.state.pending_proposals = {
        "m1": SimpleNamespace(
            action_name="integrate_patch",
            decided=False,
            payload={"framework_agent_candidate_id": _CAND_ID},
        ),
    }
    assert asyncio.run(stub._framework_agent_authoring_inflight()) is True

    # A pending framework-owned integrate_patch proposal for the unprocessed candidate counts.
    stub.state.pending_proposals = {
        "m2": SimpleNamespace(
            action_name="integrate_patch",
            decided=False,
            payload={"params": {"framework_agent_authoring": True, "framework_agent_candidate_id": _CAND_ID}},
        ),
    }
    assert asyncio.run(stub._framework_agent_authoring_inflight()) is True

    # A bare (kernel-style) integrate_patch proposal does NOT count.
    stub.state.pending_proposals = {
        "m3": SimpleNamespace(
            action_name="integrate_patch",
            decided=False,
            payload={"params": {}},
        ),
    }
    assert asyncio.run(stub._framework_agent_authoring_inflight()) is False

    # A decided proposal does NOT count even if framework-owned.
    stub.state.pending_proposals = {
        "m4": SimpleNamespace(
            action_name="integrate_patch",
            decided=True,
            payload={"framework_agent_candidate_id": _CAND_ID},
        ),
    }
    assert asyncio.run(stub._framework_agent_authoring_inflight()) is False


def test_record_authored_outcome_writes_progress_and_rolls_max_gain(
    tmp_path: Path,
):
    stub = _Stub(tmp_path, authoring=True)
    stub.shared_state.framework_agent_batches = [
        {"batch_id": "b1", "max_gain_pct_observed_in_batch": 1.0},
    ]
    task = SimpleNamespace(
        task_id="i-1",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "pr-42",
            "framework_batch_id": "b1",
            "specialist_task_id": "s-1",
            "reauthor_attempt": 1,
        },
    )
    result = SimpleNamespace(
        state="succeeded",
        result={"status": "kept", "delta_pct": 6.5, "output_throughput": 1065.0},
    )

    stub._record_framework_agent_authored_outcome(
        task=task,
        result=result,
    )

    progress = stub.shared_state.framework_agent_phase_progress
    assert len(progress) == 1
    row = progress[0]
    assert row["status"] == "kept"
    assert row["kept"] is True
    assert row["provenance"] == "authored"
    assert row["candidate_id"] == "pr-42"
    assert row["gain_pct"] == pytest.approx(6.5)
    assert row["reauthor_attempt"] == 1
    assert stub.shared_state.framework_agent_batches[0]["max_gain_pct_observed_in_batch"] == pytest.approx(6.5)


def test_record_authored_outcome_records_apply_failed_terminal(tmp_path: Path):
    """A non-keep terminal status (apply_failed) MUST still be recorded.

    Without a terminal row the FRAMEWORK pump re-selects the same candidate
    every tick. Only empty/in-progress is skipped.
    """
    stub = _Stub(tmp_path, authoring=True)
    task = SimpleNamespace(
        task_id="i-2",
        params={"framework_agent_authoring": True, "framework_batch_id": "b1"},
    )
    result = SimpleNamespace(
        state="succeeded",
        result={"status": "apply_failed"},
    )

    stub._record_framework_agent_authored_outcome(
        task=task,
        result=result,
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["status"] == "apply_failed"
    assert rows[0]["kept"] is False


def test_record_authored_outcome_requires_task_provenance(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True)
    task = SimpleNamespace(task_id="unrelated", params={"specialist_task_id": "s-other"})

    stub._record_framework_agent_authored_outcome(
        task=task,
        result={"status": "kept", "delta_pct": 1.0},
    )

    assert stub.shared_state.framework_agent_phase_progress == []


def test_record_authored_outcome_resolves_candidate_via_specialist_map(tmp_path: Path):
    """integrate_patch carries only specialist_task_id; the bridge must map it
    back to the originating PR-URL candidate so the row matches the select key.
    """
    stub = _Stub(tmp_path, authoring=True)
    stub.shared_state.framework_agent_specialist_candidate_map = {
        "spec-7": "https://github.com/ROCm/aiter/pull/3888",
    }
    task = SimpleNamespace(
        task_id="i-9",
        params={
            "framework_agent_authoring": True,
            "framework_batch_id": "b1",
            "specialist_task_id": "spec-7",
        },
    )
    result = SimpleNamespace(
        state="succeeded",
        result={"status": "reverted", "delta_pct": -0.3},
    )

    stub._record_framework_agent_authored_outcome(
        task=task,
        result=result,
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "https://github.com/ROCm/aiter/pull/3888"
    assert rows[0]["status"] == "reverted"


def test_record_authored_outcome_replaces_stale_empty_row(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True)
    stub.shared_state.phase = "FRAMEWORK_AGENT"
    stub.shared_state.framework_agent_phase_progress = [
        {
            "candidate_id": "local_explore:2",
            "status": "author_empty",
            "provenance": "authored_empty",
        }
    ]
    task = SimpleNamespace(
        task_id="integrate-local-2",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "local_explore:2",
            "specialist_task_id": "specialist-local-2",
        },
    )

    stub._record_framework_agent_authored_outcome(
        task=task,
        result={"status": "reverted", "delta_pct": -0.2},
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["status"] == "reverted"
    assert rows[0]["provenance"] == "authored"
    assert rows[0]["integrate_task_id"] == "integrate-local-2"


@pytest.mark.asyncio
async def test_enqueue_authoring_stamps_recovery_failed_when_unrecoverable(tmp_path: Path):
    """A terminal specialist with no recoverable outcome stamps a terminal row so the pump cannot re-select it forever."""
    stub = _Stub(tmp_path, authoring=True)
    candidate = {
        "candidate_id": "cand-x",
        "pr_url": "https://github.com/o/r/pull/1",
        "batch_id": "b0",
    }

    async def _create_or_return_existing(**kwargs: Any) -> Any:
        terminal = SimpleNamespace(
            kind=kwargs.get("kind"),
            task_id="specialist-terminal",
            params=kwargs.get("params") or {},
            state="succeeded",
        )
        return terminal, True

    stub.tasks.create_or_return_existing = _create_or_return_existing
    # Empty bus -> recovery finds no delegated_result and returns False.
    stub.bus.messages = []

    tid = await stub._enqueue_framework_agent_authoring_specialist(
        candidate,
    )

    assert tid == ""
    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "cand-x"
    assert rows[0]["status"] == "recovery_failed"
    # Candidate is now marked processed so the pump will not re-select it.
    assert "cand-x" in stub._framework_processed_candidate_keys()


@pytest.mark.asyncio
async def test_recover_authored_outcome_uses_persisted_integrate_result(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True)
    stub.shared_state.phase = "FRAMEWORK_AGENT"
    specialist_task = SimpleNamespace(
        task_id="specialist-local-2",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "local_explore:2",
        },
    )
    integrate_task = SimpleNamespace(
        task_id="integrate-local-2",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "local_explore:2",
            "specialist_task_id": specialist_task.task_id,
        },
    )

    async def _get(task_id: str) -> Any:
        assert task_id == integrate_task.task_id
        return integrate_task

    stub.tasks.get = _get
    stub.bus.messages = [
        SimpleNamespace(
            topic="delegated_result",
            payload={
                "task_id": specialist_task.task_id,
                "kind": "specialist",
                "result": {
                    "specialist_done": {
                        "patches_written": ["patches/local.patch"],
                        "proposal_set": [{"name": "local-source-change"}],
                    }
                },
            },
        ),
        SimpleNamespace(
            topic="delegated_result",
            payload={
                "task_id": integrate_task.task_id,
                "kind": "integrate_patch",
                "result": {"status": "reverted", "delta_pct": -0.2},
            },
        ),
    ]

    recovered = await stub._recover_framework_agent_authoring_outcome(
        specialist_task=specialist_task,
    )

    assert recovered is True
    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["status"] == "reverted"
    assert rows[0]["provenance"] == "authored"


@pytest.mark.asyncio
async def test_dispatcher_records_authored_outcome_after_phase_transition(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=False)
    stub.shared_state.phase = "FRAMEWORK_AGENT"
    recorded: list[str] = []

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    stub._record_intervention_for_task = lambda *_args, **_kwargs: None
    stub._record_framework_agent_authored_outcome = lambda *, task, result: recorded.append(
        str(result.result.get("status") or "")
    )
    stub._maybe_rearm_authored_lane = lambda *_args, **_kwargs: None
    stub._drain_apply_fail_retry_pending = _noop_async
    stub._is_promotable_result = lambda *_args, **_kwargs: False
    stub._handle_unpromotable_result = _noop_async
    stub._fact_write_hook = _noop_async
    stub._record_coordinator_exception = lambda **_kwargs: None
    task = SimpleNamespace(
        task_id="integrate-cross-phase",
        kind="integrate_patch",
        params={"framework_agent_authoring": True, "reauthor_attempt": 1},
    )
    result = SubAgentResult(
        task_id=task.task_id,
        state="succeeded",
        result={"status": "reverted"},
    )

    await DispatcherCollaborator(stub)._reap_dispatched_task(task, result, None)

    assert recorded == ["reverted"]
    assert result.result["reauthor_attempt"] == 1


def test_empty_outcome_fires_when_patch_dropped_by_vetting(tmp_path: Path):
    """A patch dropped by safety-vetting (empty patches_written) must still stamp
    a terminal row (gate on patches_written, NOT proposal_set), else the FRAMEWORK
    pump re-dispatches the candidate forever (livelock).
    """
    stub = _Stub(tmp_path, authoring=True)
    cand = "https://github.com/sgl-project/sglang/pull/28067"
    task = SimpleNamespace(
        task_id="spec-28067",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand,
            "framework_batch_id": "b1",
            "framework_audit": {"semantic_status": "not_present"},
        },
    )
    done_payload = {
        "empty": False,
        "patches_written": [],
        "proposal_set": [{"name": "serving-gc-off-critical-path"}],
        "summary": "patch target file absent from framework tree",
    }

    stub._record_framework_agent_authoring_empty_outcome(
        task=task,
        done_payload=done_payload,
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == cand
    assert rows[0]["status"] == "not_applicable"
    assert rows[0]["kept"] is False


def test_empty_outcome_records_after_phase_transition(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True)
    stub.shared_state.phase = "FRAMEWORK_AGENT"
    task = SimpleNamespace(
        task_id="spec-empty-cross-phase",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "local_explore:3",
            "framework_batch_id": "",
        },
    )

    stub._record_framework_agent_authoring_empty_outcome(
        task=task,
        done_payload={"patches_written": [], "proposal_set": [], "summary": "No safe change"},
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "local_explore:3"
    assert rows[0]["status"] == "author_empty"


def test_empty_outcome_skips_when_patches_written_present(tmp_path: Path):
    """Non-empty patches_written means autosubmit will create an integrate_patch
    that owns the terminal row; the empty-outcome bridge must NOT also stamp one.
    """
    stub = _Stub(tmp_path, authoring=True)
    task = SimpleNamespace(
        task_id="spec-x",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "pr-x",
            "framework_batch_id": "b1",
        },
    )
    done_payload = {
        "patches_written": ["patches/001.patch"],
        "proposal_set": [{"name": "v1"}],
    }

    stub._record_framework_agent_authoring_empty_outcome(
        task=task,
        done_payload=done_payload,
    )

    assert stub.shared_state.framework_agent_phase_progress == []


def test_config_levers_helper_extracts_from_proposal_set():
    """Proposal args and envs retain separate channels; patches take precedence."""
    from hyperloom.orchestrator.loop.coordinator import (
        _framework_config_levers_from_done,
    )

    done = {
        "patches_written": [],
        "proposal_set": [
            {
                "name": "mtp-spec-decode",
                "extra_args": "--speculative-num-steps 3 --enable-mtp",
                "extra_envs": {"VLLM_USE_MTP": "1"},
            }
        ],
    }
    levers = _framework_config_levers_from_done(done)
    assert levers == {
        "extra_server_args": "--speculative-num-steps 3 --enable-mtp",
        "extra_envs": {"VLLM_USE_MTP": "1"},
    }

    # A patch deliverable is NOT a config-only outcome.
    assert (
        _framework_config_levers_from_done({"patches_written": ["p.patch"], "proposal_set": done["proposal_set"]}) == {}
    )
    # No levers → empty.
    assert (
        _framework_config_levers_from_done({"patches_written": [], "proposal_set": [{"name": "research-only"}]}) == {}
    )


def test_empty_outcome_skips_when_config_levers_present(tmp_path: Path):
    """A config-lever deliverable (proposal_set with extra_args/extra_envs and no
    patch) is routed to integrate_patch, so the
    empty-outcome bridge must NOT stamp an authored_empty row for it."""
    stub = _Stub(tmp_path, authoring=True)
    task = SimpleNamespace(
        task_id="spec-cfg",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "https://github.com/ROCm/vllm/pull/1014",
            "framework_batch_id": "b1",
            "framework_audit": {"semantic_status": "not_present"},
        },
    )
    done_payload = {
        "patches_written": [],
        "proposal_set": [{"name": "shared-expert-fusion", "extra_envs": {"VLLM_FUSE_SHARED_EXPERTS": "1"}}],
        "summary": "PR maps to a config lever on this build",
    }

    stub._record_framework_agent_authoring_empty_outcome(
        task=task,
        done_payload=done_payload,
    )

    assert stub.shared_state.framework_agent_phase_progress == []


def test_authoring_specialist_same_framework_no_cross(tmp_path: Path):
    """A non-cross audit must NOT stamp cross-framework params."""
    stub = _Stub(tmp_path)
    audit = {"recommended_next_step": "author_via_specialist"}  # no cross_framework layer
    tid = asyncio.run(stub._enqueue_framework_agent_authoring_specialist(dict(_CANDIDATE), audit))
    assert tid
    params = stub.tasks.created[-1]["params"]
    assert "cross_framework" not in params
    assert "CROSS-FRAMEWORK PORT" not in (params.get("notes") or "")


def test_empty_outcome_skips_when_artifacts_written_routable(tmp_path: Path):
    """A non-diff tuned artifact (``artifacts_written`` with a real source file)
    is a FULL result: autosubmit routes it to ``integrate_patch``, so the
    empty-outcome bridge must NOT stamp an authored_empty row for it."""
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    stub = _Stub(tmp_path, authoring=True)
    sid = "spec-art"
    spec_root = runs_dir(tmp_path, "specialist", sid)
    art_dir = spec_root / "worktree" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "tuned_fmoe.csv").write_text("cu_num,token\n304,16\n", encoding="utf-8")

    task = SimpleNamespace(
        task_id=sid,
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "https://github.com/ROCm/aiter/pull/4130",
            "framework_batch_id": "b1",
            "framework_audit": {"semantic_status": "not_present"},
        },
    )
    done_payload = {
        "empty": True,
        "patches_written": [],
        "proposal_set": [],
        "artifacts_written": [
            {
                "source": "artifacts/tuned_fmoe.csv",
                "target": "configs/model_configs/qwen3_tuned_fmoe.csv",
                "kind": "aiter_tuned_fmoe_csv",
            }
        ],
        "summary": "autotuned cu_num=304 fp8 fmoe rows",
    }

    stub._record_framework_agent_authoring_empty_outcome(
        task=task,
        done_payload=done_payload,
    )
    assert stub.shared_state.framework_agent_phase_progress == []


def test_empty_outcome_stamps_when_artifacts_source_missing(tmp_path: Path):
    """``artifacts_written`` present but the source file does NOT exist:
    autosubmit cannot route it to ``integrate_patch``, so the empty-outcome
    bridge MUST still stamp a terminal row (else the FRAMEWORK pump livelocks)."""
    stub = _Stub(tmp_path, authoring=True)
    task = SimpleNamespace(
        task_id="spec-art-missing",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "https://github.com/ROCm/aiter/pull/9999",
            "framework_batch_id": "b1",
            "framework_audit": {"semantic_status": "not_present"},
        },
    )
    done_payload = {
        "empty": True,
        "patches_written": [],
        "proposal_set": [],
        "artifacts_written": [
            {
                "source": "artifacts/missing.csv",
                "target": "configs/model_configs/x.csv",
                "kind": "aiter_tuned_fmoe_csv",
            }
        ],
        "summary": "claimed artifact but no file on disk",
    }

    stub._record_framework_agent_authoring_empty_outcome(
        task=task,
        done_payload=done_payload,
    )
    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["status"] == "not_applicable"
    assert rows[0]["kept"] is False


def test_empty_outcome_stamps_when_artifacts_source_outside_sandbox(tmp_path: Path):
    """A RELATIVE artifact ``source`` that resolves (via ``..``) to a real file
    OUTSIDE the specialist sandbox is NOT routable, so the empty-outcome bridge
    MUST stamp a terminal row."""
    import os

    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    stub = _Stub(tmp_path, authoring=True)
    sid = "spec-art-fw-escape"
    worktree = runs_dir(tmp_path, "specialist", sid) / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "escape_fw.csv"
    outside.write_text("x", encoding="utf-8")
    rel_escape = os.path.relpath(outside, worktree)
    task = SimpleNamespace(
        task_id=sid,
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "https://github.com/ROCm/aiter/pull/8888",
            "framework_batch_id": "b1",
            "framework_audit": {"semantic_status": "not_present"},
        },
    )
    done_payload = {
        "empty": True,
        "patches_written": [],
        "proposal_set": [],
        "artifacts_written": [
            {
                "source": rel_escape,
                "target": "configs/model_configs/x.csv",
                "kind": "aiter_tuned_fmoe_csv",
            }
        ],
        "summary": "artifact exists but escapes sandbox via ..",
    }
    stub._record_framework_agent_authoring_empty_outcome(
        task=task,
        done_payload=done_payload,
    )
    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["status"] == "not_applicable"
    assert rows[0]["kept"] is False


@pytest.mark.asyncio
async def test_perf_explore_retry_stamps_immutable_explore_owner(
    tmp_path: Path,
) -> None:
    stub = _Stub(tmp_path, authoring=True)

    task_id = await stub._enqueue_author_specialist(
        lane="perf_explore",
        attempt=1,
    )

    assert task_id
    params = stub.tasks.created[-1]["params"]
    assert params["source_phase"] == "FRAMEWORK_AGENT"
    assert params["gap_layer"] == "perf_explore"
