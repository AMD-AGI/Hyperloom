"""Tests for the FRAMEWORK authoring track."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.framework import client as _fa_client
from hyperloom.orchestrator.framework import paths as _framework_paths
from hyperloom.orchestrator.loop.coordinator import Coordinator


class _StateStub:
    def __init__(self, *, authoring: bool = True) -> None:
        self.phase = "FRAMEWORK_AGENT"
        self.framework_agent_phase_done = False
        self.framework_agent_discover_failures = 0
        self.framework_agent_batches: list[dict[str, Any]] = []
        self.framework_agent_phase_progress: list[dict[str, Any]] = []
        self.framework_agent_critic_decisions: list[dict[str, Any]] = []
        self.framework_agent_authoring_enabled = authoring
        self.framework_agent_specialist_candidate_map: dict[str, str] = {}
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


class _Stub:
    """Binds the Coordinator methods the pump + helpers touch."""

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
    _materialize_framework_agent_candidate = Coordinator._materialize_framework_agent_candidate
    _record_framework_agent_critic_denied = Coordinator._record_framework_agent_critic_denied
    _discover_next_framework_batch = Coordinator._discover_next_framework_batch
    _framework_agent_repo_url_origin_framework = staticmethod(
        Coordinator._framework_agent_repo_url_origin_framework
    )
    _enqueue_framework_agent_task = Coordinator._enqueue_framework_agent_task
    _enqueue_framework_agent_authoring_specialist = Coordinator._enqueue_framework_agent_authoring_specialist
    _framework_agent_authoring_inflight = Coordinator._framework_agent_authoring_inflight
    _record_framework_agent_authored_outcome = Coordinator._record_framework_agent_authored_outcome
    _record_framework_agent_audit_skip = Coordinator._record_framework_agent_audit_skip
    _framework_agent_audit_seed_lines = staticmethod(Coordinator._framework_agent_audit_seed_lines)
    _framework_agent_audit_skip_confident = staticmethod(Coordinator._framework_agent_audit_skip_confident)
    _framework_agent_roots_have_git = staticmethod(Coordinator._framework_agent_roots_have_git)
    _pump_framework_agent_phase = Coordinator._pump_framework_agent_phase
    # The stub has no GPU pool, so ``_framework_gpu_params`` degrades to ``{}``
    # and the authoring task stays on research_lane only (asserted below).
    _coerce_needs_gpu = staticmethod(Coordinator._coerce_needs_gpu)
    _framework_authoring_lanes_ttl = Coordinator._framework_authoring_lanes_ttl

    def _framework_gpu_params(self) -> dict[str, Any]:
        return {}

    def __init__(self, tmp_path: Path, *, authoring: bool = True) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub(authoring=authoring)
        self.tasks = _TasksStub()
        self.framework_agent_discover_timeout_sec = 0.0
        self.backends: dict[str, Any] = {"critic": _ApproveCritic()}
        self.state = SimpleNamespace(pending_proposals={})
        self.bus = _BusStub()
        # Audit verdict the pump's _audit_framework_agent_candidate returns; default
        # unknown (empty recommended_next_step) preserves legacy both-tracks.
        self._audit_verdict: dict[str, Any] = {"recommended_next_step": ""}

    async def _record_observation(self, *_a: Any, **_k: Any) -> None:
        return None

    async def _rank_framework_agent_candidates_llm(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        # Hermetic: force deterministic discovery-order fallback (no LLM call).
        return None

    async def _audit_framework_agent_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        v = self._audit_verdict
        try:
            candidate["_audit"] = v
        except Exception:
            pass
        return v

    async def _warm_specialist_params(self, params: dict[str, Any]) -> None:
        # No-op: avoid pulling in KnowledgePlane in the unit test.
        return None

    def _framework_agent_discover_repo_urls(self, framework: str) -> list[str]:
        return [_fa_client.repo_url_for_framework(framework or "sglang")]

    def _framework_known_candidate_ids(self) -> set[str]:
        return Coordinator._framework_known_candidate_ids(self)  # type: ignore[arg-type]

    def _framework_tried_refs(self) -> list[str]:
        return Coordinator._framework_tried_refs(self)  # type: ignore[arg-type]


_CANDIDATE = {
    "pr_url": "https://github.com/sgl-project/sglang/pull/42",
    "diff_url": "https://github.com/sgl-project/sglang/pull/42.diff",
    "repo": "sgl-project/sglang",
    "ref": "perf/moe",
    "title": "perf: fused moe gemm fastpath",
    "framework": "sglang",
}


def _pump(stub: _Stub) -> None:
    asyncio.run(Coordinator._pump_framework_agent_phase(stub))  # type: ignore[arg-type]


def _pump_then_materialize(stub: _Stub) -> None:
    """Run the pump (resolves audit route + submits a candidate proposal) then materialise it.

    Mirrors the live async gate: the pump submits the candidate carrying its
    resolved ``audit_step``; an approve verdict materialises it via
    ``_materialize_framework_agent_candidate``, which performs the apply/author
    dispatch (the routing that used to live inline in the pump).
    """
    asyncio.run(Coordinator._pump_framework_agent_phase(stub))  # type: ignore[arg-type]
    pendings = [
        p
        for p in stub.state.pending_proposals.values()
        if getattr(p, "action_name", "") == "framework_agent" and not getattr(p, "decided", False)
    ]
    for p in pendings:
        asyncio.run(Coordinator._materialize_framework_agent_candidate(stub, p))  # type: ignore[arg-type]
        p.decided = True


def _materialize(stub: _Stub, *, audit_step: str = "") -> None:
    from hyperloom.orchestrator.loop.coordinator import PendingProposal

    pending = PendingProposal(
        proposal_msg_id="m-fpr",
        from_agent="coordinator",
        action_name="framework_agent",
        predicted_gain_pct=0.0,
        payload={
            "candidate": dict(_CANDIDATE),
            "framework_agent_candidate_id": _CANDIDATE["pr_url"],
            "batch_id": "b1",
            "audit": {},
            "audit_step": audit_step,
        },
    )
    asyncio.run(Coordinator._materialize_framework_agent_candidate(stub, pending))  # type: ignore[arg-type]


def test_pump_submits_candidate_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The pump submits the candidate as a ``framework_agent`` proposal; no task is created inline."""
    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)

    _pump(stub)

    assert stub.tasks.created == []
    pendings = [p for p in stub.state.pending_proposals.values() if p.action_name == "framework_agent"]
    assert len(pendings) == 1
    assert pendings[0].payload["framework_agent_candidate_id"] == _CANDIDATE["pr_url"]


def test_materialize_unknown_route_dispatches_both_tracks(
    tmp_path: Path,
):
    """An approved candidate with an unknown audit route runs the raw-diff + authoring tracks."""
    stub = _Stub(tmp_path, authoring=True)

    _materialize(stub, audit_step="")

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds.count("framework_agent") == 1
    assert kinds.count("specialist") == 1

    spec = next(c for c in stub.tasks.created if c["kind"] == "specialist")
    params = spec["params"]
    assert params["framework_agent_authoring"] is True
    assert params["domain"] == "serving_specialist"
    assert params["readonly"] is False
    assert params["framework_agent_candidate_id"] == _CANDIDATE["pr_url"]
    assert _CANDIDATE["pr_url"] in params["notes"]
    assert _CANDIDATE["diff_url"] in params["notes"]
    assert "Write" in spec["allowed_tools"]
    assert "Edit" in spec["allowed_tools"]
    # Worktree authoring lane, not the serving lane.
    assert spec["requires_lanes"] == ["research_lane"]


def test_materialize_authoring_disabled_runs_diff_track_only(
    tmp_path: Path,
):
    stub = _Stub(tmp_path, authoring=False)

    _materialize(stub, audit_step="")

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds == ["framework_agent"]


def test_authoring_inflight_detects_specialist_and_proposals(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True)
    # Set up a batch with one unprocessed candidate so the signal has a valid target.
    _CAND_ID = "https://github.com/ROCm/vllm/pull/999"
    stub.shared_state.framework_agent_batches = [
        {"batch_id": "b1", "candidates": [{"candidate_id": _CAND_ID}]}
    ]
    stub.shared_state.framework_agent_phase_progress = []

    # Nothing in flight.
    assert (
        asyncio.run(
            Coordinator._framework_agent_authoring_inflight(stub)  # type: ignore[arg-type]
        )
        is False
    )

    # A running framework-owned specialist for the unprocessed candidate counts.
    stub.tasks._running.append(
        SimpleNamespace(
            kind="specialist",
            task_id="s1",
            params={"framework_agent_authoring": True, "framework_agent_candidate_id": _CAND_ID},
        )
    )
    assert (
        asyncio.run(
            Coordinator._framework_agent_authoring_inflight(stub)  # type: ignore[arg-type]
        )
        is True
    )
    stub.tasks._running.clear()

    # A kernel-phase specialist (no framework_agent_authoring) does NOT count.
    stub.tasks._running.append(SimpleNamespace(kind="specialist", task_id="k1", params={}))
    assert (
        asyncio.run(
            Coordinator._framework_agent_authoring_inflight(stub)  # type: ignore[arg-type]
        )
        is False
    )
    stub.tasks._running.clear()

    # A queued framework-owned integrate_patch for the unprocessed candidate counts.
    stub.tasks._queued.append(
        SimpleNamespace(
            kind="integrate_patch",
            task_id="i1",
            params={"framework_agent_authoring": True, "framework_agent_candidate_id": _CAND_ID},
        )
    )
    assert (
        asyncio.run(
            Coordinator._framework_agent_authoring_inflight(stub)  # type: ignore[arg-type]
        )
        is True
    )
    stub.tasks._queued.clear()

    # A bare kernel integrate_patch task (no framework_agent_authoring) does NOT count.
    stub.tasks._queued.append(
        SimpleNamespace(kind="integrate_patch", task_id="k2", params={})
    )
    assert (
        asyncio.run(
            Coordinator._framework_agent_authoring_inflight(stub)  # type: ignore[arg-type]
        )
        is False
    )
    stub.tasks._queued.clear()

    # A pending framework_agent Critic proposal for the unprocessed candidate counts.
    stub.state.pending_proposals = {
        "m1": SimpleNamespace(
            action_name="framework_agent",
            decided=False,
            payload={"framework_agent_candidate_id": _CAND_ID},
        ),
    }
    assert (
        asyncio.run(
            Coordinator._framework_agent_authoring_inflight(stub)  # type: ignore[arg-type]
        )
        is True
    )

    # A pending framework-owned integrate_patch proposal for the unprocessed candidate counts.
    stub.state.pending_proposals = {
        "m2": SimpleNamespace(
            action_name="integrate_patch",
            decided=False,
            payload={"params": {"framework_agent_authoring": True, "framework_agent_candidate_id": _CAND_ID}},
        ),
    }
    assert (
        asyncio.run(
            Coordinator._framework_agent_authoring_inflight(stub)  # type: ignore[arg-type]
        )
        is True
    )

    # A bare (kernel-style) integrate_patch proposal does NOT count.
    stub.state.pending_proposals = {
        "m3": SimpleNamespace(
            action_name="integrate_patch",
            decided=False,
            payload={"params": {}},
        ),
    }
    assert (
        asyncio.run(
            Coordinator._framework_agent_authoring_inflight(stub)  # type: ignore[arg-type]
        )
        is False
    )

    # A decided proposal does NOT count even if framework-owned.
    stub.state.pending_proposals = {
        "m4": SimpleNamespace(
            action_name="framework_agent",
            decided=True,
            payload={"framework_agent_candidate_id": _CAND_ID},
        ),
    }
    assert (
        asyncio.run(
            Coordinator._framework_agent_authoring_inflight(stub)  # type: ignore[arg-type]
        )
        is False
    )


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
            "framework_agent_candidate_id": "pr-42",
            "framework_batch_id": "b1",
            "specialist_task_id": "s-1",
        },
    )
    result = SimpleNamespace(
        state="succeeded",
        result={"status": "kept", "delta_pct": 6.5, "output_throughput": 1065.0},
    )

    Coordinator._record_framework_agent_authored_outcome(  # type: ignore[arg-type]
        stub,
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
    assert stub.shared_state.framework_agent_batches[0]["max_gain_pct_observed_in_batch"] == pytest.approx(6.5)


def test_record_authored_outcome_records_apply_failed_terminal(tmp_path: Path):
    """A non-keep terminal status (apply_failed) MUST still be recorded.

    Without a terminal row the FRAMEWORK pump re-selects the same candidate
    every tick (the authoring specialist's ``patches_written`` is non-empty so
    the empty-outcome bridge does not fire). Only empty/in-progress is skipped.
    """
    stub = _Stub(tmp_path, authoring=True)
    task = SimpleNamespace(task_id="i-2", params={"framework_batch_id": "b1"})
    result = SimpleNamespace(
        state="succeeded",
        result={"status": "apply_failed"},
    )

    Coordinator._record_framework_agent_authored_outcome(  # type: ignore[arg-type]
        stub,
        task=task,
        result=result,
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["status"] == "apply_failed"
    assert rows[0]["kept"] is False


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
        params={"framework_batch_id": "b1", "specialist_task_id": "spec-7"},
    )
    result = SimpleNamespace(
        state="succeeded",
        result={"status": "reverted", "delta_pct": -0.3},
    )

    Coordinator._record_framework_agent_authored_outcome(  # type: ignore[arg-type]
        stub,
        task=task,
        result=result,
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "https://github.com/ROCm/aiter/pull/3888"
    assert rows[0]["status"] == "reverted"


def test_empty_outcome_fires_when_patch_dropped_by_vetting(tmp_path: Path):
    """A specialist authors a patch (proposal_set non-empty) that safety-vetting
    then DROPS as unusable (missing_target), emptying patches_written. Autosubmit
    keys off patches_written so it creates NO integrate_patch — the authored
    bridge never fires. The empty-outcome bridge MUST stamp a terminal row
    (gate on patches_written, NOT proposal_set), else the FRAMEWORK pump
    re-dispatches the candidate forever (livelock).
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
        "patches_written": [],  # dropped by safety vetting
        "proposal_set": [{"name": "serving-gc-off-critical-path"}],
        "summary": "patch target file absent from framework tree",
    }

    Coordinator._record_framework_agent_authoring_empty_outcome(  # type: ignore[arg-type]
        stub,
        task=task,
        done_payload=done_payload,
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == cand
    assert rows[0]["status"] == "not_applicable"
    assert rows[0]["kept"] is False


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

    Coordinator._record_framework_agent_authoring_empty_outcome(  # type: ignore[arg-type]
        stub,
        task=task,
        done_payload=done_payload,
    )

    assert stub.shared_state.framework_agent_phase_progress == []


# Relaxed rule — config-lever deliverable is first-class -------------------
def test_config_levers_helper_extracts_from_proposal_set():
    """A proposal_set entry carrying extra_args / extra_envs is flattened into
    a config_changes dict; patches take precedence (returns {})."""
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
    assert levers["VLLM_USE_MTP"] == "1"
    assert levers["--speculative-num-steps"] == "3"
    assert levers["--enable-mtp"] == ""

    # A patch deliverable is NOT a config-only outcome.
    assert _framework_config_levers_from_done(
        {"patches_written": ["p.patch"], "proposal_set": done["proposal_set"]}
    ) == {}
    # No levers → empty.
    assert _framework_config_levers_from_done(
        {"patches_written": [], "proposal_set": [{"name": "research-only"}]}
    ) == {}


def test_empty_outcome_skips_when_config_levers_present(tmp_path: Path):
    """A config-lever deliverable (proposal_set with extra_args/extra_envs and no
    patch) is routed to integrate_patch's config_changes channel, so the
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
        "proposal_set": [
            {"name": "shared-expert-fusion", "extra_envs": {"VLLM_FUSE_SHARED_EXPERTS": "1"}}
        ],
        "summary": "PR maps to a config lever on this build",
    }

    Coordinator._record_framework_agent_authoring_empty_outcome(  # type: ignore[arg-type]
        stub,
        task=task,
        done_payload=done_payload,
    )

    assert stub.shared_state.framework_agent_phase_progress == []


# Audit-routed dispatch ----------------------------------------------------
def test_pump_audit_skip_records_terminal_row_no_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """already_equivalent audit -> skip: no Critic, no tasks, terminal row + KB."""
    import hyperloom.orchestrator.knowledge.kb_writeback as kb_writeback

    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path / "kb" / "framework_optimization")

    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)
    stub._audit_verdict = {
        "semantic_status": "already_equivalent",
        "applicability": "not_applicable",
        "recommended_next_step": "skip",
        "confidence": 0.95,
        "evidence": [{"local_file": "vllm/x.py", "symbol": "f", "reason": "present"}],
        "risks": [],
    }

    _pump(stub)

    assert stub.tasks.created == []  # no GPU / no specialist
    assert stub.backends["critic"].call_count == 0  # no Critic
    prog = stub.shared_state.framework_agent_phase_progress
    assert len(prog) == 1
    assert prog[0]["status"] == "already_present"
    assert prog[0]["provenance"] == "audit"
    lessons = tmp_path / "kb" / "framework_optimization" / "lessons.jsonl"
    assert lessons.exists()


def test_pump_audit_direct_apply_dispatches_executor_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """direct_apply audit -> raw-diff executor only, even with authoring enabled."""

    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)
    stub._framework_agent_roots_have_git = lambda: True  # hermetic: pretend git checkout
    stub._audit_verdict = {
        "semantic_status": "not_present",
        "applicability": "direct_apply",
        "recommended_next_step": "direct_framework",
        "confidence": 0.8,
        "evidence": [],
    }

    _pump_then_materialize(stub)

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds == ["framework_agent"]  # no specialist


def test_pump_audit_direct_apply_degrades_to_author_on_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """G3: direct_apply with no git checkout -> degrade to authoring specialist."""

    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)
    stub._framework_agent_roots_have_git = lambda: False  # wheel env (no git)
    stub._audit_verdict = {
        "applicability": "direct_apply",
        "recommended_next_step": "direct_framework",
        "confidence": 0.8,
        "evidence": [],
    }

    _pump_then_materialize(stub)

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds == ["specialist"]  # degraded to authoring


def test_pump_audit_skip_low_confidence_downgrades_to_author(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """G5: a low-confidence already-present skip must NOT skip; routes to authoring."""

    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)
    stub._audit_verdict = {
        "semantic_status": "already_equivalent",
        "applicability": "not_applicable",
        "recommended_next_step": "skip",
        "confidence": 0.5,  # below 0.8 floor
        "evidence": [{"local_file": "vllm/x.py", "symbol": "f", "reason": "maybe"}],
    }

    _pump_then_materialize(stub)

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds == ["specialist"]  # not skipped
    # No terminal already_present row was written (it wasn't skipped).
    assert not any(
        r.get("status") == "already_present" for r in stub.shared_state.framework_agent_phase_progress
    )


def test_pump_audit_skip_no_evidence_downgrades_to_author(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """G5: high confidence but no evidence -> not a safe skip."""

    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)
    stub._audit_verdict = {
        "recommended_next_step": "skip",
        "confidence": 0.99,
        "evidence": [],  # no concrete evidence
    }

    _pump_then_materialize(stub)

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds == ["specialist"]


def test_pump_audit_author_dispatches_specialist_only_with_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """needs_rewrite audit -> authoring specialist only, seeded with audit evidence."""

    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)
    stub._audit_verdict = {
        "semantic_status": "partially_present",
        "applicability": "needs_rewrite",
        "recommended_next_step": "author_via_specialist",
        "confidence": 0.5,
        "evidence": [
            {"local_file": "vllm/model_executor/layer.py", "symbol": "scaled_op", "reason": "drifted"},
        ],
        "risks": ["raw diff likely conflicts"],
    }

    _pump_then_materialize(stub)

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds == ["specialist"]  # no raw-diff executor
    notes = stub.tasks.created[0]["params"]["notes"]
    assert "AUDIT EVIDENCE" in notes
    assert "vllm/model_executor/layer.py" in notes
    assert "scaled_op" in notes


def test_audit_candidate_reuses_cached_verdict_no_reaudit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Resume idempotency: a candidate carrying ``_audit`` is not re-audited."""
    calls = SimpleNamespace(n=0)

    async def _phase_audit(**_: Any) -> dict[str, Any]:
        calls.n += 1
        return {"recommended_next_step": "author_via_specialist", "semantic_status": "not_present"}

    monkeypatch.setattr(_fa_client, "phase_audit", _phase_audit)
    stub = _Stub(tmp_path, authoring=True)

    cand = dict(_CANDIDATE)
    cand["_audit"] = {"recommended_next_step": "skip", "semantic_status": "already_equivalent"}

    out = asyncio.run(
        Coordinator._audit_framework_agent_candidate(stub, cand)  # type: ignore[arg-type]
    )
    assert out["recommended_next_step"] == "skip"  # cached verdict honoured
    assert calls.n == 0  # phase_audit not invoked


def test_audit_candidate_calls_phase_audit_when_uncached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Without a cached verdict, the real audit calls phase_audit and caches it."""
    calls = SimpleNamespace(n=0)

    async def _phase_audit(**_: Any) -> dict[str, Any]:
        calls.n += 1
        return {"recommended_next_step": "direct_framework", "semantic_status": "not_present"}

    monkeypatch.setattr(_fa_client, "phase_audit", _phase_audit)
    stub = _Stub(tmp_path, authoring=True)
    cand = dict(_CANDIDATE)

    out = asyncio.run(
        Coordinator._audit_framework_agent_candidate(stub, cand)  # type: ignore[arg-type]
    )
    assert calls.n == 1
    assert out["recommended_next_step"] == "direct_framework"
    assert cand["_audit"]["recommended_next_step"] == "direct_framework"  # cached on candidate


def test_audit_candidate_same_framework_omits_target_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate framework == session framework (the common case, incl. blank
    candidate.framework) -> phase_audit called without target_framework, exactly
    as before the cross-framework wiring; same-framework is unaffected."""
    captured: dict[str, Any] = {}

    async def _phase_audit(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"recommended_next_step": "direct_framework", "semantic_status": "not_present"}

    monkeypatch.setattr(_fa_client, "phase_audit", _phase_audit)
    stub = _Stub(tmp_path, authoring=True)
    cand = dict(_CANDIDATE)  # framework="sglang", matches _StateStub.framework

    asyncio.run(Coordinator._audit_framework_agent_candidate(stub, cand))  # type: ignore[arg-type]
    assert captured["framework"] == "sglang"
    assert captured["target_framework"] == ""
    assert captured["target_framework_source_roots"] is None


def test_audit_candidate_blank_framework_omits_target_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate with no framework stamp at all (most same-repo discoveries) ->
    resolves to session framework, same as before this wiring existed."""
    captured: dict[str, Any] = {}

    async def _phase_audit(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"recommended_next_step": "direct_framework", "semantic_status": "not_present"}

    monkeypatch.setattr(_fa_client, "phase_audit", _phase_audit)
    stub = _Stub(tmp_path, authoring=True)
    cand = dict(_CANDIDATE)
    cand["framework"] = ""

    asyncio.run(Coordinator._audit_framework_agent_candidate(stub, cand))  # type: ignore[arg-type]
    assert captured["framework"] == "sglang"  # falls back to session framework
    assert captured["target_framework"] == ""


def test_audit_candidate_cross_framework_sets_target_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate discovered from a DIFFERENT framework's
    repo must be audited in cross-framework mode against this session's own
    (target) framework, with the target's own live source roots attached."""
    captured: dict[str, Any] = {}

    async def _phase_audit(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"recommended_next_step": "author_via_specialist", "layer": "cross_framework"}

    monkeypatch.setattr(_fa_client, "phase_audit", _phase_audit)
    monkeypatch.setattr(_framework_paths, "resolve_source_file_allowlist", lambda: ["/src/sglang"])
    stub = _Stub(tmp_path, authoring=True)  # _StateStub.framework == "sglang"
    cand = dict(_CANDIDATE)
    cand["framework"] = "vllm"

    out = asyncio.run(Coordinator._audit_framework_agent_candidate(stub, cand))  # type: ignore[arg-type]
    assert captured["framework"] == "vllm"  # candidate's own source framework
    assert captured["target_framework"] == "sglang"  # this session's framework
    assert captured["target_framework_source_roots"] == ["/src/sglang"]
    assert out["layer"] == "cross_framework"


def test_framework_agent_repo_url_origin_framework_known() -> None:
    """Reverse-lookup resolves each repo_map-known framework's canonical repo URL."""
    assert Coordinator._framework_agent_repo_url_origin_framework(
        "https://github.com/ROCm/vllm.git"
    ) == "vllm"
    assert Coordinator._framework_agent_repo_url_origin_framework(
        "https://github.com/sgl-project/sglang.git"
    ) == "sglang"


def test_framework_agent_repo_url_origin_framework_unknown_or_kernel_repo() -> None:
    """Kernel-level pr_intel_specialist repos (aiter/triton/rccl) have no framework mapping."""
    assert Coordinator._framework_agent_repo_url_origin_framework("https://github.com/ROCm/aiter.git") == ""
    assert Coordinator._framework_agent_repo_url_origin_framework("") == ""
    assert Coordinator._framework_agent_repo_url_origin_framework("not-a-url") == ""


def test_discover_batch_tags_cross_repo_candidates_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate discovered from
    a DIFFERENT framework's repo (already queried via the pr_intel_specialist
    cross-repo set) gets tagged with its origin framework WITHOUT any env opt-in;
    the session's own-repo candidates are left untagged, exactly as before."""
    monkeypatch.delenv("FRAMEWORK_AGENT_CROSS_DISCOVER_TAG", raising=False)
    sglang_url = _fa_client.repo_url_for_framework("sglang")
    vllm_url = _fa_client.repo_url_for_framework("vllm")

    async def _discover(*, repo_url: str, **_: Any) -> dict[str, Any]:
        if repo_url == vllm_url:
            return {
                "batch_id": "b1",
                "candidates": [
                    {"pr_url": "https://github.com/ROCm/vllm/pull/9", "repo": "ROCm/vllm", "ref": "PR:9"}
                ],
            }
        return {
            "batch_id": "b1",
            "candidates": [dict(_CANDIDATE)],  # already framework="sglang"
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)
    monkeypatch.setattr(stub, "_framework_agent_discover_repo_urls", lambda framework: [sglang_url, vllm_url])

    ok = asyncio.run(Coordinator._discover_next_framework_batch(stub))  # type: ignore[arg-type]
    assert ok
    candidates = stub.shared_state.framework_agent_batches[-1]["candidates"]
    by_ref = {c["ref"]: c for c in candidates}
    assert by_ref["PR:9"]["framework"] == "vllm"
    assert by_ref["perf/moe"]["framework"] == "sglang"


def test_discover_batch_does_not_tag_cross_repo_candidates_when_kill_switch_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kill switch (FRAMEWORK_AGENT_CROSS_DISCOVER_TAG=0): cross-repo candidates keep
    whatever framework fa returned (blank in practice) — a full revert to the
    pre-wiring behaviour for rollback/safety."""
    monkeypatch.setenv("FRAMEWORK_AGENT_CROSS_DISCOVER_TAG", "0")
    sglang_url = _fa_client.repo_url_for_framework("sglang")
    vllm_url = _fa_client.repo_url_for_framework("vllm")

    async def _discover(*, repo_url: str, **_: Any) -> dict[str, Any]:
        if repo_url == vllm_url:
            return {
                "batch_id": "b1",
                "candidates": [
                    {"pr_url": "https://github.com/ROCm/vllm/pull/9", "repo": "ROCm/vllm", "ref": "PR:9"}
                ],
            }
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=True)
    monkeypatch.setattr(stub, "_framework_agent_discover_repo_urls", lambda framework: [sglang_url, vllm_url])

    ok = asyncio.run(Coordinator._discover_next_framework_batch(stub))  # type: ignore[arg-type]
    assert ok
    candidates = stub.shared_state.framework_agent_batches[-1]["candidates"]
    by_ref = {c["ref"]: c for c in candidates}
    assert by_ref["PR:9"].get("framework") in (None, "")


def test_pump_audit_author_with_authoring_disabled_falls_back_to_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """author_via_specialist + authoring disabled -> raw-diff executor fallback (no stranding)."""

    async def _discover(**_: Any) -> dict[str, Any]:
        return {"batch_id": "b1", "candidates": [dict(_CANDIDATE)]}

    monkeypatch.setattr(_fa_client, "phase_discover", _discover)
    stub = _Stub(tmp_path, authoring=False)
    stub._audit_verdict = {
        "semantic_status": "not_present",
        "applicability": "needs_rewrite",
        "recommended_next_step": "author_via_specialist",
        "confidence": 0.4,
        "evidence": [],
    }

    _pump_then_materialize(stub)

    kinds = [c["kind"] for c in stub.tasks.created]
    assert kinds == ["framework_agent"]


# --- cross-framework authoring seed / provenance --------------------


def test_authoring_specialist_cross_framework_seed(tmp_path: Path):
    """A cross_framework audit seeds the specialist with rewrite contract + provenance."""
    stub = _Stub(tmp_path)
    stub.shared_state.framework = "vllm"  # session (target) framework
    audit = {
        "layer": "cross_framework",
        "metrics": {"src_framework": "sglang", "dst_framework": "vllm"},
        "evidence": [
            {
                "dst_module": "vllm/core/block/prefix_caching_block.py",
                "src_path": "python/sglang/srt/mem_cache/radix_cache.py",
                "feature": "radix_prefix_cache",
            }
        ],
        "recommended_next_step": "author_via_specialist",
    }
    tid = asyncio.run(
        Coordinator._enqueue_framework_agent_authoring_specialist(stub, dict(_CANDIDATE), audit)  # type: ignore[arg-type]
    )
    assert tid
    params = stub.tasks.created[-1]["params"]
    assert params.get("cross_framework") is True
    assert params.get("source_framework") == "sglang"
    assert params.get("target_framework") == "vllm"
    notes = params.get("notes") or ""
    assert "CROSS-FRAMEWORK PORT" in notes
    assert "specialist:serving:framework:cross_framework:sglang->vllm" in notes
    assert "prefix_caching_block.py" in notes


def test_authoring_specialist_same_framework_no_cross(tmp_path: Path):
    """A non-cross audit must NOT stamp cross-framework params."""
    stub = _Stub(tmp_path)
    audit = {"recommended_next_step": "author_via_specialist"}  # no cross_framework layer
    tid = asyncio.run(
        Coordinator._enqueue_framework_agent_authoring_specialist(stub, dict(_CANDIDATE), audit)  # type: ignore[arg-type]
    )
    assert tid
    params = stub.tasks.created[-1]["params"]
    assert "cross_framework" not in params
    assert "CROSS-FRAMEWORK PORT" not in (params.get("notes") or "")
