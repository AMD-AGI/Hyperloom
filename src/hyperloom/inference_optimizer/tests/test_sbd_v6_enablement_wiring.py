# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The enablement event, recorded by the real lane rather than by hand.

:mod:`test_sbd_v6_enablement_timeline` drives the recorder directly; these tests drive the production
paths -- the dispatch, the rearm, the eval writeback, the build executor, the revalidation enqueue --
so a call site that stops recording fails here even when the recorder is still correct. Nothing owns
the lane's lifetime, so there is no recorder object whose absence would be obvious: the way a fact
goes missing is one call site quietly not making a call.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import enablement_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import enablement_event_parts
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    BASELINE_EVAL_ACCURACY_FLOOR_KEY,
    BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY,
    BASELINE_EVAL_EVIDENCE_KEY,
    BASELINE_EVAL_FAILURE_KIND_KEY,
    BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
)
from hyperloom.orchestrator.enablement.lane import EnablementLane
from hyperloom.orchestrator.loop.coordinator import _ENABLEMENT_MAX_ATTEMPTS, Coordinator
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound
from hyperloom.orchestrator.state.round_store import FAILED, RoundStore

_MISSING_ARCH_LOG = (
    "Traceback (most recent call last):\n"
    '  File "/opt/sglang/server.py", line 42, in load\n'
    "ValueError: Model architecture 'Glm5ForCausalLM' is not supported"
)


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so the lane records into it."""
    with session_scope(tmp_path):
        yield tmp_path


@pytest.fixture(autouse=True)
def _single_node(monkeypatch):
    """The lane is a no-op on multi-node, which is not what is under test."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)


@pytest.fixture(autouse=True)
def _no_candidate_discovery(monkeypatch):
    """Keep the real param builder off the network."""
    import hyperloom.agents.framework.sources as sources

    monkeypatch.setattr(sources, "enumerate_candidates", lambda _request: [])


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "enablement"]


def _ext() -> dict[str, Any]:
    """The lane's assembled ``ext``, whether or not it has closed yet."""
    ext, _status = enablement_event.assemble_enablement_ext(
        enablement_event_parts(),
        event=enablement_event.enablement_event_id(),
    )
    return ext


class _FakeTasks:
    """Enough of the task registry for the lane to open rows against."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create_or_return_existing(self, **kwargs: Any):
        self.created.append(kwargs)
        return types.SimpleNamespace(task_id=f"spec-{len(self.created)}", state="queued"), False

    async def get(self, task_id: str):
        from hyperloom.orchestrator.state.task_registry import TaskNotFound

        raise TaskNotFound(task_id)

    async def queued(self):
        return []

    async def running(self):
        return []


def _rounds(session_dir: Path) -> RoundStore:
    """A real round ledger, which is what the lane now counts stalls from."""
    db = SqliteConnection(Path(session_dir) / "coordinator.db")
    ensure_schema(db.raw)
    return RoundStore(db)


async def _seed_stalled(rounds: RoundStore, n: int) -> None:
    """Settle N FAILED rounds, the ledger's way of saying the streak is N."""
    for i in range(n):
        rid = f"seed-stalled-{i:03d}"
        holder = f"holder-{i:03d}"
        await rounds.open(rid, holder_task_id=holder, lease_sec=3600.0, now_unix=float(i), request_id=rid)
        await rounds.settle(
            rid,
            holder_task_id=holder,
            fence=1,
            outcome=FAILED,
            now_unix=float(i) + 1.0,
            request_id=f"settle-{rid}",
        )


def _lane(session_dir: Path, **overrides: Any):
    """A lane bound to the real dispatch, rearm and revalidation methods."""
    state = types.SimpleNamespace(
        framework="sglang",
        model_name="zai-org/GLM-5",
        model_path="",
        reference_model="",
        gpu_type="mi300x",
        tp=8,
        max_model_len=8192,
        enablement_mode=overrides.get("mode", "all"),
        enablement=EnablementRound(
            origin=overrides.get("origin", ""),
            launch_log=overrides.get("launch_log", _MISSING_ARCH_LOG),
            last_specialist_task_id=overrides.get("last_specialist_task_id", ""),
            validation_pending=overrides.get("validation_pending", False),
            revalidation_generation=overrides.get("revalidation_generation", 0),
            accepted_config_path=overrides.get("accepted_config_path", ""),
            accepted_config=overrides.get("accepted_config", {}),
            accuracy_floor=overrides.get("accuracy_floor", 0.0),
            human_review_logged=[],
        ),
        baseline_tput=0.0,
        baseline_failure_streak=1,
        baseline_arg_error_streak=0,
        baseline_total_failures=0,
        tick=0,
        stop_reason="",
        save=lambda *a, **k: None,
    )
    state.set_stop_reason = lambda value, **k: setattr(state, "stop_reason", str(value or ""))

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    fake = types.SimpleNamespace(
        shared_state=state,
        state=types.SimpleNamespace(pending_proposals={}),
        tasks=_FakeTasks(),
        rounds=overrides.get("rounds") or _rounds(session_dir),
        session_dir=str(session_dir),
        _run_deadline=None,
        _warm_specialist_params=_noop,
        _record_observation=_noop,
        _maybe_enqueue_specialist_requested_build=_noop,
        _maybe_escalate_to_targeted_build=_noop,
        _read_enablement_source_context=lambda _sig: "",
        _derive_checkpoint_weight_facts=lambda _log: "",
        _framework_gpu_params=lambda: {},
        _framework_authoring_lanes_ttl=lambda params, *, base_ttl_sec: (["research_lane"], base_ttl_sec),
        _time_budget_denial_for_action=lambda _action: None,
        action_registry=ACTION_CATALOGUE,
        # The host preflight would stat a checkpoint named by the ambient ``MODEL_PATH``, which belongs
        # to whichever test ran before this one, so the host answers that it cannot tell.
        _environment_verdict=lambda: None,
    )
    for name in (
        "_registry_lanes_ttl",
        "_build_enablement_specialist_params",
        "_discover_enablement_candidate_refs",
        "_maybe_enqueue_enablement_specialist",
        "_maybe_record_enablement_human_review",
        "_maybe_rearm_enablement",
        "_maybe_enqueue_enablement_baseline_revalidation",
        "_open_revalidation_row",
        "_open_row_past_spent_generations",
        "_open_round_past_spent_generations",
    ):
        setattr(fake, name, types.MethodType(getattr(Coordinator, name), fake))
    # The round ledger's own surface: the cap, the lease and the settle all live on it.
    for name in (
        "_refused_argv_is_terminal",
        "_environment_fault_is_terminal",
        "_enablement_in_flight",
        "_round_has_live_work",
        "_open_authoring_round",
        "_renew_enablement_round",
        "_settle_enablement_round",
    ):
        setattr(fake, name, types.MethodType(getattr(EnablementLane, name), fake))
    fake._close_enablement_lane = types.MethodType(WritebackCollaborator._close_enablement_lane, fake)
    return fake


def _writeback(session_dir: Path, **overrides: Any):
    """A writeback bound to the real eval-failure persistence."""
    state = types.SimpleNamespace(
        enablement_mode=overrides.get("mode", "all"),
        enablement=EnablementRound(
            validation_pending=overrides.get("validation_pending", False),
            revalidation_generation=overrides.get("revalidation_generation", 0),
            baseline_eval_kind=overrides.get("baseline_eval_kind", ""),
            observed_accuracy=overrides.get("observed_accuracy", 0.0),
            observed_task=overrides.get("observed_task", ""),
        ),
        stop_reason="",
    )
    state.set_stop_reason = lambda value, **k: setattr(state, "stop_reason", str(value or ""))
    fake = types.SimpleNamespace(
        shared_state=state,
        rounds=overrides.get("rounds") or _rounds(session_dir),
    )
    for name in ("_persist_eval_failure", "_record_enablement_eval_trigger", "_close_enablement_lane"):
        setattr(fake, name, types.MethodType(getattr(WritebackCollaborator, name), fake))
    return fake


@pytest.mark.asyncio
async def test_the_real_dispatch_records_the_kind_it_classified(_bound_session):
    lane = _lane(_bound_session)

    task_id = await lane._maybe_enqueue_enablement_specialist()

    assert task_id
    rows = _ext()["attempts"]["rows"]
    assert len(rows) == 1
    assert rows[0]["task_id"] == task_id
    assert rows[0]["attempt"] == 1
    assert rows[0]["failure_kind"] == "missing_model_arch"
    assert "Glm5ForCausalLM" in rows[0]["launch_log_excerpt"]


@pytest.mark.asyncio
async def test_the_dispatch_opens_the_lane_the_trigger_missed(_bound_session):
    lane = _lane(_bound_session, mode="launch")

    await lane._maybe_enqueue_enablement_specialist()

    events = _events(_bound_session)
    assert len(events) == 1
    ext = _ext()
    assert ext["mode"] == "launch"
    assert ext["origin"] == enablement_event.ORIGIN_BOOT


@pytest.mark.asyncio
async def test_two_dispatches_are_two_rounds_on_one_event(_bound_session):
    lane = _lane(_bound_session)
    first = await lane._maybe_enqueue_enablement_specialist()
    await lane._maybe_rearm_enablement({"enablement": True, "status": "reverted", "specialist_task_id": first})
    second = await lane._maybe_enqueue_enablement_specialist()

    assert len(_events(_bound_session)) == 1
    rows = _ext()["attempts"]["rows"]
    assert [row["task_id"] for row in rows] == [first, second]
    assert [row["attempt"] for row in rows] == [1, 2]


@pytest.mark.asyncio
async def test_a_kept_round_closes_the_lane_it_landed(_bound_session):
    lane = _lane(_bound_session)
    task_id = await lane._maybe_enqueue_enablement_specialist()

    await lane._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "kept",
            "specialist_task_id": task_id,
            "patches_applied": ["/s/patches/arch.diff"],
            "setup_commands_applied": ["pip install -e ."],
            "framework_root": "/fw/sglang",
        }
    )

    events = _events(_bound_session)
    assert len(events) == 1
    assert events[0]["status"] == "succeeded"
    ext = events[0]["ext"]
    assert ext["attempts"]["landed"] == 1
    assert ext["attempts"]["rows"][0]["failure_kind"] == "missing_model_arch"
    assert ext["result"]["outcome"] == enablement_event.OUTCOME_SUCCEEDED
    assert ext["result"]["kept_patches"] == ["/s/patches/arch.diff"]
    assert ext["result"]["framework_root"] == "/fw/sglang"


@pytest.mark.asyncio
async def test_an_advanced_round_records_the_gap_it_revealed(_bound_session):
    lane = _lane(_bound_session)
    task_id = await lane._maybe_enqueue_enablement_specialist()

    await lane._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "advanced": True,
            "specialist_task_id": task_id,
            "patches_applied": ["/s/patches/arch.diff"],
            "enablement_launch_log": "ValueError: Following weights were not initialized from checkpoint",
        }
    )

    row = _ext()["attempts"]["rows"][0]
    assert row["advanced"] is True
    assert "Glm5ForCausalLM" in row["launch_log_excerpt"]
    assert "not initialized from checkpoint" in row["next_launch_log_excerpt"]
    assert _events(_bound_session)[0]["status"] == "running"


@pytest.mark.asyncio
async def test_the_stall_cap_closes_the_lane_as_failed(_bound_session):
    lane = _lane(_bound_session)
    for _ in range(_ENABLEMENT_MAX_ATTEMPTS):
        task_id = await lane._maybe_enqueue_enablement_specialist()
        lane.shared_state.enablement.last_specialist_task_id = task_id
        await lane._maybe_rearm_enablement({"enablement": True, "status": "reverted", "specialist_task_id": task_id})
    await lane._maybe_enqueue_enablement_specialist()

    assert lane.shared_state.stop_reason == "enablement_attempts_exhausted"
    events = _events(_bound_session)
    assert events[0]["status"] == "failed"
    assert events[0]["ext"]["result"]["reason"] == "enablement_attempts_exhausted"
    assert events[0]["ext"]["attempts"]["count"] == _ENABLEMENT_MAX_ATTEMPTS
    assert events[0]["ext"]["attempts"]["landed"] == 0


@pytest.mark.asyncio
async def test_an_eval_origin_keep_does_not_close_the_lane(_bound_session):
    lane = _lane(_bound_session, origin="eval")
    lane.shared_state.enablement.last_specialist_task_id = "spec-1"

    await lane._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "kept",
            "specialist_task_id": "spec-1",
            "enablement_accepted_config_path": "/s/accepted.yaml",
        }
    )

    assert lane.shared_state.enablement.succeeded is False
    assert lane.shared_state.enablement.validation_pending is True
    events = _events(_bound_session)
    assert events[0]["status"] == "running"
    row = _ext()["attempts"]["rows"][0]
    assert row["status"] == "kept"
    assert row["validation_pending"] is True
    assert row["landed"] is False
    # Nothing was there to copy, so the row names nothing rather than the
    # ``runs/`` path the round reported.
    assert row["files"] == []
    assert row["accepted_config_path"] is None


@pytest.mark.asyncio
async def test_the_rearm_hands_the_round_its_archive(_bound_session):
    lane = _lane(_bound_session)
    lane.shared_state.enablement.last_specialist_task_id = "spec-1"
    config = _bound_session / "runs" / "integrate_patch" / "spec-1" / "integrate_patch.with_envs.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("tp: 8\n", encoding="utf-8")

    await lane._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "kept",
            "specialist_task_id": "spec-1",
            "enablement_accepted_config_path": str(config),
        }
    )

    rows = _ext()["attempts"]["rows"]
    # The archive and the verdict are two calls on one tick, onto one row.
    assert len(rows) == 1
    assert {"path": "reports/enablement/spec-1/launch_config.yaml", "role": "launch_config"} in rows[0]["files"]
    assert rows[0]["accepted_config_path"] == "reports/enablement/spec-1/launch_config.yaml"
    assert rows[0]["status"] == "kept"


@pytest.mark.asyncio
async def test_a_snapshot_that_raised_leaves_the_row_silent(_bound_session, monkeypatch):
    monkeypatch.setattr(
        "hyperloom.orchestrator.enablement.lane.snapshot_round",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no space left on device")),
    )
    lane = _lane(_bound_session)
    lane.shared_state.enablement.last_specialist_task_id = "spec-1"

    await lane._maybe_rearm_enablement({"enablement": True, "status": "reverted", "specialist_task_id": "spec-1"})

    # Absent, not empty: an archive that blew up did not establish that
    # nothing landed. The round is still ruled.
    row = _ext()["attempts"]["rows"][0]
    assert "files" not in row
    assert "accepted_config_path" not in row
    assert row["status"] == "reverted"


@pytest.mark.asyncio
async def test_a_round_still_open_leaves_one_unruled_row(_bound_session):
    lane = _lane(_bound_session)
    first = await lane._maybe_enqueue_enablement_specialist()
    # The specialist row is gone from the registry, but the round it holds is still open.
    assert await lane._maybe_enqueue_enablement_specialist() == ""

    rows = _ext()["attempts"]["rows"]
    assert [row["task_id"] for row in rows] == [first]
    assert rows[0]["failure_kind"] == "missing_model_arch"
    assert rows[0].get("status") in (None, "")


@pytest.mark.asyncio
async def test_an_unclassifiable_failure_is_recorded_without_a_round(_bound_session):
    lane = _lane(_bound_session)
    enablement_event.record_trigger(origin=enablement_event.ORIGIN_BOOT, mode="all", kind="unknown")

    await lane._maybe_record_enablement_human_review("Segmentation fault (core dumped)")

    ext = _ext()
    assert ext["attempts"]["count"] == 0
    assert ext["human_review"]["count"] == 1
    row = ext["human_review"]["rows"][0]
    assert row["failure_kind"]
    assert "human triage" in row["reason"]


@pytest.mark.asyncio
async def test_the_same_unclassifiable_failure_is_filed_once(_bound_session):
    lane = _lane(_bound_session)
    for _ in range(3):
        await lane._maybe_record_enablement_human_review("Segmentation fault (core dumped)")

    assert _ext()["human_review"]["count"] == 1


def test_the_eval_writeback_opens_the_lane_with_what_it_measured(_bound_session):
    writeback = _writeback(_bound_session)

    writeback._persist_eval_failure(
        {
            BASELINE_EVAL_FAILURE_KIND_KEY: "accuracy_below_floor",
            BASELINE_EVAL_OBSERVED_ACCURACY_KEY: 0.21,
            BASELINE_EVAL_ACCURACY_FLOOR_KEY: 0.5,
            BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY: "fp-abc",
            BASELINE_EVAL_EVIDENCE_KEY: "gsm8k exact_match 0.21",
            "accuracy_task": "gsm8k",
            "accuracy_metric": "exact_match",
            "materialized_config": "/s/runs/baseline/materialized.yaml",
        }
    )

    ext = _ext()
    assert ext["origin"] == enablement_event.ORIGIN_EVAL
    trigger = ext["trigger"]
    assert trigger["kind"] == "accuracy_below_floor"
    assert trigger["observed_accuracy"] == 0.21
    assert trigger["accuracy_floor"] == 0.5
    assert trigger["observed_task"] == "gsm8k"
    assert trigger["eval_contract_fingerprint"] == "fp-abc"
    assert trigger["probe_config_path"] == "/s/runs/baseline/materialized.yaml"


def test_an_eval_less_rebaseline_cannot_downgrade_the_trigger(_bound_session):
    writeback = _writeback(_bound_session)
    writeback._persist_eval_failure(
        {
            BASELINE_EVAL_FAILURE_KIND_KEY: "accuracy_below_floor",
            BASELINE_EVAL_OBSERVED_ACCURACY_KEY: 0.21,
            BASELINE_EVAL_ACCURACY_FLOOR_KEY: 0.5,
            "accuracy_task": "gsm8k",
        }
    )
    writeback._persist_eval_failure({BASELINE_EVAL_FAILURE_KIND_KEY: "accuracy_unavailable"})

    trigger = _ext()["trigger"]
    assert trigger["kind"] == "accuracy_below_floor"
    assert trigger["observed_accuracy"] == 0.21


def test_a_failed_revalidation_closes_the_window_it_was_opened_for(_bound_session):
    writeback = _writeback(
        _bound_session,
        validation_pending=True,
        revalidation_generation=2,
        baseline_eval_kind="accuracy_below_floor",
    )

    writeback._persist_eval_failure(
        {
            BASELINE_EVAL_FAILURE_KIND_KEY: "accuracy_below_floor",
            BASELINE_EVAL_OBSERVED_ACCURACY_KEY: 0.33,
            BASELINE_EVAL_ACCURACY_FLOOR_KEY: 0.5,
            "accuracy_task": "gsm8k",
        }
    )

    revalidations = _ext()["revalidations"]["rows"]
    assert len(revalidations) == 1
    assert revalidations[0]["generation"] == 2
    assert revalidations[0]["promoted"] is False
    assert revalidations[0]["accuracy"] == 0.33


def test_a_failed_revalidation_is_not_itself_the_lanes_terminal(_bound_session):
    writeback = _writeback(
        _bound_session,
        validation_pending=True,
        revalidation_generation=3,
        baseline_eval_kind="accuracy_below_floor",
    )

    writeback._persist_eval_failure(
        {
            BASELINE_EVAL_FAILURE_KIND_KEY: "accuracy_below_floor",
            BASELINE_EVAL_ACCURACY_FLOOR_KEY: 0.5,
            "accuracy_task": "gsm8k",
        }
    )

    assert writeback.shared_state.stop_reason == ""
    assert _events(_bound_session)[0]["status"] == "running"


@pytest.mark.asyncio
async def test_the_revalidation_enqueue_records_the_window_it_opened(_bound_session):
    lane = _lane(
        _bound_session,
        origin="eval",
        validation_pending=True,
        revalidation_generation=1,
        accepted_config_path="/s/accepted.yaml",
    )

    task_id = await lane._maybe_enqueue_enablement_baseline_revalidation()

    assert task_id
    rows = _ext()["revalidations"]["rows"]
    assert len(rows) == 1
    assert rows[0]["generation"] == 1
    assert rows[0]["task_id"] == task_id
    assert rows[0]["config_path"] == "/s/accepted.yaml"
    assert "closed_at" not in rows[0]


def test_the_build_executor_records_the_build_it_ran(_bound_session):
    from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor

    result = types.SimpleNamespace(
        ok=False,
        failure_class="compile_error",
        failure_summary="hipcc: unsupported arch",
        error="",
        attempt_root="/s/enablement/builds/build-7",
        to_state=lambda: {
            "ok": False,
            "failure_class": "compile_error",
            "failure_summary": "hipcc: unsupported arch",
            "action": {"component": "aiter", "gpu_arch": "gfx942", "max_jobs": 32},
            "installed_versions": {"aiter_ref": "abc123"},
            "attempt_root": "/s/enablement/builds/build-7",
        },
    )
    enablement_event.record_trigger(origin=enablement_event.ORIGIN_BOOT, mode="all", kind="compiled_miss")

    TargetedBuildExecutor._record_result(result, None, task_id="build-7")

    builds = _ext()["builds"]
    assert builds["count"] == 1
    assert builds["failed"] == 1
    assert builds["rows"][0]["task_id"] == "build-7"
    assert builds["rows"][0]["component"] == "aiter"
    assert builds["rows"][0]["ref"] == "abc123"


@pytest.mark.asyncio
async def test_a_lane_with_no_session_bound_still_dispatches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.session.session_binding.bound_session_or_none",
        lambda: None,
    )
    lane = _lane(tmp_path)

    task_id = await lane._maybe_enqueue_enablement_specialist()
    await lane._maybe_rearm_enablement({"enablement": True, "status": "kept", "specialist_task_id": task_id})

    assert task_id
    assert lane.shared_state.enablement.succeeded is True
