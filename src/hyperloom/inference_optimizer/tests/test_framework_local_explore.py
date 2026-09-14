# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the FRAMEWORK_AGENT local-exploration arm (direct-dispatch path)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.phases import framework as _phase_framework
from hyperloom.orchestrator.phases import machine_state as _phase_state
from hyperloom.orchestrator.state.shared_state import SharedState

from ._optimize_fixtures import FakeCoordinator, optimize_state


def test_the_capped_phases_spend_the_session_on_the_work_phases():
    """The split covers the phases that carry a cap, and spends the session."""
    budget = _phase_state.DEFAULT_PHASE_BUDGET_PCT
    # ENABLEMENT is uncapped by design: a combo that cannot run has nothing to
    # optimise, so a share of the optimisation budget is the wrong unit for it.
    assert set(budget) == set(_phase_state.PHASE_NAMES) - {_phase_state.PHASE_ENABLEMENT}
    assert sum(budget.values()) <= 1.0
    # How the two work phases divide their share is a tuning call; that the session is spent on them rather than on
    # setup and wind-down is not.
    work = budget[_phase_state.PHASE_FRAMEWORK_AGENT] + budget[_phase_state.PHASE_KERNEL_AGENT]
    overhead = budget[_phase_state.PHASE_PRELUDE] + budget[_phase_state.PHASE_CLOSE]
    assert work >= 0.8
    assert overhead <= 0.1


# --------------------------------------------------------------------------- # Shared stub for the arm behavior
# --------------------------------------------------------------------------- #
def _state(*, authoring: bool, local_explore: bool) -> SharedState:
    """Real ``SharedState`` with both arms' switches set."""
    return optimize_state(
        framework_agent_authoring_enabled=authoring,
        framework_local_explore_enabled=local_explore,
        framework="sglang",
        gpu_type="MI300X",
        model_class="dense",
        precision="fp8",
        model="test-model",
    )


class _Tasks:
    def __init__(self) -> None:
        self._queued: list[Any] = []
        self._running: list[Any] = []
        self.created: list[dict[str, Any]] = []
        self._by_idem: dict[str, Any] = {}

    async def queued(self) -> list[Any]:
        return list(self._queued)

    async def running(self) -> list[Any]:
        return list(self._running)

    async def create_or_return_existing(self, **kwargs: Any) -> tuple[Any, bool]:
        self.created.append(kwargs)
        key = str(kwargs.get("idempotency_key") or "")
        existing = self._by_idem.get(key)
        if existing is not None:
            return existing, True
        task = SimpleNamespace(
            kind=kwargs.get("kind"),
            task_id=f"t-{len(self.created)}",
            params=kwargs.get("params") or {},
            state="queued",
        )
        self._queued.append(task)
        self._by_idem[key] = task
        return task, False


class _Bus:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def append_and_seq(self, msg: Any) -> Any:
        self.messages.append(msg)
        return msg

    async def tail(self, n: int = 200, **_: Any) -> list[Any]:
        return list(reversed(self.messages[-n:]))


class _Stub(FakeCoordinator):
    """The state the arm reads; the rest resolves to the real collaborators."""

    def __init__(self, tmp_path: Path, *, authoring: bool = True, local_explore: bool = True) -> None:
        state = _state(authoring=authoring, local_explore=local_explore)
        super().__init__(
            tmp_path,
            shared_state=state,
            state=SimpleNamespace(pending_proposals={}),
            tasks=_Tasks(),
            # No GPU pool: the specialist dispatch stays on the research lane.
            framework_gpu_pool=None,
            bus=_Bus(),
        )

    async def _warm_specialist_params(self, _params: dict[str, Any]) -> None:
        return None

    async def _framework_agent_authoring_inflight(self) -> bool:
        return False


# --------------------------------------------------------------------------- # 3.
def test_arm_disabled_dispatch_is_noop(tmp_path: Path):
    """When either arm flag is off the local-explore dispatch returns empty."""
    disabled_auth = _Stub(tmp_path, authoring=False, local_explore=True)
    assert disabled_auth._framework_local_explore_arm_enabled() is False

    disabled_arm = _Stub(tmp_path, authoring=True, local_explore=False)
    assert disabled_arm._framework_local_explore_arm_enabled() is False


def test_arm_enabled_when_both_flags_on(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    assert stub._framework_local_explore_arm_enabled() is True


def test_gap_compose_includes_framework_and_gpu(tmp_path: Path):
    """_compose_framework_local_explore_gap returns a gap string and keywords."""
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    gap, keywords = stub._compose_framework_local_explore_gap()
    assert isinstance(gap, str)
    assert isinstance(keywords, list)
    assert "sglang" in keywords


# --------------------------------------------------------------------------- # 4.
def test_local_explore_direct_dispatch_disabled_arm_creates_nothing(tmp_path: Path):
    """When authoring is off the pump falls through to phase_done without dispatching."""
    stub = _Stub(tmp_path, authoring=False, local_explore=True)
    stub.shared_state.framework_agent_empty_discoveries = _phase_framework.DISCOVER_FAILURE_RETRY_LIMIT
    asyncio.run(stub._pump_framework_agent_phase())
    assert stub.tasks.created == []
    assert stub.shared_state.framework_agent_phase_done is True


def test_local_explore_direct_dispatch_creates_specialist(tmp_path: Path):
    """The pump dispatches a local-explore specialist directly, without a pseudo-candidate."""
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    stub.shared_state.framework_agent_empty_discoveries = _phase_framework.DISCOVER_FAILURE_RETRY_LIMIT
    asyncio.run(stub._pump_framework_agent_phase())
    assert len(stub.tasks.created) == 1
    created = stub.tasks.created[0]
    assert created["kind"] == "specialist"
    params = created["params"]
    assert params["framework_agent_authoring"] is True
    assert params["framework_local_explore"] is True
    assert params["domain"] == "serving_specialist"
    assert params.get("task_kind") == "framework_local_explore"
    assert "allowed_tools" not in created
    # The specialist->candidate provenance map is recorded.
    cand_id = params.get("framework_agent_candidate_id", "")
    assert cand_id.startswith("local_explore:")
    assert stub.shared_state.framework_agent_specialist_candidate_map.get("t-1") == cand_id


def test_a_failed_local_explore_specialist_is_retried(tmp_path: Path):
    """One interrupted run must not retire the candidate; the loop picks :r1."""
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    stub.shared_state.framework_agent_empty_discoveries = _phase_framework.DISCOVER_FAILURE_RETRY_LIMIT
    asyncio.run(stub._pump_framework_agent_phase())
    first_task = stub.tasks._queued[-1]
    first_key = stub.tasks.created[-1]["idempotency_key"]
    # Mark as failed and remove from queued so _framework_agent_authoring_inflight
    # won't see it as still running and block the pump.
    first_task.state = "failed"
    stub.tasks._queued.remove(first_task)

    stub.shared_state.framework_agent_specialist_candidate_map = {}
    asyncio.run(stub._pump_framework_agent_phase())
    retry_key = stub.tasks.created[-1]["idempotency_key"]
    assert retry_key != first_key, f"expected new key, got same {retry_key!r}"
    # The retry key shares the base prefix.
    base = first_key.split(":r")[0]
    assert retry_key.startswith(base), f"{retry_key!r} should start with {base!r}"


def test_a_repeatedly_failing_local_explore_specialist_stops(tmp_path: Path):
    """Retrying is bounded: exhausting all attempts stops dispatching."""
    from hyperloom.orchestrator.phases.framework import _LOCAL_EXPLORE_MAX_ATTEMPTS

    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    stub.shared_state.framework_agent_empty_discoveries = _phase_framework.DISCOVER_FAILURE_RETRY_LIMIT
    for _ in range(_LOCAL_EXPLORE_MAX_ATTEMPTS):
        stub.shared_state.framework_agent_specialist_candidate_map = {}
        asyncio.run(stub._pump_framework_agent_phase())
        task = stub.tasks._queued[-1]
        task.state = "failed"
        stub.tasks._queued.remove(task)

    before = len(stub.tasks._queued)
    stub.shared_state.framework_agent_specialist_candidate_map = {}
    asyncio.run(stub._pump_framework_agent_phase())
    assert len(stub.tasks._queued) == before


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- # 6.
class _PumpStub(_Stub):
    """The pump with the enablement lane it shares the tick with shimmed out."""

    def __init__(self, tmp_path: Path, **kwargs: Any) -> None:
        super().__init__(tmp_path, **kwargs)
        # Discovery has come back empty its full retry budget, so the upstream lane declines and the tick reaches the
        # arm below it.
        self.shared_state.framework_agent_empty_discoveries = _phase_framework.DISCOVER_FAILURE_RETRY_LIMIT

    async def _maybe_enqueue_enablement_specialist(self) -> str:
        return ""


@pytest.mark.parametrize(
    ("empties", "local_explore", "expect"),
    [
        # Discovery outranks the arm: while the upstream lane still has budget to look, the tick belongs to it.
        (0, True, "candidate_discovery"),
        (0, False, "candidate_discovery"),
        # Discovery spent its retries. The arm takes over...
        (_phase_framework.DISCOVER_FAILURE_RETRY_LIMIT, True, "framework_local_explore"),
        # ...and with the arm off there is nothing left, so the source arm reports itself dry instead of idling.
        (_phase_framework.DISCOVER_FAILURE_RETRY_LIMIT, False, "phase_done"),
    ],
)
def test_an_empty_pool_walks_discovery_then_the_arm_then_done(
    tmp_path: Path,
    empties: int,
    local_explore: bool,
    expect: str,
):
    """The pump's ladder for an empty candidate pool, in precedence order."""
    stub = _Stub(tmp_path, authoring=True, local_explore=local_explore)
    stub.shared_state.framework_agent_empty_discoveries = empties
    stub._maybe_enqueue_enablement_specialist = lambda: _none()  # type: ignore[assignment]

    asyncio.run(stub._pump_framework_agent_phase())

    if expect == "phase_done":
        assert stub.tasks.created == []
        assert stub.shared_state.framework_agent_phase_done is True
    else:
        assert stub.shared_state.framework_agent_phase_done is False
        assert [c["params"].get("task_kind") for c in stub.tasks.created] == [expect]


async def _none() -> str:
    return ""


def test_pump_pivots_to_local_explore_on_discovery_failure(tmp_path: Path):
    stub = _PumpStub(tmp_path, authoring=True, local_explore=True)
    asyncio.run(stub._pump_framework_agent_phase())
    # The phase did NOT give up; a local-exploration specialist was dispatched.
    assert stub.shared_state.framework_agent_phase_done is False
    assert len(stub.tasks.created) == 1
    assert stub.tasks.created[0]["params"]["framework_local_explore"] is True
    assert stub.tasks.created[0]["params"]["framework_agent_candidate_id"] == "local_explore:0"


def test_pump_falls_back_to_exit_when_arm_disabled(tmp_path: Path):
    """No candidates, no discovery to be had, arm off -> the source arm is dry."""
    stub = _PumpStub(tmp_path, authoring=True, local_explore=False)
    asyncio.run(stub._pump_framework_agent_phase())
    assert stub.tasks.created == []
    assert stub.shared_state.framework_agent_phase_done is True


def test_forward_enablement_carriers_eval_origin():
    from hyperloom.orchestrator.phases.framework import _forward_enablement_carriers

    src = {
        "enablement_origin": "eval",
        "enablement_accuracy_floor": 0.4,
        "enablement_probe_config_path": "/runs/baseline/materialized.yaml",
        "enablement_eval_contract_fingerprint": "fp1",
    }
    dst: dict[str, Any] = {}
    _forward_enablement_carriers(src, dst)
    assert dst["enablement_origin"] == "eval"
    assert dst["enablement_accuracy_floor"] == 0.4
    assert dst["enablement_probe_config_path"] == "/runs/baseline/materialized.yaml"
    # The eval-contract fingerprint is no longer forwarded: nothing downstream reads it.
    assert "enablement_eval_contract_fingerprint" not in dst
    # Benches against the original workload config, not the shipped default.
    assert dst["config_path"] == "/runs/baseline/materialized.yaml"


def test_forward_enablement_carriers_boot_origin_noop():
    from hyperloom.orchestrator.phases.framework import _forward_enablement_carriers

    dst: dict[str, Any] = {}
    _forward_enablement_carriers({}, dst)
    assert dst == {}
    # An existing config_path is not overwritten for boot-origin.
    dst2 = {"config_path": "/keep.yaml"}
    _forward_enablement_carriers({"enablement_origin": ""}, dst2)
    assert dst2 == {"config_path": "/keep.yaml"}


# --------------------------------------------------------------------------- # Stage-3 guard: local_explore gap is
# registered and has a real canonical id --------------------------------------------------------------------------- #
def test_local_explore_gap_canonical_id_is_not_literal_local_explore():
    """The dispatched specialist must carry a per-candidate gap id."""
    stub = _Stub(Path("/tmp/t"), authoring=True, local_explore=True)
    stub.shared_state.framework_agent_empty_discoveries = _phase_framework.DISCOVER_FAILURE_RETRY_LIMIT
    asyncio.run(stub._pump_framework_agent_phase())
    assert len(stub.tasks.created) == 1
    params = stub.tasks.created[0]["params"]
    cid = params.get("gap_canonical_id", "")
    assert cid != "local_explore", "gap_canonical_id must not be the bare literal 'local_explore'"
    assert cid.startswith("gap.framework.local_explore."), f"expected gap.framework.local_explore.<id>, got {cid!r}"


def test_local_explore_dispatch_registers_gap_on_real_state():
    """upsert_gap is called during dispatch so find_gap resolves the new id."""
    tmp = Path("/tmp")
    stub = _Stub(tmp, authoring=True, local_explore=True)
    # Use the stub's already-wired SharedState with gap support.
    stub.shared_state.framework_agent_specialist_candidate_map = {}
    stub.shared_state.gaps = []

    # Call _enqueue_framework_agent_local_explore_specialist directly with a
    # synthetic candidate dict, bypassing the pump to keep this test focused.
    asyncio.run(
        stub._enqueue_framework_agent_local_explore_specialist(
            {
                "title": "local source exploration (gemm)",
                "repo": "(local source)",
                "framework": "sglang",
                "gap_description": "improve sglang gemm throughput on MI300X",
                "gap_keywords": ["sglang", "gemm"],
            },
            reason="test",
        )
    )

    assert len(stub.tasks.created) == 1
    params = stub.tasks.created[0]["params"]
    gap_cid = params["gap_canonical_id"]
    resolved = stub.shared_state.find_gap(gap_cid)
    assert resolved is not None, f"find_gap({gap_cid!r}) returned None; gap was not registered"
    assert resolved["layer"] == "framework"


def test_each_discovery_retry_takes_a_fresh_idempotency_key(tmp_path: Path):
    """Retries must not collide on one key, or the streak can never advance."""
    stub = _Stub(tmp_path, authoring=True, local_explore=False)
    keys = []
    for empties in range(_phase_framework.DISCOVER_FAILURE_RETRY_LIMIT):
        stub.shared_state.framework_agent_empty_discoveries = empties
        assert asyncio.run(stub._maybe_enqueue_candidate_discovery(reason="candidate_pool_empty")) is True
        keys.append(stub.tasks.created[-1]["idempotency_key"])
        # The dispatched round settles: nothing is in flight, so the next tick asks again and the key is all that
        # stands between a real retry and a re-fetch of the finished one.
        stub.tasks._queued.clear()
    assert len(set(keys)) == len(keys)

    # Budget spent: the lane declines so the rungs below it are reachable.
    stub.shared_state.framework_agent_empty_discoveries = _phase_framework.DISCOVER_FAILURE_RETRY_LIMIT
    assert asyncio.run(stub._maybe_enqueue_candidate_discovery(reason="candidate_pool_empty")) is False


def test_a_failed_discovery_round_is_not_an_empty_one(tmp_path: Path):
    """A crashed specialist reports nothing about what is out there."""
    stub = _Stub(tmp_path, authoring=True, local_explore=False)
    task = SimpleNamespace(task_id="t1", params={"candidate_discovery": True})

    stub._ingest_candidate_discovery(task=task, done_payload={}, run_error="specialist died")
    assert stub.shared_state.framework_agent_empty_discoveries == 0

    # A round that completed and genuinely found nothing still counts.
    stub._ingest_candidate_discovery(task=task, done_payload={"proposal_set": []})
    assert stub.shared_state.framework_agent_empty_discoveries == 1


def test_a_registry_that_cannot_answer_is_not_an_idle_one(tmp_path: Path):
    """The pump must not read a failed task query as \"nothing in flight\"."""
    stub = _Stub(tmp_path, authoring=True, local_explore=True)

    async def _boom():
        raise RuntimeError("registry unavailable")

    stub.tasks.queued = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        asyncio.run(stub._pump_framework_agent_phase())


def test_a_lane_that_cannot_run_retires_on_its_own_budget(tmp_path: Path):
    """Failed rounds get their own counter, a fresh key, and the same limit."""
    stub = _Stub(tmp_path, authoring=True, local_explore=False)
    task = SimpleNamespace(task_id="t1", params={"candidate_discovery": True})
    keys = []

    for _ in range(_phase_framework.DISCOVER_FAILURE_RETRY_LIMIT):
        assert asyncio.run(stub._maybe_enqueue_candidate_discovery(reason="candidate_pool_empty")) is True
        keys.append(stub.tasks.created[-1]["idempotency_key"])
        stub.tasks._queued.clear()
        stub._ingest_candidate_discovery(task=task, done_payload={}, run_error="no runner")

    assert len(set(keys)) == len(keys)
    assert stub.shared_state.framework_agent_empty_discoveries == 0
    assert asyncio.run(stub._maybe_enqueue_candidate_discovery(reason="candidate_pool_empty")) is False


def test_a_round_that_ran_clears_the_failure_streak(tmp_path: Path):
    """Whatever it came back with, it proves the lane works."""
    stub = _Stub(tmp_path, authoring=True, local_explore=False)
    task = SimpleNamespace(task_id="t1", params={"candidate_discovery": True})

    stub._ingest_candidate_discovery(task=task, done_payload={}, run_error="no runner")
    assert stub.shared_state.framework_agent_discover_failures == 1

    stub._ingest_candidate_discovery(task=task, done_payload={"proposal_set": []})
    assert stub.shared_state.framework_agent_discover_failures == 0
    assert stub.shared_state.framework_agent_empty_discoveries == 1


@pytest.mark.asyncio
async def test_a_commit_failure_after_a_keep_does_not_rearm_the_author(tmp_path: Path):
    """The patch applied, benched and was rolled back; only the commit failed.

    Reported as ``apply_failed`` it sent a specialist to rewrite a diff that
    had already passed the bench and the accuracy gate. Reported as the
    rollback it is, the rearm skips it on the status alone -- and the ledger,
    which skips ``apply_failed`` on a perf lane for the retry loop's sake,
    still records the candidate's outcome.
    """
    stub = _Stub(tmp_path, authoring=True, local_explore=False)
    res = {
        "status": "reverted",
        "error_class": "keep_commit_failed",
        "lane": "perf_framework",
        "candidate": {"candidate_id": "c1"},
        "specialist_task_id": "t1",
    }

    await stub._maybe_rearm_authored_lane(res)
    assert stub.shared_state.apply_fail_retry_pending == []

    # A genuine apply failure on the same lane still queues a retry.
    await stub._maybe_rearm_authored_lane({**res, "status": "apply_failed", "error_class": "patch_did_not_apply"})
    assert len(stub.shared_state.apply_fail_retry_pending) == 1


@pytest.mark.asyncio
async def test_an_apply_retry_is_scoped_to_the_cycle_that_queued_it(tmp_path: Path):
    """The gap id and the attempt number both repeat across macro-cycles."""
    stub = _Stub(tmp_path, authoring=True, local_explore=True)

    await stub._enqueue_author_specialist(lane="perf_explore", specialist_task_id="t-spec")
    stub.shared_state.macro_cycle = 2
    await stub._enqueue_author_specialist(lane="perf_explore", specialist_task_id="t-spec")

    keys = [c["idempotency_key"] for c in stub.tasks.created]
    assert len(set(keys)) == 2
    assert keys[1].endswith("-c2")


@pytest.mark.asyncio
async def test_the_retry_drain_carries_the_batch_the_failure_belonged_to(tmp_path: Path):
    """The candidate row does not always carry the batch the round ran under."""
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    stub.shared_state.apply_fail_retry_pending = [
        {
            "cand_id": "c1",
            "batch_id": "batch-7",
            "lane": "perf_framework",
            "attempt": 2,
            "candidate": {"candidate_id": "c1", "framework": "sglang"},
        }
    ]

    await stub._drain_apply_fail_retry_pending()

    keys = [c["idempotency_key"] for c in stub.tasks.created]
    assert any("batch-7" in k for k in keys), keys


def test_the_arms_deliverable_decides_its_lever():
    """This arm dispatches without a lever_kind, so the row's lever is derived.

    A config proposal and a diff are both valid returns from the same dispatch,
    and a row left with an empty lever is invisible to the dryness judgment.
    """
    from hyperloom.inference_optimizer.breakdown.agent_ownership import (
        LEVER_CONFIG,
        LEVER_SOURCE_PATCH,
    )
    from hyperloom.orchestrator.state.attempt_ledger import record_patch_attempt

    def _lever(**deliverable: Any) -> str:
        state = SharedState()
        record_patch_attempt(
            state,
            task_id="t-1",
            specialist_task_id="spec-1",
            outcome="reverted",
            gain_pct=None,
            before_tput=5000.0,
            after_tput=4900.0,
            error_class="",
            evidence={"framework_agent_candidate_id": "local_explore:0", **deliverable},
        )
        return str(state.attempts[0]["lever_kind"])

    assert _lever(patches_applied=["001_fix.patch"]) == LEVER_SOURCE_PATCH
    assert _lever() == LEVER_CONFIG
