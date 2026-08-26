# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the FRAMEWORK_AGENT local-exploration arm.

Covers the phase-budget split, the candidate-free pseudo-candidate and its id
sequencing, the discovery-exhaustion pivot to a local-exploration specialist
(instead of flipping ``framework_agent_phase_done``), and the resident arm
offering the pseudo-candidate to the ranker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.framework import client as _fa_client
from hyperloom.orchestrator.phases import machine_state as _phase_state
from hyperloom.orchestrator.phases.framework import FrameworkPhase
from hyperloom.orchestrator.state.shared_state import SharedState

from ._optimize_fixtures import FakeCoordinator, optimize_state


def test_every_phase_gets_a_budget_share_and_the_shares_sum_to_one():
    """The split covers exactly the phases that exist, and spends the session.

    Stated over ``PHASE_NAMES`` rather than as a list of numbers: a phase added
    to the machine without a budget row would otherwise run to whatever it
    costs, which is how PRELUDE once took 73% of a three-hour session.
    """
    budget = _phase_state.DEFAULT_PHASE_BUDGET_PCT
    assert set(budget) == set(_phase_state.PHASE_NAMES)
    assert sum(budget.values()) == pytest.approx(1.0)
    # The merged optimisation phase carries both levers, so it holds the
    # largest share; rotation between the arms is their plateau judgement, not
    # a wall-clock cap.
    assert max(budget, key=lambda p: budget[p]) == _phase_state.PHASE_FRAMEWORK_AGENT


# --------------------------------------------------------------------------- #
# Shared stub for the arm behavior
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


# --------------------------------------------------------------------------- #
# 3. Pseudo-candidate + id sequencing
# --------------------------------------------------------------------------- #
def test_pseudo_candidate_none_when_arm_disabled(tmp_path: Path):
    disabled_auth = _Stub(tmp_path, authoring=False, local_explore=True)
    assert disabled_auth._make_local_explore_pseudo_candidate() is None
    disabled_arm = _Stub(tmp_path, authoring=True, local_explore=False)
    assert disabled_arm._make_local_explore_pseudo_candidate() is None


def test_pseudo_candidate_shape_when_enabled(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    pseudo = stub._make_local_explore_pseudo_candidate()
    assert pseudo is not None
    assert pseudo["kind"] == FrameworkPhase._LOCAL_EXPLORE_KIND
    assert pseudo["candidate_id"] == "local_explore:0"
    assert pseudo["framework"] == "sglang"
    # Gap composed from workload taxonomy (framework / gpu / arch / precision).
    assert isinstance(pseudo["gap_keywords"], list)
    assert "sglang" in pseudo["gap_keywords"]


def test_next_local_explore_id_increments_with_progress(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    assert stub._next_local_explore_candidate_id() == "local_explore:0"
    stub.shared_state.framework_agent_phase_progress.append(
        {"candidate_id": "local_explore:0", "status": "author_empty", "kept": False}
    )
    assert stub._next_local_explore_candidate_id() == "local_explore:1"
    # Non-local rows do not advance the sequence.
    stub.shared_state.framework_agent_phase_progress.append(
        {"candidate_id": "https://example.com/pr/9", "status": "reverted", "kept": False}
    )
    assert stub._next_local_explore_candidate_id() == "local_explore:1"


# --------------------------------------------------------------------------- #
# 4. Dispatch behavior
# --------------------------------------------------------------------------- #
def test_maybe_dispatch_local_explore_disabled_is_noop(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=False, local_explore=True)
    dispatched = asyncio.run(stub._maybe_dispatch_local_explore(reason="discover_exhausted"))
    assert dispatched is False
    assert stub.tasks.created == []


def test_maybe_dispatch_local_explore_enabled_creates_specialist(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    dispatched = asyncio.run(stub._maybe_dispatch_local_explore(reason="discover_exhausted"))
    assert dispatched is True
    assert len(stub.tasks.created) == 1
    created = stub.tasks.created[0]
    assert created["kind"] == "specialist"
    params = created["params"]
    assert params["framework_agent_authoring"] is True
    assert params["framework_local_explore"] is True
    assert params["framework_agent_candidate_id"] == "local_explore:0"
    assert params["domain"] == "serving_specialist"
    # Boilerplate is in _TASK_KIND_BRIEFS; notes is empty on a fresh dispatch.
    assert params.get("task_kind") == "framework_local_explore"
    assert params.get("notes", "") == ""
    # The per-task tool whitelist is gone; the specialist tool policy is a denylist.
    assert "allowed_tools" not in created
    # Idempotency keyed on the candidate id.
    assert created["idempotency_key"] == "framework_agent_local_explore:local_explore:0"
    # The specialist->candidate provenance map is recorded.
    assert stub.shared_state.framework_agent_specialist_candidate_map == {"t-1": "local_explore:0"}


def test_a_candidate_whose_specialist_failed_is_dispatched_again(tmp_path: Path):
    """One interrupted run must not retire a candidate.

    The registry de-duplicates by key and returns whatever row it finds, so a
    candidate whose specialist failed kept resolving to that failure: the phase
    re-selected it every tick, logged a dispatch, and nothing ran. A restart
    reclaims an in-flight specialist as failed, so a single stop was enough to
    park the phase on a candidate it could never start — and it held the whole
    framework budget while doing it.
    """
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    assert asyncio.run(stub._maybe_dispatch_local_explore(reason="discover_exhausted")) is True
    first = stub.tasks._queued[-1]
    first.state = "failed"

    stub.shared_state.framework_agent_specialist_candidate_map = {}
    assert asyncio.run(stub._maybe_dispatch_local_explore(reason="discover_exhausted")) is True

    retry = stub.tasks.created[-1]
    assert retry["idempotency_key"] != "framework_agent_local_explore:local_explore:0"
    assert retry["idempotency_key"].startswith("framework_agent_local_explore:local_explore:0")
    assert stub.tasks._queued[-1] is not first


def test_a_candidate_that_keeps_failing_is_left_for_the_phase_to_replace(tmp_path: Path):
    """Retrying is bounded: a candidate that cannot author is not worth the
    wall clock the framework budget is there to spend."""
    from hyperloom.orchestrator.phases.framework import _LOCAL_EXPLORE_MAX_ATTEMPTS

    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    for _ in range(_LOCAL_EXPLORE_MAX_ATTEMPTS):
        stub.shared_state.framework_agent_specialist_candidate_map = {}
        asyncio.run(stub._maybe_dispatch_local_explore(reason="discover_exhausted"))
        stub.tasks._queued[-1].state = "failed"

    before = len(stub.tasks._queued)
    stub.shared_state.framework_agent_specialist_candidate_map = {}
    asyncio.run(stub._maybe_dispatch_local_explore(reason="discover_exhausted"))

    assert len(stub.tasks._queued) == before


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 6. Pump pivot: discovery exhaustion dispatches local-explore, not phase_done
# --------------------------------------------------------------------------- #
class _PumpStub(_Stub):
    """The pump with the enablement lane it shares the tick with shimmed out."""

    def __init__(self, tmp_path: Path, **kwargs: Any) -> None:
        super().__init__(tmp_path, **kwargs)
        # Discovery has come back empty its full retry budget, so the upstream
        # lane declines and the tick reaches the arm below it.
        self.shared_state.framework_agent_empty_discoveries = _fa_client.DISCOVER_FAILURE_RETRY_LIMIT

    async def _maybe_enqueue_enablement_specialist(self) -> str:
        return ""


@pytest.mark.parametrize(
    ("empties", "local_explore", "expect"),
    [
        # Discovery outranks the arm: while the upstream lane still has budget
        # to look, the tick belongs to it.
        (0, True, "candidate_discovery"),
        (0, False, "candidate_discovery"),
        # Discovery spent its retries. The arm takes over...
        (_fa_client.DISCOVER_FAILURE_RETRY_LIMIT, True, "framework_local_explore"),
        # ...and with the arm off there is nothing left, so the source arm
        # reports itself dry instead of idling.
        (_fa_client.DISCOVER_FAILURE_RETRY_LIMIT, False, "phase_done"),
    ],
)
def test_an_empty_pool_walks_discovery_then_the_arm_then_done(
    tmp_path: Path,
    empties: int,
    local_explore: bool,
    expect: str,
):
    """The pump's ladder for an empty candidate pool, in precedence order.

    Each rung must be reachable: while discovery answered "asked again"
    unconditionally the two below it were dead code, and the source arm had no
    way to tell ``exit_normal_optimize`` it had plateaued.
    """
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
    """No candidates, no discovery to be had, arm off -> the source arm is dry.

    The pump must reach a terminal answer rather than idle: with nothing left
    to dispatch, ``framework_agent_phase_done`` is what lets
    ``exit_normal_optimize`` consider the source arm plateaued.
    """
    stub = _PumpStub(tmp_path, authoring=True, local_explore=False)
    asyncio.run(stub._pump_framework_agent_phase())
    assert stub.tasks.created == []
    assert stub.shared_state.framework_agent_phase_done is True


def test_forward_enablement_carriers_eval_origin():
    from hyperloom.orchestrator.phases.explore import _forward_enablement_carriers

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
    # The eval-contract fingerprint is no longer forwarded: nothing downstream
    # reads it. Correctness is judged from the candidate's own measurement.
    assert "enablement_eval_contract_fingerprint" not in dst
    # Benches against the original workload config, not the shipped default.
    assert dst["config_path"] == "/runs/baseline/materialized.yaml"


def test_forward_enablement_carriers_boot_origin_noop():
    from hyperloom.orchestrator.phases.explore import _forward_enablement_carriers

    dst: dict[str, Any] = {}
    _forward_enablement_carriers({}, dst)
    assert dst == {}
    # An existing config_path is not overwritten for boot-origin.
    dst2 = {"config_path": "/keep.yaml"}
    _forward_enablement_carriers({"enablement_origin": ""}, dst2)
    assert dst2 == {"config_path": "/keep.yaml"}


# --------------------------------------------------------------------------- #
# Stage-3 guard: local_explore gap is registered and has a real canonical id
# --------------------------------------------------------------------------- #
def test_pseudo_candidate_gap_canonical_id_is_not_literal_local_explore():
    """The pseudo-candidate must carry a per-candidate gap id, not the old
    literal 'local_explore' string that prevented find_gap from matching."""
    stub = _Stub(Path("/tmp/t"), authoring=True, local_explore=True)
    pseudo = stub._make_local_explore_pseudo_candidate()
    assert pseudo is not None
    cid = pseudo["gap_canonical_id"]
    assert cid != "local_explore", "gap_canonical_id must not be the bare literal 'local_explore'"
    assert cid.startswith("gap.framework.local_explore."), f"expected gap.framework.local_explore.<id>, got {cid!r}"


def test_local_explore_dispatch_registers_gap_on_real_state():
    """upsert_gap is called during dispatch so find_gap resolves the new id."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    tmp = Path("/tmp")
    shared = SharedState(session_id="test-le-gap")
    stub = _Stub(tmp, authoring=True, local_explore=True)
    # Patch in a real SharedState that supports upsert_gap.
    stub.shared_state = shared
    stub.state = shared
    shared.framework_agent_authoring_enabled = True
    shared.framework_local_explore_enabled = True
    shared.framework_agent_batches = []
    shared.framework_agent_phase_progress = []
    shared.framework_agent_specialist_candidate_map = {}
    shared.gaps = []

    asyncio.run(stub._maybe_dispatch_local_explore(reason="discover_exhausted"))

    assert len(stub.tasks.created) == 1
    params = stub.tasks.created[0]["params"]
    gap_cid = params["gap_canonical_id"]
    resolved = shared.find_gap(gap_cid)
    assert resolved is not None, f"find_gap({gap_cid!r}) returned None; gap was not registered"
    assert resolved["layer"] == "framework"
