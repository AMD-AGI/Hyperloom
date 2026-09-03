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
    # How the two work phases divide their share is a tuning call; that the
    # session is spent on them rather than on setup and wind-down is not.
    work = budget[_phase_state.PHASE_FRAMEWORK_AGENT] + budget[_phase_state.PHASE_KERNEL_AGENT]
    overhead = budget[_phase_state.PHASE_PRELUDE] + budget[_phase_state.PHASE_CLOSE]
    assert work >= 0.8
    assert overhead <= 0.1


@pytest.fixture(autouse=True)
def _no_predictor(monkeypatch):
    """Take the predictor out of the picture for the whole module.

    An operator's shell often carries these -- the launch env exports them --
    and a configured predictor holds the LLM specialists back, which is the
    first rung of the ladder these tests walk. Without this the module passes
    or fails depending on whose terminal it ran in.
    """
    from hyperloom.orchestrator.predictor import config as predictor_config

    for name in (
        predictor_config.ENV_ENDPOINT,
        predictor_config.ENV_MODE,
        predictor_config.ENV_MAX_CHAIN,
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_each_discovery_retry_takes_a_fresh_idempotency_key(tmp_path: Path):
    """Retries must not collide on one key, or the streak can never advance.

    The registry hands back whatever row a key already names. With a fixed key
    the second request re-fetches the finished first attempt, no discovery
    runs, ``framework_agent_empty_discoveries`` stays at 1, and the lane
    answers "asked again" on every tick forever -- the source arm never goes
    dry and the phase never leaves.
    """
    stub = _Stub(tmp_path, authoring=True, local_explore=False)
    keys = []
    for empties in range(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT):
        stub.shared_state.framework_agent_empty_discoveries = empties
        assert asyncio.run(stub._maybe_enqueue_candidate_discovery(reason="candidate_pool_empty")) is True
        keys.append(stub.tasks.created[-1]["idempotency_key"])
        # The dispatched round settles: nothing is in flight, so the next tick
        # asks again and the key is all that stands between a real retry and a
        # re-fetch of the finished one.
        stub.tasks._queued.clear()
    assert len(set(keys)) == len(keys)

    # Budget spent: the lane declines so the rungs below it are reachable.
    stub.shared_state.framework_agent_empty_discoveries = _fa_client.DISCOVER_FAILURE_RETRY_LIMIT
    assert asyncio.run(stub._maybe_enqueue_candidate_discovery(reason="candidate_pool_empty")) is False


def test_a_failed_discovery_round_is_not_an_empty_one(tmp_path: Path):
    """A crashed specialist reports nothing about what is out there.

    Counting it toward the empty streak walks the source arm to "exhausted" on
    the strength of its own failures, and the signal is available at the call
    site: the dispatcher already reads the task error for the sibling bridge.
    """
    stub = _Stub(tmp_path, authoring=True, local_explore=False)
    task = SimpleNamespace(task_id="t1", params={"candidate_discovery": True})

    stub._ingest_candidate_discovery(task=task, done_payload={}, run_error="specialist died")
    assert stub.shared_state.framework_agent_empty_discoveries == 0

    # A round that completed and genuinely found nothing still counts.
    stub._ingest_candidate_discovery(task=task, done_payload={"proposal_set": []})
    assert stub.shared_state.framework_agent_empty_discoveries == 1


def test_a_registry_that_cannot_answer_is_not_an_idle_one(tmp_path: Path):
    """The pump must not read a failed task query as "nothing in flight".

    Treating it as idle dispatches a second candidate on top of a live one.
    """
    stub = _Stub(tmp_path, authoring=True, local_explore=True)

    async def _boom():
        raise RuntimeError("registry unavailable")

    stub.tasks.queued = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        asyncio.run(stub._pump_framework_agent_phase())


def test_a_lane_that_cannot_run_retires_on_its_own_budget(tmp_path: Path):
    """Failed rounds get their own counter, a fresh key, and the same limit.

    Not counting a failure toward the empty streak is right -- it found
    nothing because it never looked -- but it must still count somewhere. With
    neither counter moving, the idempotency key never changes, so every retry
    re-fetches the settled failure and the lane asks forever without running.
    """
    stub = _Stub(tmp_path, authoring=True, local_explore=False)
    task = SimpleNamespace(task_id="t1", params={"candidate_discovery": True})
    keys = []

    for _ in range(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT):
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


def test_a_commit_failure_after_a_keep_does_not_rearm_the_author(tmp_path: Path):
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

    stub._maybe_rearm_authored_lane(res)
    assert stub.shared_state.apply_fail_retry_pending == []

    # A genuine apply failure on the same lane still queues a retry.
    stub._maybe_rearm_authored_lane({**res, "status": "apply_failed", "error_class": "patch_did_not_apply"})
    assert len(stub.shared_state.apply_fail_retry_pending) == 1


@pytest.mark.asyncio
async def test_an_apply_retry_is_scoped_to_the_cycle_that_queued_it(tmp_path: Path):
    """The gap id and the attempt number both repeat across macro-cycles.

    Without the cycle scope, a second cycle's first retry for the same gap
    returns the settled task from the first cycle, and the re-author it asked
    for never runs.
    """
    stub = _Stub(tmp_path, authoring=True, local_explore=True)

    await stub._enqueue_author_specialist(lane="perf_explore", specialist_task_id="t-spec")
    stub.shared_state.macro_cycle = 2
    await stub._enqueue_author_specialist(lane="perf_explore", specialist_task_id="t-spec")

    keys = [c["idempotency_key"] for c in stub.tasks.created]
    assert len(set(keys)) == 2
    assert keys[1].endswith("-c2")


@pytest.mark.asyncio
async def test_the_retry_drain_carries_the_batch_the_failure_belonged_to(tmp_path: Path):
    """The candidate row does not always carry the batch the round ran under.

    The rearm resolves it from the executor result and records it; the drain
    dropped it, so the re-dispatched specialist keyed its idempotency on an
    empty batch and collided with every other batch-less retry.
    """
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
