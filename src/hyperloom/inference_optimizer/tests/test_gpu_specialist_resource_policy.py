# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A GPU specialist is admitted against the pool it will lease from, at the count the dispatcher leases."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.bus.gpu_pool import SpecialistGpuPool
from hyperloom.orchestrator.loop._admission import SPECIALIST_GPU_REQUEST_EXCEEDS_CAPACITY
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.dispatcher import HeldLeases
from hyperloom.orchestrator.policy.projection import resource_pools_summary
from hyperloom.orchestrator.state.task_registry import TaskSpec

from .conftest import silent_backends


@pytest.fixture
async def coord(session_dir):
    c = Coordinator(session_dir, backends=silent_backends())
    c.shared_state.phase = "FRAMEWORK_AGENT"
    yield c
    await c.stop()


def _pools(c: Coordinator, *, serving_disjoint: int = 0, whole_machine: int = 0, serving_tp: int = 0) -> None:
    """Give the session pools of the stated sizes; the whole-machine pool starts at card 0, as serving does."""
    c.shared_state.tp = serving_tp
    c.gpu_specialist_pool = SpecialistGpuPool(c.db, gpu_ids=range(serving_tp, serving_tp + serving_disjoint))
    c.framework_gpu_pool = SpecialistGpuPool(c.db, gpu_ids=range(whole_machine))


def _specialist(**params) -> dict:
    return {"tags": ["kernel_agent"], "gap_canonical_id": "gap.kernel.microbench.session-test", **params}


async def _delegate(c: Coordinator, **params) -> None:
    payload = {"action_name": "specialist", "params": _specialist(**{"mode": "research", **params})}
    await c.router.handle_intent("orchestration", Intent(type=IntentType.DELEGATE, payload=payload))


async def _gpu_denials(c: Coordinator) -> list[dict]:
    return [
        m.payload
        for m in await c.bus.tail(topic="observation")
        if m.payload.get("kind") == "policy_denied" and m.payload.get("rule") == SPECIALIST_GPU_REQUEST_EXCEEDS_CAPACITY
    ]


async def test_a_request_within_the_pool_is_queued(coord):
    _pools(coord, serving_disjoint=2)

    await _delegate(coord, needs_gpu=True, gpu_count=2)

    assert len(await coord.tasks.by_state("queued")) == 1
    assert await _gpu_denials(coord) == []


async def test_a_request_above_the_pool_is_denied_and_leaves_no_task(coord):
    """The pool answers a busy acquire and an impossible one alike, so the row would wait for good."""
    _pools(coord, serving_disjoint=1)

    await _delegate(coord, needs_gpu=True, gpu_count=2)

    assert await coord.tasks.by_state("queued") == []
    (denial,) = await _gpu_denials(coord)
    assert "effective gpu_count=2" in denial["reason"]
    assert "serving-disjoint GPU specialist pool size=1" in denial["reason"]


async def test_an_empty_pool_is_denied_in_words_no_smaller_count_answers(coord):
    """Serving owns every card: the denial says the pool is empty rather than that the count is too large."""
    coord.shared_state.gpu_specialist_capacity = 4
    _pools(coord, serving_disjoint=0, whole_machine=4, serving_tp=4)

    await _delegate(coord, needs_gpu=True, gpu_count=1)

    assert await coord.tasks.by_state("queued") == []
    (denial,) = await _gpu_denials(coord)
    assert "holds no card (configured capacity=4, serving_tp=4)" in denial["reason"]
    assert "gpu_count=" not in denial["reason"]
    assert "No gpu_count fits an empty pool" in denial["hint"]


async def test_a_count_the_pool_can_never_lease_is_denied_whatever_its_sign(coord):
    _pools(coord, serving_disjoint=2)

    await _delegate(coord, needs_gpu=True, gpu_count=-1)

    assert await coord.tasks.by_state("queued") == []
    assert len(await _gpu_denials(coord)) == 1


async def test_the_pool_is_read_when_the_intent_is_judged(coord):
    """A request refused against one pool is admitted once the session holds a pool that funds it."""
    _pools(coord, serving_disjoint=1)
    await _delegate(coord, needs_gpu=True, gpu_count=2)
    assert await coord.tasks.by_state("queued") == []

    _pools(coord, serving_disjoint=2)
    await _delegate(coord, needs_gpu=True, gpu_count=2)

    assert len(await coord.tasks.by_state("queued")) == 1


async def test_an_unfundable_request_is_what_a_specialist_is_told_in_a_phase_that_offers_none(coord):
    """PRELUDE offers no specialist, and the request is the refusal no later phase lifts."""
    coord.shared_state.phase = "PRELUDE"
    _pools(coord, serving_disjoint=1)

    await _delegate(coord, needs_gpu=True, gpu_count=2)

    denied = [m.payload for m in await coord.bus.tail(topic="observation") if m.payload.get("kind") == "policy_denied"]
    assert [d["rule"] for d in denied] == [SPECIALIST_GPU_REQUEST_EXCEEDS_CAPACITY]


_FREEFORM = {"scope": "freeform", "task_description": "micro-bench the decode attention kernel"}
_DOMAIN = {"domain": "serving_specialist", "gap_canonical_id": "gap.framework.x.session-1"}
_BENCH = {**_FREEFORM, "mode": "patch", "bench": True}


@pytest.mark.parametrize(
    ("params", "pools", "denied_with"),
    [
        pytest.param(
            {**_FREEFORM, "needs_gpu": True, "gpu_count": 2}, {"serving_disjoint": 2}, None, id="freeform-fits"
        ),
        pytest.param(
            {**_FREEFORM, "needs_gpu": True, "gpu_count": 4},
            {"serving_disjoint": 1},
            "cannot be leased from the serving-disjoint GPU specialist pool size=1",
            id="freeform-above",
        ),
        pytest.param({**_FREEFORM, "needs_gpu": True}, {}, "holds no card", id="freeform-empty-pool"),
        pytest.param({**_DOMAIN, "needs_gpu": True}, {}, "holds no card", id="domain-empty-pool"),
        pytest.param(_BENCH, {}, "the whole-machine GPU pool holds no card", id="bench-implies-a-request"),
        pytest.param(
            {**_BENCH, "gpu_count": 1}, {"whole_machine": 4, "serving_tp": 4}, None, id="bench-shares-a-full-node"
        ),
        pytest.param(
            _BENCH,
            {"whole_machine": 2, "serving_tp": 4},
            "effective gpu_count=4 cannot be leased from the whole-machine GPU pool size=2",
            id="bench-node-smaller-than-tp",
        ),
        pytest.param(_BENCH, {"whole_machine": 8, "serving_tp": 4}, None, id="bench-omitted-count"),
        pytest.param({**_FREEFORM, "mode": "research"}, {}, None, id="research-needs-no-card"),
    ],
)
def test_every_scope_is_judged_against_the_pool_it_leases_from(coord, params, pools, denied_with):
    """Freeform is not a way around the pool, and a bench specialist needs no ``needs_gpu`` to be counted."""
    _pools(coord, **pools)

    denied = coord.dispatcher.admission_denial_for_action("specialist", params=params)

    if denied_with is None:
        assert denied is None
    else:
        assert denied is not None and denied.rule == SPECIALIST_GPU_REQUEST_EXCEEDS_CAPACITY
        assert denied_with in str(denied)


def test_an_enablement_specialist_is_judged_only_where_it_is_leased_cards(coord, monkeypatch):
    """Enablement takes the machine on a single node that shows cards, and nothing where it does not."""
    _pools(coord, whole_machine=2)
    params = {"enablement": True, "domain": "enablement_specialist", "gap_canonical_id": "gap.x", "gpu_count": 4}
    available = "hyperloom.orchestrator.specialists.profile._whole_machine_available"

    monkeypatch.setattr(available, lambda: True)
    denied = coord.dispatcher.admission_denial_for_action("specialist", params=params)
    assert denied is not None and denied.rule == SPECIALIST_GPU_REQUEST_EXCEEDS_CAPACITY

    monkeypatch.setattr(available, lambda: False)
    assert coord.dispatcher.specialist_gpu_request(params) is None
    assert coord.dispatcher.admission_denial_for_action("specialist", params=params) is None


def test_a_specialist_that_needs_no_card_is_not_sized(coord):
    _pools(coord)

    assert coord.dispatcher.specialist_gpu_request(_specialist(mode="research")) is None
    assert coord.dispatcher.admission_denial_for_action("specialist", params=_specialist(mode="research")) is None


def test_only_a_specialist_is_sized(coord):
    """Another action can carry the keys a specialist's request is read from; it leases no card from these pools."""
    _pools(coord)
    params = {"enablement": True, "needs_gpu": True}
    assert coord.dispatcher.specialist_gpu_request(params) is not None

    assert coord.dispatcher.admission_denial_for_action("integrate_patch", params=params) is None


_NO_COUNT_FITS = (
    "No gpu_count fits: a bench specialist takes at least serving TP cards (4) and the pool holds 2. "
    "Dispatch the work without bench=true."
)


@pytest.mark.parametrize(
    ("params", "pools", "hint", "admitted_counts"),
    [
        pytest.param(
            {**_FREEFORM, "needs_gpu": True, "gpu_count": 4},
            {"serving_disjoint": 2},
            "Pass a gpu_count between 1 and 2.",
            [1, 2],
            id="above-the-pool",
        ),
        pytest.param(
            {**_FREEFORM, "needs_gpu": True},
            {"serving_disjoint": 2, "serving_tp": 4},
            "Pass a gpu_count between 1 and 2.",
            [1, 2],
            id="default-count-above-the-pool",
        ),
        pytest.param(
            {**_BENCH, "gpu_count": 16},
            {"whole_machine": 8, "serving_tp": 4},
            "Pass a gpu_count between 4 and 8.",
            # A count below serving TP is admitted too, and leased at TP.
            [1, 2, 3, 4, 5, 6, 7, 8],
            id="bench-above-the-pool",
        ),
        pytest.param(
            {**_BENCH, "gpu_count": 1}, {"whole_machine": 2, "serving_tp": 4}, _NO_COUNT_FITS, [], id="bench-floor"
        ),
        pytest.param(
            {**_FREEFORM, "needs_gpu": True, "bench": True, "gpu_count": 1},
            {"serving_disjoint": 2, "serving_tp": 4},
            _NO_COUNT_FITS,
            [],
            id="research-bench-floor",
        ),
    ],
)
def test_the_hint_names_a_count_only_when_one_would_be_admitted(coord, params, pools, hint, admitted_counts):
    """A range the bench floor overrides sends the model back with a smaller count that is refused again."""
    _pools(coord, **pools)

    def denial(request: dict):
        return coord.dispatcher.admission_denial_for_action("specialist", params=request)

    denied = denial(params)

    assert denied is not None and denied.rule == SPECIALIST_GPU_REQUEST_EXCEEDS_CAPACITY
    assert denied.hint == hint
    assert [count for count in range(1, 17) if denial({**params, "gpu_count": count}) is None] == admitted_counts


@pytest.mark.parametrize(
    ("params", "pools", "expected_pool", "expected_count"),
    [
        pytest.param({"needs_gpu": True}, {"serving_disjoint": 4, "serving_tp": 2}, "disjoint", 2, id="default-is-tp"),
        pytest.param({"needs_gpu": True}, {"serving_disjoint": 4}, "disjoint", 1, id="default-without-tp"),
        pytest.param({"needs_gpu": True, "gpu_count": 0}, {"serving_disjoint": 4}, "disjoint", 1, id="zero-is-default"),
        pytest.param({"needs_gpu": True, "gpu_count": "3"}, {"serving_disjoint": 4}, "disjoint", 3, id="explicit"),
        pytest.param(
            {"mode": "patch", "bench": True, "gpu_count": 1},
            {"whole_machine": 8, "serving_tp": 4},
            "whole_machine",
            4,
            id="bench-floors-to-tp",
        ),
        pytest.param({"mode": "patch", "bench": True}, {"whole_machine": 8}, "whole_machine", 8, id="bench-without-tp"),
        pytest.param(
            {"enablement": True, "framework_agent_authoring": True},
            {"whole_machine": 8, "serving_tp": 4},
            "whole_machine",
            8,
            id="authoring-takes-the-machine",
        ),
    ],
)
def test_the_request_names_the_pool_and_count_the_dispatcher_leases(
    coord, monkeypatch, params, pools, expected_pool, expected_count
):
    monkeypatch.setattr("hyperloom.orchestrator.specialists.profile._whole_machine_available", lambda: True)
    _pools(coord, **pools)

    request = coord.dispatcher.specialist_gpu_request(_specialist(**params))

    expected = coord.framework_gpu_pool if expected_pool == "whole_machine" else coord.gpu_specialist_pool
    assert request.pool is expected
    assert request.count == expected_count


async def test_the_dispatcher_leases_the_request_admission_judged(coord, monkeypatch):
    """One sizing: the cards a specialist is given are the ones its admission counted."""
    monkeypatch.setattr("hyperloom.orchestrator.specialists.profile._whole_machine_available", lambda: True)
    _pools(coord, whole_machine=8, serving_tp=4)
    params = _specialist(mode="patch", bench=True, gpu_count=1)
    assert coord.dispatcher.admission_denial_for_action("specialist", params=params) is None
    task, _ = await coord.tasks.create_or_return_existing(
        TaskSpec(kind="specialist", params=params, idempotency_key="bench-specialist")
    )
    extra_context: dict = {}

    held = await coord.dispatcher._admit_gpu_specialist(task, None, extra_context)

    assert isinstance(held, HeldLeases)
    assert extra_context["gpu_ids"] == [0, 1, 2, 3]
    await coord.framework_gpu_pool.release(held.gpu_lease)


def test_the_prompt_block_prints_the_sizes_a_request_is_admitted_against(coord):
    """The model is told to size a request from this block, so it shows the pools the rule reads."""
    coord.shared_state.gpu_specialist_capacity = 6
    _pools(coord, serving_disjoint=2, whole_machine=6, serving_tp=4)

    block = resource_pools_summary(
        coord.shared_state,
        serving_disjoint_pool=coord.gpu_specialist_pool,
        whole_machine_pool=coord.framework_gpu_pool,
    )

    assert block.splitlines()[:4] == [
        "serving_tp=4",
        "gpu_specialist_capacity=6",
        "serving_disjoint_gpu_pool=2  (non-bench needs_gpu specialists admit against this)",
        "whole_machine_gpu_pool=6  (bench / enablement-authoring specialists admit against this)",
    ]


def test_a_pool_funds_only_a_count_it_could_ever_lease(tmp_path):
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    db = SqliteConnection(tmp_path / "coordinator.db")
    pool = SpecialistGpuPool(db, gpu_ids=[4, 5])

    assert [pool.can_fund(n) for n in (-1, 0, 1, 2, 3)] == [False, False, True, True, False]
    db.close()
